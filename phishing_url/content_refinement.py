"""Training-only exact and normalized content evidence, without fetching pages."""

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from urllib.parse import urlsplit

import numpy as np

from .content_memory import text_fingerprint
from .features import group_for_url


REFINEMENT_VERSION = 1
MODES = ("raw", "compact", "visible", "normalized", "layout", "host", "domain")
PAGE_MODES = MODES[:5]
SPACE = re.compile(r"\s+")
URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I)
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[a-z]{2,}\b", re.I)
HEX = re.compile(r"\b[a-f0-9]{12,}\b", re.I)
DIGITS = re.compile(r"\d+")
TAGS = re.compile(r"(?:^|\s)(tag_[a-z][a-z0-9]*)\b")
TYPES = re.compile(r"(?:^|\s)(type_[a-z]+|method_[a-z]+)\b")


def normalized_text(value):
    value = unicodedata.normalize("NFKC", value).lower()
    value = URL.sub(" URL ", value)
    value = EMAIL.sub(" EMAIL ", value)
    value = HEX.sub(" HEX ", value)
    value = DIGITS.sub(" NUM ", value)
    return SPACE.sub(" ", value).strip()


def signature_digest(value):
    if len(value) < 40:
        return None
    letters = 0
    for character in value:
        letters += character.isalpha()
        if letters >= 10:
            break
    if letters < 10:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def signature_keys(records, hashes):
    if len(records) != len(hashes):
        raise ValueError("Refinement records and hashes must align")
    keys = {mode: [] for mode in MODES}
    for record, raw in zip(records, hashes, strict=True):
        visible = SPACE.sub(" ", record["title"] + "\n" + record["text"]).strip().lower()
        normalized = normalized_text(visible)
        layout = " ".join(TAGS.findall(record["markup"]) + TYPES.findall(record["markup"]))
        try:
            host = (urlsplit(record["url"]).hostname or "").lower()
        except ValueError:
            host = ""
        keys["raw"].append(raw)
        keys["compact"].append(text_fingerprint(record))
        keys["visible"].append(signature_digest(visible))
        keys["normalized"].append(signature_digest(normalized))
        # Keep visible content in a layout signature; bare tag sequences are weak evidence.
        keys["layout"].append(signature_digest(normalized + "\n" + layout))
        keys["host"].append(host or None)
        keys["domain"].append(group_for_url(record["url"]))
    return keys


def fit_evidence(keys, labels, supports):
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Refinement labels must be binary")
    result = {}
    for mode, minimum in supports.items():
        if mode not in MODES or minimum < 1:
            raise ValueError("Unknown signature or invalid minimum support")
        if len(keys[mode]) != len(labels):
            raise ValueError("Refinement keys and labels must align")
        counts = defaultdict(Counter)
        for key, label in zip(keys[mode], labels, strict=True):
            if key is not None:
                counts[key][int(label)] += 1
        result[mode] = {key: next(iter(count)) for key, count in counts.items()
                        if len(count) == 1 and sum(count.values()) >= minimum}
    return result


def evidence_values(memory, keys):
    return {mode: np.asarray([lookup.get(key, -1) if key is not None else -1
                             for key in keys[mode]], dtype=np.int8)
            for mode, lookup in memory.items()}


def apply_values(probabilities, values, policy="consensus", confidence_floor=0., domain_floor=.05):
    """Apply pure training groups, with either consensus or stronger-page priority."""
    p = np.asarray(probabilities, dtype=float).copy()
    if policy not in {"consensus", "page_priority", "hierarchy"}:
        raise ValueError("Unknown refinement policy")
    if not 0 <= confidence_floor < .5 or not 0 <= domain_floor < .5:
        raise ValueError("Invalid refinement probability interval")
    page_zero = np.zeros(len(p), dtype=bool)
    page_one = np.zeros(len(p), dtype=bool)
    primary = np.full(len(p), -1, dtype=np.int8)
    for mode in PAGE_MODES:
        if mode in values:
            value = values[mode]
            if len(value) != len(p):
                raise ValueError("Evidence predictions must align")
            page_zero |= value == 0
            page_one |= value == 1
            take = (primary < 0) & (value >= 0)
            primary[take] = value[take]
    page_present = page_zero | page_one
    page_conflict = page_zero & page_one
    host_zero, host_one = np.zeros(len(p), bool), np.zeros(len(p), bool)
    for mode in ("host", "domain"):
        if mode in values:
            value = values[mode]
            if len(value) != len(p):
                raise ValueError("Evidence predictions must align")
            host_zero |= value == 0
            host_one |= value == 1
    host_eligible = (p >= domain_floor) & (p <= 1-domain_floor)
    page_eligible = (p >= confidence_floor) & (p <= 1-confidence_floor)
    if policy == "consensus":
        host_eligible |= page_present
        zero = page_zero | (host_zero & host_eligible)
        one = page_one | (host_one & host_eligible)
        take_zero = zero & ~one & page_eligible
        take_one = one & ~zero & page_eligible
    else:
        if policy == "hierarchy":
            take_zero = (primary == 0) & page_eligible
            take_one = (primary == 1) & page_eligible
        else:
            take_zero = page_zero & ~page_conflict & page_eligible
            take_one = page_one & ~page_conflict & page_eligible
        no_page = ~page_present
        take_zero |= no_page & host_eligible & host_zero & ~host_one
        take_one |= no_page & host_eligible & host_one & ~host_zero
    p[take_zero] = 0.
    p[take_one] = 1.
    return p, take_zero | take_one


def apply_evidence(memory, keys, probabilities, config):
    return apply_values(probabilities, evidence_values(memory, keys), config["policy"],
                        config["confidence_floor"], config["domain_floor"])
