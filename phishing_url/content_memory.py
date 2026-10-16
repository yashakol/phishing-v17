"""Use unambiguous page matches from the fitted training partition only."""

import hashlib
from collections import Counter, defaultdict

import numpy as np

from .features import group_for_url


def text_fingerprint(record):
    # URL itself is excluded: a copied phishing page may move between hosts.
    source = "\n".join(record[key].lower() for key in ("title", "text", "markup"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def fit_memory(records, hashes, labels, minimum_support=2, domain_minimum_support=0, fingerprints=None):
    if not (len(records) == len(hashes) == len(labels)):
        raise ValueError("Content memory inputs must align")
    counts = defaultdict(Counter)
    if fingerprints is None:
        fingerprints = [text_fingerprint(r) for r in records]
    for record, raw_hash, label, fingerprint in zip(records, hashes, labels, fingerprints, strict=True):
        if minimum_support:
            counts[("raw", raw_hash)][int(label)] += 1
            counts[("text", fingerprint)][int(label)] += 1
        if domain_minimum_support:
            counts[("domain", group_for_url(record["url"]))][int(label)] += 1
    return {key: next(iter(frequencies)) for key, frequencies in counts.items()
            if len(frequencies) == 1 and sum(frequencies.values()) >=
            (domain_minimum_support if key[0] == "domain" else minimum_support)}


def apply_memory(memory, records, hashes, probabilities, fingerprints=None):
    if not (len(records) == len(hashes) == len(probabilities)):
        raise ValueError("Content memory prediction inputs must align")
    result = np.asarray(probabilities, dtype=float).copy()
    hits = np.zeros(len(records), dtype=bool)
    if fingerprints is None:
        fingerprints = [text_fingerprint(r) for r in records]
    for i, (record, raw_hash, fingerprint) in enumerate(zip(records, hashes, fingerprints, strict=True)):
        page_labels = {memory[key] for key in (("raw", raw_hash), ("text", fingerprint)) if key in memory}
        labels = page_labels.copy()
        domain = ("domain", group_for_url(record["url"]))
        # Use domain-only evidence for uncertain predictions; confident conflicts
        # retain the learned model. Page/domain disagreements also retain it.
        if domain in memory and (page_labels or .05 <= result[i] <= .95):
            labels.add(memory[domain])
        # Different signatures may disagree; keep the learned probability then.
        if len(labels) == 1:
            result[i] = next(iter(labels))
            hits[i] = True
    return result, hits
