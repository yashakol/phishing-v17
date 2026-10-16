"""Offline compound template evidence. No classifier, URL visits or JS execution.

fit() accepts training labels only. Selection never operates on individual keys.
JSON serialization retains each accepted key's training-row provenance.
"""
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from urllib.parse import urlsplit, parse_qsl, unquote

FAMILIES = ("title_js", "title_css", "title_resources", "markup_js", "markup_css", "markup_resources")
SPACE = re.compile(r"\s+")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
URL = re.compile(r"(?:https?:)?//[^\s\"'<>`\\)\]}]+", re.I)
DATA = re.compile(r"data:[^\s\"'<>`]+", re.I)
UUID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
HEX = re.compile(r"(?<![a-z0-9])[a-f0-9]{12,}(?![a-z0-9])", re.I)
OPAQUE = re.compile(r"[a-z0-9_+/=-]{40,}", re.I)
DIGITS = re.compile(r"\d+")
VARIABLE = re.compile(r'''(\b(?:nonce|csrf(?:token)?|xsrf(?:token)?|timestamp|session(?:id)?|user(?:name|id)?|email|token)\b[\"']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;{}<>]+)''', re.I)
QUERY = re.compile(r"\?[^\s\"'<>`\\)\]}]+")
WORDS = re.compile(r"[a-z_][a-z_-]{2,}", re.I)
TAG = re.compile(r"\btag_[a-z][a-z0-9]*\b")
RESOURCE = re.compile(r"(?:^|\s)(?:script\s+(?!src\b)|(?:link|iframe|form|img|base|input)\s+(?:href|src|action)\s+)")
BOILERPLATE = re.compile(r"^(?:home|index|untitled|welcome|loading|error|404|403|access denied|just a moment|attention required|please wait|page not found|not found|redirecting)[.!\s]*$", re.I)
PLACEHOLDERS = {"var", "num", "email", "opaque", "hex", "uuid", "host", "data", "query", "longtoken"}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query(value):
    # Keep parameter names, discard all request/user/version-specific values.
    try:
        names = sorted({DIGITS.sub("NUM", key.lower()) for key, _ in parse_qsl(value, keep_blank_values=True, max_num_fields=256)})
    except ValueError:
        return "?QUERY"
    return "?" + "&".join(name + "=VAR" for name in names) if names else ""


def _url(match):
    try:
        parts = urlsplit(match.group(0) if not match.group(0).startswith("//") else "https:" + match.group(0))
        return "HOST" + parts.path + _query(parts.query)
    except ValueError:
        return "HOST"


def normalize(value):
    if not isinstance(value, str):
        raise TypeError("Expected cached static text")
    # The caches already bound length. These transformations never evaluate text.
    value = html.unescape(unquote(value)).lower().replace("\\/", "/")
    value = DATA.sub("DATA", value)
    value = EMAIL.sub("EMAIL", value)
    value = VARIABLE.sub(lambda m: m.group(1) + '"VAR"', value)
    value = URL.sub(_url, value)
    value = QUERY.sub(lambda m: _query(m.group(0)[1:]), value)
    value = UUID.sub("UUID", value)
    value = HEX.sub("HEX", value)
    value = OPAQUE.sub("OPAQUE", value)
    value = DIGITS.sub("NUM", value)
    return SPACE.sub(" ", value).strip()


def substantial(value, minimum, unique_words):
    words = {w.lower() for w in WORDS.findall(value)} - PLACEHOLDERS
    return len(value) >= minimum and len(words) >= unique_words and sum(map(len, words)) >= minimum // 4


def signature_keys(record, extra):
    if record["url"] != extra["url"]:
        raise ValueError("Content and extra URLs differ")
    title = normalize(record.get("title", ""))
    markup = normalize(record.get("markup", ""))
    skeleton = " ".join(TAG.findall(markup))
    js = normalize(extra.get("scripts", ""))
    css = normalize(extra.get("styles", ""))
    resources = normalize(extra.get("resources", ""))
    title_ok = len(title) >= 5 and len(re.sub(r"[^a-z]", "", title.lower())) >= 5 and not BOILERPLATE.fullmatch(title)
    markup_ok = substantial(markup, 320, 12) and len(TAG.findall(markup)) >= 20
    skeleton_ok = len(TAG.findall(skeleton)) >= 30 and len(set(TAG.findall(skeleton))) >= 6
    js_ok = substantial(js, 320, 20)
    css_ok = substantial(css, 256, 16)
    resource_ok = substantial(resources, 120, 10) and len(RESOURCE.findall(resources)) >= 3
    payloads = {}
    # All keys combine independent views. No standalone normalized JS/CSS key.
    if title_ok and js_ok: payloads["title_js"] = (title, js)
    if title_ok and css_ok: payloads["title_css"] = (title, css)
    if title_ok and resource_ok and skeleton_ok: payloads["title_resources"] = (title, resources, skeleton)
    if markup_ok and js_ok: payloads["markup_js"] = (markup, js)
    if markup_ok and css_ok: payloads["markup_css"] = (markup, css)
    if markup_ok and resource_ok: payloads["markup_resources"] = (markup, resources)
    return {family: family + ":" + digest(json.dumps(parts, ensure_ascii=True, separators=(",", ":"))) for family, parts in payloads.items()}


class TemplateEvidence:
    def __init__(self, families=FAMILIES, minimum_support=3, minimum_domains=2, weight=.85, confidence_limit=.98):
        self.config = dict(families=list(families), minimum_support=minimum_support, minimum_domains=minimum_domains, weight=weight, confidence_limit=confidence_limit)
        if not families or not set(families) <= set(FAMILIES): raise ValueError("Unknown families")
        if minimum_support < 3 or minimum_domains < 2: raise ValueError("Unsafe support")
        if not 0 < weight <= 1 or confidence_limit not in (.98, 1.): raise ValueError("Invalid confidence policy")

    def fit(self, keys, urls, domains, labels, row_ids):
        if not len(keys) == len(urls) == len(domains) == len(labels) == len(row_ids): raise ValueError("Alignment mismatch")
        if len(set(urls)) != len(urls) or len(set(row_ids)) != len(row_ids): raise ValueError("Duplicate training rows/URLs")
        self.evidence = {}
        self.ledger = {}
        self.statistics = {}
        frequencies = defaultdict(lambda: {"counts": [0, 0], "rows": [], "domains": set(), "url_hashes": []})
        for row, url, domain, label, rid in zip(keys, urls, domains, labels, row_ids, strict=True):
            if int(label) not in (0, 1): raise ValueError("Nonbinary training label")
            for family in self.config["families"]:
                key = row.get(family)
                if key is None: continue
                entry = frequencies[key]
                entry["counts"][int(label)] += 1
                entry["rows"].append(int(rid))
                entry["url_hashes"].append(digest(url))
                if domain and domain != "INVALID": entry["domains"].add(digest(domain))
        for family in self.config["families"]:
            self.statistics[family] = dict(observed_keys=0, conflicting_keys=0, insufficient_urls=0, insufficient_domains=0, accepted_keys=0, accepted_training_rows=0)
        for key, group in sorted(frequencies.items()):
            negative, positive = group["counts"]
            support, domain_count = negative + positive, len(group["domains"])
            reason = "conflict" if negative and positive else "urls" if support < self.config["minimum_support"] else "domains" if domain_count < self.config["minimum_domains"] else "accepted"
            stats = self.statistics[key.split(":")[0]]
            stats["observed_keys"] += 1
            stats[{"conflict": "conflicting_keys", "urls": "insufficient_urls", "domains": "insufficient_domains", "accepted": "accepted_keys"}[reason]] += 1
            entry = dict(label=int(positive > 0) if reason != "conflict" else None, counts=group["counts"], support=support, domain_count=domain_count, training_indices=group["rows"], training_url_sha256=group["url_hashes"], training_domain_sha256=sorted(group["domains"]), status=reason)
            self.ledger[key] = entry
            if reason == "accepted":
                stats["accepted_training_rows"] += support
                self.evidence[key] = entry
        self.training_rows = len(keys)
        return self

    def predict_keys(self, keys, probabilities, hard_mask=None):
        import numpy as np
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(keys),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise ValueError("Invalid baseline")
        hard = np.zeros(len(keys), bool) if hard_mask is None else np.asarray(hard_mask, bool)
        if hard.shape != p.shape: raise ValueError("Invalid hard mask")
        out = p.copy()
        hard = hard | (p == 0) | (p == 1)
        limit = self.config["confidence_limit"]
        protected = hard | (p < 1 - limit) | (p > limit)
        flags = {k: np.zeros(len(keys), bool) for k in ("matched", "conflict", "adjusted", "protected_match")}
        flags["hard_protected"] = hard
        matches = []
        for i, row in enumerate(keys):
            hit = sorted({row[f] for f in self.config["families"] if f in row and row[f] in self.evidence})
            matches.append(hit)
            labels = {self.evidence[k]["label"] for k in hit}
            flags["matched"][i] = bool(hit)
            flags["conflict"][i] = len(labels) > 1
            flags["protected_match"][i] = bool(hit) and protected[i]
            if len(labels) == 1 and not protected[i]:
                out[i] = (1 - self.config["weight"]) * p[i] + self.config["weight"] * next(iter(labels))
                flags["adjusted"][i] = True
        assert np.array_equal(out[hard], p[hard])
        return out, flags, matches

    def predict(self, records, extras, probabilities, hard_mask=None):
        if len(records) != len(extras): raise ValueError("Record alignment mismatch")
        return self.predict_keys([signature_keys(r, e) for r, e in zip(records, extras, strict=True)], probabilities, hard_mask)

    def to_dict(self):
        return dict(format_version=1, config=self.config, training_rows=self.training_rows, evidence=self.evidence, statistics=self.statistics)

    @classmethod
    def from_dict(cls, value):
        if value["format_version"] != 1: raise ValueError("Unknown model format")
        obj = cls(**value["config"])
        obj.training_rows, obj.evidence, obj.statistics = value["training_rows"], value["evidence"], value["statistics"]
        for key, entry in obj.evidence.items():
            if key.split(":")[0] not in obj.config["families"]: raise ValueError("Unknown evidence family")
            if entry["status"] != "accepted" or entry["support"] < obj.config["minimum_support"] or entry["domain_count"] < obj.config["minimum_domains"] or min(entry["counts"]) != 0: raise ValueError("Invalid saved evidence")
        return obj
