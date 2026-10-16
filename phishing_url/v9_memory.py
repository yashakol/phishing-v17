"""Partition-fitted page/domain evidence with explicit support/conflict policy."""

from collections import defaultdict

import numpy as np

from .content_memory import text_fingerprint
from .features import group_for_url


DEFAULT_POLICY = {"page_support": 3, "domain_support": 5, "domain_confidence": .95,
                  "veto_support": 0, "veto_confidence": .9}


def validate_policy(policy):
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError("Unexpected memory policy fields")
    if any(type(policy[key]) is not int or policy[key] < minimum for key, minimum in
           (("page_support", 2), ("domain_support", 2), ("veto_support", 0))):
        raise ValueError("Invalid evidence support")
    if not .5 <= policy["domain_confidence"] <= 1 or not .5 <= policy["veto_confidence"] <= 1:
        raise ValueError("Invalid evidence confidence")


def memory_keys(records, hashes):
    if len(records) != len(hashes):
        raise ValueError("Page hashes must align")
    return {"raw": list(hashes), "text": [text_fingerprint(r) for r in records],
            "domain": [group_for_url(r["url"]) for r in records]}


def pure_values(counts, support):
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 2 or (counts < 0).any() or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Counts must be nonnegative integer pairs")
    zero, one = counts.T
    result = np.full(len(counts), -1, dtype=np.int8)
    result[(zero >= support) & (one == 0)] = 0
    result[(one >= support) & (zero == 0)] = 1
    return result


def apply_evidence(base, counts, policy):
    validate_policy(policy)
    base = np.asarray(base, dtype=float)
    if base.ndim != 1 or not np.isfinite(base).all() or (base < 0).any() or (base > 1).any():
        raise ValueError("Invalid baseline probabilities")
    if set(counts) != {"raw", "text", "domain"} or any(np.asarray(v).shape != (len(base), 2) for v in counts.values()):
        raise ValueError("Evidence must align with baseline rows")
    raw, text = [pure_values(counts[kind], policy["page_support"]) for kind in ("raw", "text")]
    domain = pure_values(counts["domain"], policy["domain_support"])
    page = np.where(raw >= 0, raw, text)
    page_present = page >= 0
    lower = 1-policy["domain_confidence"]
    # Match legacy .05 and .95 inclusive bounds without subtraction roundoff.
    if policy["domain_confidence"] == .95:
        lower = .05
    eligible = (domain >= 0) & (page_present | ((base >= lower) & (base <= policy["domain_confidence"])))
    conflict = ((raw >= 0) & (text >= 0) & (raw != text)) | (eligible & page_present & (domain != page))
    if policy["veto_support"]:
        hint = pure_values(counts["domain"], policy["veto_support"])
        agrees_with_learner = ((hint == 0) & (base <= 1-policy["veto_confidence"])) | \
                              ((hint == 1) & (base >= policy["veto_confidence"]))
        conflict |= page_present & (hint >= 0) & (hint != page) & agrees_with_learner
    result = base.copy()
    take_page = page_present & ~conflict
    take_domain = eligible & ~page_present & ~conflict
    result[take_page] = page[take_page]
    result[take_domain] = domain[take_domain]
    return result, take_page | take_domain


class MemoryEvidence:
    def __init__(self, policy):
        validate_policy(policy)
        self.policy = dict(policy)

    def fit_keys(self, keys, labels):
        y = np.asarray(labels)
        if set(keys) != {"raw", "text", "domain"} or any(len(v) != len(y) for v in keys.values()):
            raise ValueError("Evidence training rows must align")
        if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(y) == 0:
            raise ValueError("Binary training labels required")
        self.counts_ = {}
        for kind, values in keys.items():
            groups = defaultdict(lambda: [0, 0])
            for key, label in zip(values, y, strict=True):
                groups[key][int(label)] += 1
            self.counts_[kind] = {key: tuple(count) for key, count in groups.items()}
        self.training_rows_ = len(y)
        return self

    def fit(self, records, hashes, labels):
        if len({r["url"] for r in records}) != len(records):
            raise ValueError("Training URLs must be unique")
        return self.fit_keys(memory_keys(records, hashes), labels)

    def lookup(self, keys):
        if set(keys) != set(self.counts_) or len({len(v) for v in keys.values()}) != 1:
            raise ValueError("Evidence query keys must align")
        return {kind: np.asarray([self.counts_[kind].get(key, (0, 0)) for key in values], dtype=np.int32).reshape(-1, 2)
                for kind, values in keys.items()}

    def predict(self, records, hashes, base):
        return apply_evidence(base, self.lookup(memory_keys(records, hashes)), self.policy)
