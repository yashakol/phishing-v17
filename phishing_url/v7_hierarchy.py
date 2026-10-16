"""Fold-local supervised evidence for URL host/path families."""

import re
from collections import defaultdict
from urllib.parse import parse_qsl, unquote

import numpy as np

from .features import group_for_url, split_url


IDENTIFIER = re.compile(r"[a-f0-9]{12,}|\d{3,}", re.I)


def hierarchy_keys(url):
    parsed = split_url(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return {}
    domain = group_for_url(url)
    segments = [IDENTIFIER.sub("ID", unquote(part).lower()) for part in parsed.path.split("/") if part]
    keys = {"host": ("host", host)}
    for depth in (1, 2):
        if len(segments) >= depth:
            prefix = "/".join(segments[:depth])
            keys[f"host_path{depth}"] = (f"host_path{depth}", host, prefix)
            keys[f"domain_path{depth}"] = (f"domain_path{depth}", domain, prefix)
    query = tuple(sorted({name.lower() for name, _ in parse_qsl(parsed.query, keep_blank_values=True)}))
    if query:
        keys["host_query"] = ("host_query", host, query)
    shape = "/".join(segments)
    if len(segments) >= 2 and len(shape) >= 25:
        keys["global_path"] = ("global_path", shape)
    return keys


class HierarchicalEvidence:
    def __init__(self, families, minimum_support=3, alpha=5., weight=.6, confidence_limit=.98):
        self.families = tuple(families)
        self.minimum_support, self.alpha = int(minimum_support), float(alpha)
        self.weight, self.confidence_limit = float(weight), float(confidence_limit)
        if self.minimum_support < 2 or self.alpha < 0 or not 0 < self.weight <= 1:
            raise ValueError("Invalid hierarchy configuration")

    def fit_keys(self, keys, labels):
        labels = np.asarray(labels)
        if len(keys) != len(labels) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Training keys and binary labels must align")
        self.prior_ = float(labels.mean())
        counts = defaultdict(lambda: [0, 0])
        for row, label in zip(keys, labels, strict=True):
            for family in self.families:
                if family in row:
                    counts[row[family]][int(label)] += 1
        self.evidence_ = {key: (int(pos > 0), neg + pos, (pos + self.alpha * self.prior_) / (neg + pos + self.alpha))
                          for key, (neg, pos) in counts.items()
                          if not (neg and pos) and neg + pos >= self.minimum_support}
        self.training_rows_ = len(keys)
        return self

    def predict_keys(self, keys, base):
        base = np.asarray(base, dtype=float)
        if base.shape != (len(keys),) or not np.isfinite(base).all() or (base < 0).any() or (base > 1).any():
            raise ValueError("Baseline probabilities must align")
        result = base.copy()
        hits = np.zeros(len(base), dtype=bool)
        for index, row in enumerate(keys):
            p = base[index]
            if p == 0 or p == 1 or p < 1 - self.confidence_limit or p > self.confidence_limit:
                continue
            evidence = [self.evidence_[row[f]] for f in self.families if f in row and row[f] in self.evidence_]
            if not evidence or len({item[0] for item in evidence}) > 1:
                continue
            _, _, posterior = max(evidence, key=lambda item: item[1])
            result[index] = (1-self.weight) * p + self.weight * posterior
            hits[index] = True
        return result, hits

