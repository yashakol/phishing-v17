"""Training-only evidence from repeated static page components.

This module never loads a reference dataset or obtains labels for test URLs.
Empty/short boilerplate is excluded, contradictory training groups are ignored,
and an existing exact training rule is preserved during prediction.
"""

import hashlib
import re
from collections import defaultdict

import numpy as np

from .features import group_for_url


FAMILIES = ("scripts", "styles", "title_scripts", "title_styles", "title_resources", "title_markup")
SPACE = re.compile(r"\s+")


def signature_keys(record, extra):
    if record["url"] != extra["url"]:
        raise ValueError("Page and extra record URLs differ")
    title = SPACE.sub(" ", record.get("title", "").lower()).strip()
    fields = {name: SPACE.sub(" ", extra.get(name, "")).strip()
              for name in ("scripts", "styles", "resources")}
    fields["markup"] = SPACE.sub(" ", record.get("markup", "")).strip()
    output = {}
    for name in FAMILIES:
        field = name.removeprefix("title_")
        content = fields[field]
        minimum = 160 if field in ("scripts", "markup") else 100
        if len(content) < minimum or (name.startswith("title_") and len(title) < 5):
            continue
        source = title + "\x00" + content if name.startswith("title_") else content
        output[name] = name + ":" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    return output


class SignatureEvidence:
    def __init__(self, families, minimum_support=3, minimum_domains=1,
                 weight=.85, confidence_limit=.98):
        self.families = tuple(families)
        self.minimum_support = int(minimum_support)
        self.minimum_domains = int(minimum_domains)
        self.weight = float(weight)
        self.confidence_limit = float(confidence_limit)
        if not self.families or any(f not in FAMILIES for f in self.families):
            raise ValueError("Unknown or empty signature families")
        if self.minimum_support < 2 or self.minimum_domains < 1:
            raise ValueError("Evidence needs multiple training examples")
        if not 0 < self.weight <= 1 or not .5 <= self.confidence_limit <= 1:
            raise ValueError("Invalid evidence adjustment")

    def fit_keys(self, keys, urls, labels):
        labels = np.asarray(labels)
        if len(keys) != len(urls) or len(keys) != len(labels):
            raise ValueError("Training inputs must align")
        if len(set(urls)) != len(urls) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Training URLs must be unique with binary labels")
        frequencies = defaultdict(lambda: [0, 0])
        domains = defaultdict(set)
        for row, url, label in zip(keys, urls, labels, strict=True):
            domain = group_for_url(url)
            for family in self.families:
                key = row.get(family)
                if key:
                    frequencies[key][int(label)] += 1
                    domains[key].add(domain)
        self.evidence_ = {}
        self.conflicting_groups_ = 0
        for key, (negative, positive) in frequencies.items():
            if negative and positive:
                self.conflicting_groups_ += 1
                continue
            if negative + positive >= self.minimum_support and len(domains[key]) >= self.minimum_domains:
                self.evidence_[key] = (int(positive > 0), negative + positive, len(domains[key]))
        self.training_rows_ = len(keys)
        return self

    def predict_keys(self, keys, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(keys),) or not np.isfinite(probabilities).all() or \
                (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Invalid aligned baseline probabilities")
        output = probabilities.copy()
        hits = np.zeros(len(keys), dtype=bool)
        conflicts = np.zeros(len(keys), dtype=bool)
        for index, row in enumerate(keys):
            # Preserve previous exact training rules and highly confident answers.
            p = probabilities[index]
            if p == 0 or p == 1 or p < 1 - self.confidence_limit or p > self.confidence_limit:
                continue
            labels = {self.evidence_[row[f]][0] for f in self.families
                      if f in row and row[f] in self.evidence_}
            if len(labels) > 1:
                conflicts[index] = True
            elif len(labels) == 1:
                output[index] = (1 - self.weight) * p + self.weight * next(iter(labels))
                hits[index] = True
        return output, {"signature_match": hits, "signature_conflict": conflicts}

