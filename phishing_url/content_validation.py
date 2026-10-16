"""Keep the previous URL holdout intact when adding organizer HTML."""

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def aligned_splits(root, cache):
    root = Path(root)
    with (root / "reports/clean_train.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    lookup = {url: i for i, url in enumerate(cache["urls"])}
    if len(lookup) != len(cache["urls"]):
        raise ValueError("Cache contains duplicate training URLs")
    mapping = np.asarray([lookup[row["url"]] for row in old_rows], dtype=np.int64)
    old_y = np.asarray([int(row["result"]) for row in old_rows], dtype=np.int8)
    if not np.array_equal(cache["labels"][mapping], old_y):
        raise ValueError("Original/HTML labels disagree")
    original = np.load(root / "reports/split_indices.npz")
    development = mapping[original["development"]]
    holdout = mapping[original["holdout"]]
    added = np.asarray(sorted(set(range(len(cache["urls"]))) - set(mapping.tolist())), dtype=np.int64)
    screening_train, screening_validation = train_test_split(
        development, test_size=.2, stratify=cache["labels"][development], random_state=20261006)
    inner_train, inner_validation = train_test_split(
        screening_train, test_size=.15, stratify=cache["labels"][screening_train], random_state=18428)
    return {"old_to_cache": mapping, "development": development, "holdout": holdout,
            "added": added, "screening_train": screening_train,
            "screening_validation": screening_validation, "inner_train": inner_train,
            "inner_validation": inner_validation}


def records_at(cache, indices):
    return [cache["records"][int(i)] for i in indices]
