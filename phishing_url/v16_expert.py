"""Broader own-trained specialists with optional raw categorical page features."""

import re

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .v12_expert import eligibility
from .v15_resources import KEYS

URL_KEYS = ("cat_scheme", "cat_host", "cat_domain", "cat_suffix", "cat_extension", "cat_first_path")
HTML_KEYS = ("cat_html_language", "cat_html_generator")
CATEGORY_KEYS = URL_KEYS + HTML_KEYS + KEYS + ("cat_normalized_title",)
KINDS = ("numeric_deep", "categorical")


def category_frame(cache, extra, url_frame):
    size = len(cache["urls"])
    if cache["urls"] != extra["urls"] or any(len(source) != size for source in
            (cache["records"], cache["features"], extra["features"], url_frame)):
        raise ValueError("Categorical rows must align")
    if any(record["url"] != url for record, url in zip(cache["records"], cache["urls"], strict=True)):
        raise ValueError("Categorical record order differs")
    output = {}
    for keys, source in ((URL_KEYS, url_frame), (HTML_KEYS, cache["features"]), (KEYS, extra["features"])):
        for key in keys:
            values = source[key].tolist()
            if any(not isinstance(value, str) for value in values):
                raise ValueError("Categories must be strings")
            output[key] = [value or "EMPTY" for value in values]
    output["cat_normalized_title"] = [re.sub(r"\s+", " ", record["title"]).strip().lower()[:160] or "EMPTY"
                                      for record in cache["records"]]
    return pd.DataFrame(output, columns=CATEGORY_KEYS)


def specialist_frame(numeric, categories, kind):
    if kind not in KINDS:
        raise ValueError("Unknown specialist kind")
    if len(numeric) != len(categories) or numeric.columns.duplicated().any() or \
            not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Specialist numeric rows must align and be finite")
    result = numeric.reset_index(drop=True).copy()
    if kind == "categorical":
        if list(categories.columns) != list(CATEGORY_KEYS) or any(not isinstance(value, str)
                for value in categories.to_numpy().flat):
            raise ValueError("Categorical specialist schema differs")
        result = pd.concat([result, categories.reset_index(drop=True)], axis=1)
        if result.columns.duplicated().any():
            raise ValueError("Specialist column names must be unique")
    return result


def make_specialist(kind):
    if kind not in KINDS:
        raise ValueError("Unknown specialist kind")
    params = dict(learning_rate=.04, l2_leaf_reg=30., loss_function="Logloss",
                  random_seed=20261010, thread_count=2, verbose=False, allow_writing_files=False)
    if kind == "numeric_deep":
        params.update(depth=7, iterations=1200)
    else:
        params.update(depth=6, iterations=1000, cat_features=list(CATEGORY_KEYS))
    return CatBoostClassifier(**params)


def mix_specialist(v10_probability, v15_probability, expert_probability, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid specialist mixture weight")
    mask = eligibility(v10_probability)
    p, q = np.asarray(v15_probability, dtype=float), np.asarray(expert_probability, dtype=float)
    if p.shape != mask.shape or p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all() or \
            (p < 0).any() or (p > 1).any() or (q < 0).any() or (q > 1).any():
        raise ValueError("Specialist probabilities must align and be finite")
    result = p.copy()
    result[mask] = (1-weight)*p[mask]+weight*q[mask]
    return result
