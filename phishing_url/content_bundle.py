"""Inference for the fitted URL + static HTML ensemble."""

import numpy as np
import pandas as pd

from .content import EXTRACTOR_VERSION
from .content_memory import apply_memory
from .features import make_features


def prediction_frame(cache, url_frame=None):
    if url_frame is None:
        url_frame = make_features(cache["urls"])
    return pd.concat([url_frame.reset_index(drop=True), cache["features"].reset_index(drop=True)], axis=1)


def predict_content(bundle, cache, url_frame=None):
    if bundle["extractor_version"] != EXTRACTOR_VERSION:
        raise ValueError("Model and HTML extractor versions differ")
    if len(cache["urls"]) != len(cache["records"]):
        raise ValueError("Content cache rows disagree")
    if any(url != record["url"] for url, record in zip(cache["urls"], cache["records"], strict=True)):
        raise ValueError("Content cache record order differs")
    weight = bundle["text_weight"]
    p = np.zeros(len(cache["urls"]), dtype=float)
    if weight:
        p += weight * bundle["text"].predict_proba(cache["records"])
    if weight < 1:
        frame = prediction_frame(cache, url_frame)
        if list(frame.columns) != bundle["feature_columns"]:
            raise ValueError("HTML/URL feature schema differs")
        p += (1-weight) * bundle["boost"].predict_proba(frame)[:, 1]
    auxiliary = bundle.get("aux_url_weight", 0.)
    if auxiliary:
        if url_frame is None:
            url_frame = make_features(cache["urls"])
        p_url = .5 * bundle["aux_url_text"].predict_proba(cache["urls"]) + .5 * bundle["aux_url_boost"].predict_proba(url_frame)[:, 1]
        p = (1-auxiliary) * p + auxiliary * p_url
    page_hits = np.zeros(len(p), dtype=bool)
    if bundle["memory_minimum_support"] or bundle.get("domain_minimum_support", 0):
        p, page_hits = apply_memory(bundle["memory"], cache["records"], cache["hashes"], p)
    known_hits = np.zeros(len(p), dtype=bool)
    for i, url in enumerate(cache["urls"]):
        if url in bundle["known_urls"]:
            p[i] = bundle["known_urls"][url]
            known_hits[i] = True
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid content model probabilities")
    return p, {"training_match": page_hits, "known_url": known_hits}
