"""Combine a saved learned baseline with new static extra-HTML models."""

import numpy as np
import pandas as pd

from .content_bundle import prediction_frame, predict_content
from .v5_bundle import predict_v5
from .neighbors import adjust_neighbors


def mix_extra(base, text, boost, config):
    base = np.asarray(base, dtype=float)
    tw, bw = config.get("extra_text_weight", 0.), config.get("extra_boost_weight", 0.)
    if tw < 0 or bw < 0 or tw + bw > 1:
        raise ValueError("Invalid extra model weights")
    p = (1 - tw - bw) * base
    for values, weight in ((text, tw), (boost, bw)):
        if weight:
            if values is None or np.asarray(values).shape != base.shape:
                raise ValueError("Extra probabilities must align with the baseline")
            p += weight * np.asarray(values)
    if config.get("preserve_training_rules", True):
        hard = (base == 0) | (base == 1)
        p[hard] = base[hard]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid extra ensemble probabilities")
    return p


def gate_extra(base, mixed, text, visible_lengths, config):
    gate = np.ones(len(base), dtype=bool)
    confidence = config.get("extra_confidence")
    if confidence is not None:
        gate &= (text >= confidence) | (text <= 1 - confidence)
    maximum = config.get("maximum_visible_text")
    if maximum is not None:
        gate &= np.asarray(visible_lengths) <= maximum
    result = np.asarray(mixed).copy()
    result[~gate] = np.asarray(base)[~gate]
    return result


def predict_v6(bundle, cache, extra, url_frame):
    if cache["urls"] != extra["urls"] or len(cache["urls"]) != len(extra["records"]):
        raise ValueError("Extra HTML and baseline URL orders differ")
    if any(r["url"] != u for r, u in zip(extra["records"], cache["urls"], strict=True)):
        raise ValueError("Extra records are out of order")
    config = bundle["config"]
    function = predict_v5 if config["base_version"] == 5 else predict_content
    base, hits = function(bundle["base"], cache, url_frame)
    text = bundle["extra_text"].predict_proba(extra["records"]) if config["extra_text_weight"] else None
    boost = None
    if config["extra_boost_weight"]:
        frame = pd.concat([prediction_frame(cache, url_frame), extra["features"].reset_index(drop=True)], axis=1)
        if list(frame.columns) != bundle["feature_columns"]:
            raise ValueError("Extra numeric feature schema differs")
        boost = bundle["extra_boost"].predict_proba(frame)[:, 1]
    p = mix_extra(base, text, boost, config)
    p = gate_extra(base, p, text, cache["features"].html_text_length.to_numpy(), config)
    if config.get("neighbor_adjustment"):
        similarities, labels = bundle["neighbors"].lookup(cache["records"])
        p = adjust_neighbors(p, similarities, labels,
                             {**config["neighbor_adjustment"], "prior": bundle["neighbors"].prior_})
    return p, hits
