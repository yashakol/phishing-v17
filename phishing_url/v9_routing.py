"""Use existing learned URL/HTML components with an availability-based weight."""

import re

import numpy as np

from .v7_bundle import predict_v7


ERROR_TITLE = re.compile(r"(?:\b(?:403|404|410|500|502|503|forbidden|unavailable)\b|access denied|not found|bad gateway|"
                         r"internal server error|just a moment|attention required|checking your browser|domain (?:expired|parked))", re.I)


def routing_mask(cache, selector):
    if len(cache["records"]) != len(cache["urls"]) or len(cache["features"]) != len(cache["urls"]):
        raise ValueError("Routing records must align")
    short = cache["features"].html_text_length.to_numpy() < 100
    error = np.asarray([bool(ERROR_TITLE.search(r["title"])) for r in cache["records"]], dtype=bool)
    if selector == "short100_or_error":
        return short | error
    if selector == "short100":
        return short
    if selector == "error_title":
        return error
    if selector == "empty":
        return cache["features"].html_empty.to_numpy().astype(bool)
    raise ValueError("Unknown HTML routing selector")


def replace_url_weight(original, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Invalid URL mixture weight")
    v7 = dict(original)
    v6 = dict(v7["base"])
    v5 = dict(v6["base"])
    v5["config"] = dict(v5["config"], aux_url_weight=weight)
    v6["base"] = v5
    v7["base"] = v6
    return v7


def subset_cache(cache, indices):
    result = dict(cache)
    for name in ("urls", "records", "hashes"):
        if name in cache:
            result[name] = [cache[name][int(i)] for i in indices]
    if "features" in cache:
        result["features"] = cache["features"].iloc[indices].reset_index(drop=True)
    if cache.get("labels") is not None:
        result["labels"] = np.asarray(cache["labels"])[indices]
    return result


def predict_v9(bundle, cache, extra, url_frame):
    if cache["urls"] != extra["urls"] or len(url_frame) != len(cache["urls"]):
        raise ValueError("Routing inputs have different URL orders or lengths")
    mask = routing_mask(cache, bundle["routing"]["selector"])
    p = np.empty(len(mask), dtype=float)
    hits = {"url_routed": mask.copy()}
    for active in (False, True):
        indices = np.flatnonzero(mask == active)
        if len(indices) == 0:
            continue
        model = replace_url_weight(bundle["base"], bundle["routing"]["url_weight"]) if active else bundle["base"]
        values, matched = predict_v7(model, subset_cache(cache, indices), subset_cache(extra, indices),
                                    url_frame.iloc[indices].reset_index(drop=True))
        p[indices] = values
        for kind, flags in matched.items():
            if kind not in hits:
                hits[kind] = np.zeros(len(mask), dtype=bool)
            hits[kind][indices] = flags
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid routed probabilities")
    for kind in ("known_url", "training_match"):
        hits.setdefault(kind, np.zeros(len(mask), dtype=bool))
    return p, hits
