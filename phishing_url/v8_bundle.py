"""Reweight the saved v7 learned components without modifying their baseline."""

import numpy as np

from .v7_bundle import predict_v7


def reweighted_base(bundle):
    base = dict(bundle["base"])
    v6 = dict(base["base"])
    v5 = dict(v6["base"])
    config = dict(bundle["v5_config"])
    if any(not 0 <= config.get(key, 0.) <= 1 for key in
           ("text_weight", "aux_url_weight", "hash_weight", "cat_weight", "leaf_weight")):
        raise ValueError("Invalid component weight")
    if config.get("cat_weight", 0.)+config.get("leaf_weight", 0.) > 1:
        raise ValueError("Invalid tree mixture")
    v5["config"] = config
    v6["base"] = v5
    v6["config"] = dict(bundle["v6_config"])
    base["base"] = v6
    replacement = bundle.get("replacement_text")
    if replacement is not None:
        base["replacement_text"] = replacement
    return base


def predict_v8(bundle, cache, extra, url_frame):
    base, hits = predict_v7(reweighted_base(bundle), cache, extra, url_frame)
    weight = bundle.get("resource_weight", 0.)
    resource = bundle["resource_text"].predict_proba(extra["records"]) if weight else None
    return mix_resources(base, resource, weight), hits


def mix_resources(base, resource, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Invalid resource mixture weight")
    base = np.asarray(base, dtype=float)
    if base.ndim != 1 or not np.isfinite(base).all() or (base < 0).any() or (base > 1).any():
        raise ValueError("Invalid baseline probabilities")
    if not weight:
        return base.copy()
    if resource is None:
        raise ValueError("Resource probabilities required")
    resource = np.asarray(resource, dtype=float)
    if resource.shape != base.shape or not np.isfinite(resource).all() or (resource < 0).any() or (resource > 1).any():
        raise ValueError("Invalid resource probabilities")
    p = (1-weight)*base+weight*resource
    hard = (base == 0) | (base == 1)
    p[hard] = base[hard]
    return p
