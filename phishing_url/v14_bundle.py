"""A fixed equal pair of own-trained specialists for uncertain base outputs."""

import numpy as np

from .v12_bundle import inference_inputs
from .v12_expert import blend_expert, eligibility
from .v13_features import extend_frame


def pair_inputs(base, cache, extra, url_frame):
    structural, baseline, hits = inference_inputs(base, cache, extra, url_frame)
    static = extend_frame(structural, extra["features"])
    return structural, static, baseline, hits


def pair_probability(bundle, structural, static, baseline):
    if list(structural.columns) != bundle["structural_columns"] or \
            list(static.columns) != bundle["static_columns"]:
        raise ValueError("Specialist pair feature schemas differ")
    mask, result = eligibility(baseline), np.asarray(baseline).copy()
    if mask.any():
        first = bundle["structural_expert"].predict_proba(structural.iloc[mask], thread_count=2)[:, 1]
        second = bundle["static_expert"].predict_proba(static.iloc[mask], thread_count=2)[:, 1]
        result[mask] = .5*first+.5*second
    return result, mask


def predict_v14(bundle, cache, extra, url_frame):
    structural, static, baseline, hits = pair_inputs(bundle["base"], cache, extra, url_frame)
    probability, mask = pair_probability(bundle, structural, static, baseline)
    hits["expert_routed"] = mask
    return blend_expert(baseline, probability, bundle["expert_weight"]), hits
