"""A saved v15 ensemble plus a separately guarded broader specialist."""

import numpy as np

from .v12_expert import blend_expert, eligibility
from .v15_bundle import resource_inputs
from .v16_expert import category_frame, mix_specialist, specialist_frame


def specialist_inputs(v15, kind, cache, extra, url_frame):
    numeric, v10_probability, hits = resource_inputs(v15["base"], v15["resource_context"], cache, extra, url_frame)
    if numeric.columns.tolist() != v15["feature_columns"]:
        raise ValueError("Frozen v15 numeric schema differs")
    eligible = eligibility(v10_probability)
    old_expert = v10_probability.copy()
    if eligible.any():
        old_expert[eligible] = v15["expert"].predict_proba(numeric.iloc[eligible], thread_count=2)[:, 1]
    baseline = blend_expert(v10_probability, old_expert, v15["expert_weight"])
    categories = category_frame(cache, extra, url_frame)
    frame = specialist_frame(numeric, categories, kind)
    hits["expert_routed"] = eligible
    return frame, v10_probability, baseline, hits


def predict_v16(bundle, cache, extra, url_frame):
    frame, v10_probability, v15_probability, hits = specialist_inputs(bundle["base"], bundle["kind"], cache, extra, url_frame)
    if frame.columns.tolist() != bundle["feature_columns"]:
        raise ValueError("Broader specialist feature schema differs")
    eligible = eligibility(v10_probability)
    probability = v15_probability.copy()
    if eligible.any():
        probability[eligible] = bundle["expert"].predict_proba(frame.iloc[eligible], thread_count=2)[:, 1]
    return mix_specialist(v10_probability, v15_probability, probability, bundle["expert_weight"]), hits
