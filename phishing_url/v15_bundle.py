"""An own-trained resource-domain specialist restricted to uncertain outputs."""

from .v12_bundle import inference_inputs
from .v12_expert import blend_expert, eligibility
from .v13_features import extend_frame
from .v15_resources import add_resource_evidence


def resource_inputs(base, context, cache, extra, url_frame):
    structural, probability, hits = inference_inputs(base, cache, extra, url_frame)
    static = extend_frame(structural, extra["features"])
    frame = add_resource_evidence(static, extra["features"], context)
    return frame, probability, hits


def predict_v15(bundle, cache, extra, url_frame):
    frame, baseline, hits = resource_inputs(bundle["base"], bundle["resource_context"], cache, extra, url_frame)
    if list(frame.columns) != bundle["feature_columns"]:
        raise ValueError("Resource specialist feature schema differs")
    mask, probability = eligibility(baseline), baseline.copy()
    if mask.any():
        probability[mask] = bundle["expert"].predict_proba(frame.iloc[mask], thread_count=2)[:, 1]
    hits["expert_routed"] = mask
    return blend_expert(baseline, probability, bundle["expert_weight"]), hits
