"""Inference for an uncertainty specialist with unchanged frozen base outputs."""

import numpy as np

from .content_bundle import prediction_frame
from .v11_bundle import component_inputs, finish_probabilities
from .v12_expert import blend_expert, eligibility, expert_frame


class PreparedProbability:
    def __init__(self, records, probability):
        self.urls = tuple(r["url"] for r in records)
        self.probability = np.asarray(probability)

    def predict_proba(self, records):
        if tuple(r["url"] for r in records) != self.urls:
            raise ValueError("Prepared component records are out of order")
        return self.probability


def shared_inputs(base, cache, extra, url_frame):
    # Both models were fitted with the same guarded C16 vocabulary and IDF.
    # Verify the complete representation before transforming the records once.
    margin = base["margin"]
    c16 = base["base"]["base"]["replacement_text"]
    transformer = margin.transformer_
    if c16.get_params() != transformer.get_params() or c16.n_features_ != margin.n_features_ or \
            c16.nb or c16.ratio_ is not None or transformer.ratio_ is not None:
        raise ValueError("Shared text representation parameters differ")
    if set(c16.vectorizers_) != set(transformer.vectorizers_):
        raise ValueError("Shared text vectorizer schema differs")
    for name, old in c16.vectorizers_.items():
        copied = transformer.vectorizers_[name]
        if old.vocabulary_ != copied.vocabulary_ or not np.array_equal(old.idf_, copied.idf_):
            raise ValueError("Shared text vocabulary or IDF differs")
    matrix = transformer._matrix(cache["records"])
    p_svm = margin.classifier_.predict_proba(matrix)[:, 1]
    p_c16 = c16.classifier_.predict_proba(matrix)[:, 1]
    del matrix
    prepared = dict(base)
    routed = dict(base["base"])
    v7 = dict(routed["base"])
    prepared["margin"] = PreparedProbability(cache["records"], p_svm)
    v7["replacement_text"] = PreparedProbability(cache["records"], p_c16)
    routed["base"], prepared["base"] = v7, routed
    return component_inputs(prepared, cache, extra, url_frame)


def inference_inputs(base, cache, extra, url_frame):
    meta, raw, extra_probability, routed = shared_inputs(base, cache, extra, url_frame)
    probability, hits = finish_probabilities(base, cache, raw, extra_probability, routed)
    frame = expert_frame(meta, prediction_frame(cache, url_frame), probability)
    return frame, probability, hits


def predict_v12(bundle, cache, extra, url_frame):
    weight = float(bundle["expert_weight"])
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid uncertainty specialist weight")
    frame, baseline, hits = inference_inputs(bundle["base"], cache, extra, url_frame)
    if list(frame.columns) != bundle["feature_columns"]:
        raise ValueError("Specialist feature schema differs")
    mask, probability = eligibility(baseline), baseline.copy()
    if weight and mask.any():
        if bundle.get("expert") is None:
            raise ValueError("A trained uncertainty specialist is required")
        probability[mask] = bundle["expert"].predict_proba(frame.iloc[mask], thread_count=2)[:, 1]
    hits["expert_routed"] = mask
    return blend_expert(baseline, probability, weight), hits
