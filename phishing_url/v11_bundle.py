"""A small own-trained ensemble over frozen URL and HTML classifiers."""

import numpy as np

from .content import EXTRACTOR_VERSION
from .content_bundle import prediction_frame
from .content_memory import apply_memory
from .soft_context import apply_context, context_keys, lookup_counts
from .v5_bundle import mix_probabilities
from .v6_bundle import gate_extra, mix_extra
from .v7_hierarchy import hierarchy_keys
from .v9_routing import routing_mask
from .v11_meta import STRUCTURAL, ensemble_frame


def component_inputs(base, cache, extra, url_frame):
    v9, v7 = base["base"], base["base"]["base"]
    v6, v5 = v7["base"], v7["base"]["base"]
    v3 = v5["base"]
    if base["text_replacement"] != 1 or v7["config"]["text_replacement"] != 1:
        raise ValueError("V11 requires the frozen complete calibrated-text replacement")
    if v3["extractor_version"] != EXTRACTOR_VERSION:
        raise ValueError("Model and HTML extractor versions differ")
    if cache["urls"] != extra["urls"] or len(url_frame) != len(cache["urls"]) or \
            len(cache["urls"]) != len(cache["records"]) or len(cache["urls"]) != len(extra["records"]):
        raise ValueError("Ensemble prediction inputs must align")
    for records in (cache["records"], extra["records"]):
        if any(r["url"] != u for r, u in zip(records, cache["urls"], strict=True)):
            raise ValueError("Ensemble records differ in URL order")
    frame = prediction_frame(cache, url_frame)
    if list(frame.columns) != v3["feature_columns"]:
        raise ValueError("Frozen numeric feature schema differs")
    if v5["config"].get("leaf_weight", 0) or v6["config"].get("extra_boost_weight", 0) or \
            v6["config"].get("neighbor_adjustment"):
        raise ValueError("Unsupported frozen auxiliary configuration")
    components = {
        "svm": base["margin"].predict_proba(cache["records"]),
        "c16": v7["replacement_text"].predict_proba(cache["records"]),
        "numeric": v3["boost"].predict_proba(frame)[:, 1],
        "url": .5*v3["aux_url_text"].predict_proba(cache["urls"])+
               .5*v3["aux_url_boost"].predict_proba(url_frame)[:, 1],
        "cat": v5["cat"].predict_proba(frame)[:, 1],
        "hashed": v5["hashed"].predict_proba(cache["records"]),
        "extra": v6["extra_text"].predict_proba(extra["records"])}
    routed = routing_mask(cache, v9["routing"]["selector"])
    config = v5["config"]
    normal = mix_probabilities(components["svm"], components["numeric"], components["url"],
                              None, components["cat"], config, components["hashed"])
    weak = mix_probabilities(components["svm"], components["numeric"], components["url"],
                            None, components["cat"], dict(config, aux_url_weight=v9["routing"]["url_weight"]),
                            components["hashed"])
    raw = np.where(routed, weak, normal)
    counts = lookup_counts(v5["context"], context_keys(cache["records"]))
    meta = ensemble_frame(components, frame[list(STRUCTURAL)], routed, counts, v5["context"]["prior"])
    return meta, raw, components["extra"], routed


def finish_probabilities(base, cache, raw, extra_probability, routed):
    v7 = base["base"]["base"]
    v6, v5 = v7["base"], v7["base"]["base"]
    hard, matched = apply_memory(v5["memory"], cache["records"], cache["hashes"], raw)
    probability = apply_context(v5["context"], cache["records"], raw, v5["context_config"])
    probability[matched] = hard[matched]
    known = np.zeros(len(probability), dtype=bool)
    for i, url in enumerate(cache["urls"]):
        if url in v5["base"]["known_urls"]:
            probability[i] = v5["base"]["known_urls"][url]
            known[i] = True
    mixed = mix_extra(probability, extra_probability, None, v6["config"])
    probability = gate_extra(probability, mixed, extra_probability,
                             cache["features"].html_text_length.to_numpy(), v6["config"])
    hits = {"known_url": known, "training_match": matched, "url_routed": routed}
    if v7["config"]["hierarchy"]:
        probability, hits["hierarchy_match"] = v7["hierarchy"].predict_keys(
            [hierarchy_keys(url) for url in cache["urls"]], probability)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Invalid learned ensemble probabilities")
    return probability, hits


def predict_v11(bundle, cache, extra, url_frame):
    weight = float(bundle["meta_weight"])
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid learned ensemble mixture weight")
    meta, raw, extra_probability, routed = component_inputs(bundle["base"], cache, extra, url_frame)
    if weight:
        if bundle.get("meta") is None:
            raise ValueError("A trained learned ensemble is required")
        raw = (1-weight)*raw+weight*bundle["meta"].predict_proba(meta)
    return finish_probabilities(bundle["base"], cache, raw, extra_probability, routed)
