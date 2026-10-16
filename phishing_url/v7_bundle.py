"""Own supervised v7: replace HTML text and optionally add train-only URL context."""

from .v5_bundle import TextMixture
from .v6_bundle import predict_v6
from .v7_hierarchy import hierarchy_keys


def predict_v7(bundle, cache, extra, url_frame):
    base = dict(bundle["base"])
    weight = bundle["config"]["text_replacement"]
    if not 0 <= weight <= 1:
        raise ValueError("Invalid replacement weight")
    if weight:
        v5 = dict(base["base"])
        v3 = dict(v5["base"])
        new_text = bundle["replacement_text"]
        if new_text is None:
            raise ValueError("Replacement text model is required")
        v3["text"] = new_text if weight == 1 else TextMixture(v3["text"], new_text, weight)
        v5["base"] = v3
        base["base"] = v5
    probability, hits = predict_v6(base, cache, extra, url_frame)
    if bundle["config"]["hierarchy"]:
        probability, matched = bundle["hierarchy"].predict_keys([hierarchy_keys(url) for url in cache["urls"]], probability)
        hits["hierarchy_match"] = matched
    return probability, hits

