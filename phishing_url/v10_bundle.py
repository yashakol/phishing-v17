"""Add a trained calibrated margin model to the frozen v9 text component."""

import numpy as np

from .v5_bundle import TextMixture
from .v9_routing import predict_v9


def predict_v10(bundle, cache, extra, url_frame):
    weight = float(bundle["text_replacement"])
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid calibrated-text mixture weight")
    routed = dict(bundle["base"])
    base = dict(routed["base"])
    if base["config"]["text_replacement"] != 1:
        raise ValueError("V10 requires the frozen complete C16 text replacement")
    if weight:
        if bundle.get("margin") is None:
            raise ValueError("A trained calibrated margin model is required")
        base["replacement_text"] = TextMixture(base["replacement_text"], bundle["margin"], weight)
    routed["base"] = base
    return predict_v9(routed, cache, extra, url_frame)
