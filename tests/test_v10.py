"""Integration contracts for the new classifier inside frozen routing."""

import copy

import numpy as np
import pytest

from phishing_url import v10_bundle


class Constant:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict_proba(self, records):
        return self.values[:len(records)]


def model(weight, margin=True):
    return {"text_replacement": weight,
            "margin": Constant([.1, .9]) if margin else None,
            "base": {"routing": {"selector": "short100_or_error", "url_weight": .5},
                     "base": {"config": {"text_replacement": 1},
                              "replacement_text": Constant([.9, .1])}}}


def test_margin_is_inserted_before_existing_routing_without_mutation(monkeypatch):
    original = model(.25)
    old_text = original["base"]["base"]["replacement_text"]
    config = copy.deepcopy(original["base"]["base"]["config"])

    def routing(bundle, cache, extra, frame):
        assert bundle["routing"] == original["base"]["routing"]
        assert bundle["base"]["config"] == config
        return bundle["base"]["replacement_text"].predict_proba(cache["records"]), {"url_routed": np.array([False, True])}

    monkeypatch.setattr(v10_bundle, "predict_v9", routing)
    p, hits = v10_bundle.predict_v10(original, {"records": [{}, {}]}, None, None)
    np.testing.assert_allclose(p, [.7, .3], atol=1e-15)
    np.testing.assert_array_equal(hits["url_routed"], [False, True])
    assert original["base"]["base"]["replacement_text"] is old_text
    np.testing.assert_array_equal(old_text.values, [.9, .1])


def test_zero_margin_weight_needs_no_extra_model(monkeypatch):
    original = model(0, margin=False)
    monkeypatch.setattr(v10_bundle, "predict_v9", lambda b, *args: b["base"]["replacement_text"])
    assert v10_bundle.predict_v10(original, None, None, None) is original["base"]["base"]["replacement_text"]


@pytest.mark.parametrize("weight", [float("nan"), -0.1, 1.1])
def test_invalid_margin_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        v10_bundle.predict_v10(model(weight), None, None, None)


def test_missing_calibrated_model_is_rejected():
    with pytest.raises(ValueError, match="trained"):
        v10_bundle.predict_v10(model(.5, margin=False), None, None, None)


def test_incompatible_original_text_configuration_is_rejected():
    original = model(.5)
    original["base"]["base"]["config"]["text_replacement"] = .5
    with pytest.raises(ValueError, match="C16"):
        v10_bundle.predict_v10(original, None, None, None)
