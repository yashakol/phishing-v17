import numpy as np
import pytest

from phishing_url.v6_bundle import mix_extra, predict_v6, gate_extra


def test_extra_models_keep_known_training_rule_answers():
    base = np.array([0., 1., .5])
    p = mix_extra(base, np.array([1., 0., .9]), None,
                  {"extra_text_weight": .35, "extra_boost_weight": 0., "preserve_training_rules": True})
    np.testing.assert_array_equal(p[:2], [0., 1.])
    assert .5 < p[2] < .9


def test_extra_prediction_rejects_reordered_records():
    cache = {"urls": ["https://first.invalid", "https://second.invalid"]}
    extra = {"urls": cache["urls"], "records": [{"url": cache["urls"][1]}, {"url": cache["urls"][0]}]}
    with pytest.raises(ValueError, match="out of order"):
        predict_v6({}, cache, extra, None)


def test_extra_mixture_requires_each_nonzero_component():
    with pytest.raises(ValueError, match="must align"):
        mix_extra(np.array([.4]), None, None, {"extra_text_weight": .1})


def test_expert_gate_requires_confidence_and_short_text_when_configured():
    base = np.array([.6, .6, .6])
    mixed = np.array([.3, .3, .3])
    text = np.array([.01, .5, .01])
    p = gate_extra(base, mixed, text, [20, 20, 1000], {"extra_confidence": .9, "maximum_visible_text": 80})
    np.testing.assert_array_equal(p, [.3, .6, .6])
