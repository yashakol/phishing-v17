import numpy as np
import pytest

from phishing_url.v8_bundle import mix_resources, reweighted_base
from scripts.refit_v8_regularization import semantic_snapshot
from types import SimpleNamespace


def test_resource_mixture_preserves_exact_training_answers():
    base = np.array([0., .4, 1.])
    p = mix_resources(base, [.9, .8, .1], .25)
    np.testing.assert_allclose(p, [0., .5, 1.])
    np.testing.assert_array_equal(base, [0., .4, 1.])


def test_resource_mixture_rejects_misaligned_or_nonfinite_predictions():
    with pytest.raises(ValueError, match="resource probabilities"):
        mix_resources([.2, .3], [.8], .1)
    with pytest.raises(ValueError, match="resource probabilities"):
        mix_resources([.2, .3], [.8, np.nan], .1)


def test_reweighting_does_not_modify_previous_models():
    previous = {"base": {"base": {"config": {"text_weight": .7}}}, "replacement_text": "previous text"}
    bundle = {"base": previous, "v5_config": {"text_weight": .85}, "v6_config": {"extra_text_weight": .1},
              "replacement_text": "replacement"}
    changed = reweighted_base(bundle)
    assert changed["base"]["base"]["config"]["text_weight"] == .85
    assert changed["replacement_text"] == "replacement"
    assert previous["base"]["base"]["config"]["text_weight"] == .7
    assert previous["replacement_text"] == "previous text"


def test_reweighting_rejects_negative_or_overfull_mixtures():
    previous = {"base": {"base": {}}}
    with pytest.raises(ValueError, match="component weight"):
        reweighted_base({"base": previous, "v5_config": {"text_weight": -.2}})
    with pytest.raises(ValueError, match="tree mixture"):
        reweighted_base({"base": previous, "v5_config": {"leaf_weight": .6, "cat_weight": .5}})


def test_vocabulary_signature_handles_numpy_indices_without_changing_semantics():
    vectorizer = SimpleNamespace(vocabulary_={"alpha": 0, "beta": np.int64(1)}, idf_=np.array([1., 2.], dtype=np.float32))
    classifier = SimpleNamespace(coef_=np.array([[.1, .2]]), intercept_=np.array([.3]))
    model = SimpleNamespace(vectorizers_={"word": vectorizer}, classifier_=classifier)
    before = semantic_snapshot(model)
    vectorizer.vocabulary_["beta"] = 1
    assert semantic_snapshot(model) == before
    vectorizer.vocabulary_["beta"] = 2
    assert semantic_snapshot(model) != before
