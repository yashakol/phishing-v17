import numpy as np
import pytest

from phishing_url.v7_bundle import predict_v7
from phishing_url.v7_hierarchy import HierarchicalEvidence, hierarchy_keys
from phishing_url.v7_text import FullVocabularyTextModel


def test_text_replacement_keeps_saved_baseline_immutable(monkeypatch):
    original = {"base": {"base": {"text": "old"}}}
    replacement = object()
    seen = []
    def predictor(base, cache, extra, frame):
        seen.append(base["base"]["base"]["text"])
        return np.array([.2]), {}
    monkeypatch.setattr("phishing_url.v7_bundle.predict_v6", predictor)
    model = {"base": original, "config": {"text_replacement": 1., "hierarchy": False},
             "replacement_text": replacement}
    p, _ = predict_v7(model, {}, {}, None)
    assert seen == [replacement]
    assert original["base"]["base"]["text"] == "old"
    np.testing.assert_array_equal(p, [.2])


def test_hierarchy_preserves_hard_training_rules():
    urls = [f"https://example.org/auth/{i}" for i in range(3)]
    keys = [hierarchy_keys(url) for url in urls]
    model = HierarchicalEvidence(["host_path1"], minimum_support=3).fit_keys(keys, [1, 1, 1])
    probabilities, hits = model.predict_keys(keys, [0., 1., .45])
    np.testing.assert_array_equal(probabilities[:2], [0., 1.])
    assert probabilities[2] > .45 and hits.tolist() == [False, False, True]


def test_hierarchy_ignores_conflicting_training_labels():
    keys = [hierarchy_keys(f"https://example.org/auth/{i}") for i in range(3)]
    model = HierarchicalEvidence(["host_path1"], minimum_support=2).fit_keys(keys, [0, 1, 0])
    p, hits = model.predict_keys(keys, [.4, .5, .6])
    np.testing.assert_array_equal(p, [.4, .5, .6])
    assert not hits.any()


def test_hierarchy_does_not_invent_evidence_for_unseen_host():
    train = [hierarchy_keys(f"https://example.org/auth/{i}") for i in range(3)]
    model = HierarchicalEvidence(["host_path1"], minimum_support=3).fit_keys(train, [1, 1, 1])
    p, hits = model.predict_keys([hierarchy_keys("https://example.net/auth/new")], [.4])
    np.testing.assert_array_equal(p, [.4])
    assert not hits.any()


def test_full_text_vocabulary_does_not_use_validation_text():
    records = [{"url": f"https://example.org/train/{i}", "title": "training title",
                "text": "ordinary news article" if i < 2 else "account password login",
                "markup": "html body form input"} for i in range(4)]
    model = FullVocabularyTextModel(min_df=1, word_features=100, char_features=100,
                                   markup_features=100, url_features=100, verbose=False)
    model.vocabulary_fit_rows = 1
    model.fit(records, np.array([0, 0, 1, 1]))
    validation = {**records[0], "url": "https://example.org/validation", "text": "withheldonly"}
    assert model.vocabulary_fit_rows_ == 4
    assert "withheldonly" not in model.vectorizers_["word"].vocabulary_
    assert model.predict_proba([validation]).shape == (1,)


def test_invalid_replacement_weight_is_rejected():
    with pytest.raises(ValueError, match="Invalid replacement"):
        predict_v7({"base": {}, "config": {"text_replacement": 1.5}}, {}, {}, None)

