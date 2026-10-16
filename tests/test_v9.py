"""Guard evidence conflicts and calibrated margin serialization boundaries."""

import joblib
import numpy as np
import pytest
import pandas as pd

from phishing_url.v9_memory import DEFAULT_POLICY, MemoryEvidence, apply_evidence
from phishing_url.v9_margin import CalibratedMarginText
from phishing_url.v7_text import FullVocabularyTextModel
from phishing_url import v9_routing


def test_conflicting_page_and_domain_evidence_retains_learner():
    counts = {"raw": np.array([[0, 3], [0, 3], [3, 0]]),
              "text": np.array([[0, 3], [3, 0], [3, 0]]),
              "domain": np.array([[5, 0], [0, 0], [0, 5]])}
    base = np.array([.8, .2, .4])
    actual, hits = apply_evidence(base, counts, DEFAULT_POLICY)
    np.testing.assert_array_equal(actual, base)
    assert not hits.any()


def test_unseen_keys_do_not_add_evidence_and_query_is_read_only():
    keys = {kind: ["a", "a", "b"] for kind in ("raw", "text", "domain")}
    model = MemoryEvidence(DEFAULT_POLICY).fit_keys(keys, [0, 0, 1])
    before = dict(model.counts_["domain"])
    query = {kind: ["a", "unseen"] for kind in keys}
    for value in model.lookup(query).values():
        np.testing.assert_array_equal(value, [[2, 0], [0, 0]])
    assert model.counts_["domain"] == before


def test_domain_rule_inclusive_confidence_boundaries():
    base = np.array([.05, .95, .049, .951])
    counts = {"raw": np.zeros((4, 2), dtype=int), "text": np.zeros((4, 2), dtype=int),
              "domain": np.tile([5, 0], (4, 1))}
    p, hits = apply_evidence(base, counts, DEFAULT_POLICY)
    np.testing.assert_array_equal(p, [0, 0, .049, .951])
    np.testing.assert_array_equal(hits, [True, True, False, False])


def test_single_domain_vote_veto_needs_model_agreement():
    counts = {"raw": np.tile([0, 3], (2, 1)), "text": np.tile([0, 3], (2, 1)),
              "domain": np.tile([1, 0], (2, 1))}
    p, hits = apply_evidence(np.array([.05, .7]), counts, {**DEFAULT_POLICY, "veto_support": 1})
    np.testing.assert_array_equal(p, [.05, 1.])
    np.testing.assert_array_equal(hits, [False, True])


def test_invalid_evidence_shapes_and_labels_are_rejected():
    model = MemoryEvidence(DEFAULT_POLICY)
    with pytest.raises(ValueError):
        model.fit_keys({kind: ["a"] for kind in ("raw", "text", "domain")}, [2])
    with pytest.raises(ValueError):
        apply_evidence([.5], {kind: np.zeros((2, 2), dtype=int) for kind in ("raw", "text", "domain")}, DEFAULT_POLICY)


@pytest.fixture
def small_representation():
    y = np.tile([0, 1], 4)
    records = [{"url": f"https://example{i}.com/{'login' if label else 'news'}", "title": "Login" if label else "News",
                "text": "password account verify" if label else "news article information",
                "markup": "input password form" if label else "article link"} for i, label in enumerate(y)]
    original = FullVocabularyTextModel(min_df=1, word_features=50, char_features=50,
        markup_features=50, url_features=50, verbose=False).fit(records, y)
    return original, records, y


def test_margin_rejects_foreign_vocabulary_scope(small_representation):
    original, records, y = small_representation
    matrix = original._matrix(records)
    with pytest.raises(ValueError):
        CalibratedMarginText().fit_representation(original, matrix[:-1], y[:-1])


def test_calibrated_margin_saved_model_repeats_probabilities(tmp_path, small_representation):
    original, records, y = small_representation
    matrix = original._matrix(records)
    original_coef = original.classifier_.coef_.copy()
    model = CalibratedMarginText().fit_representation(original, matrix, y)
    p = model.predict_proba(records)
    assert p.shape == (8,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert model.classifier_fits_ == 4
    np.testing.assert_array_equal(original.classifier_.coef_, original_coef)
    path = tmp_path / "margin.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(records), p)


def test_route_selector_uses_visible_length_or_error_title():
    cache = {"urls": ["u0", "u1", "u2"], "records": [{"title": t} for t in ("News", "403 Forbidden", "News")],
             "features": pd.DataFrame({"html_text_length": [99, 200, 100], "html_empty": [0, 0, 0]})}
    np.testing.assert_array_equal(v9_routing.routing_mask(cache, "short100_or_error"), [True, True, False])
    with pytest.raises(ValueError):
        v9_routing.routing_mask(cache, "unknown")


def test_route_reweighting_does_not_mutate_original_model():
    base = {"base": {"base": {"config": {"aux_url_weight": .2, "text_weight": .7}}}}
    changed = v9_routing.replace_url_weight(base, .5)
    assert base["base"]["base"]["config"]["aux_url_weight"] == .2
    assert changed["base"]["base"]["config"]["aux_url_weight"] == .5
    assert changed["base"]["base"]["config"]["text_weight"] == .7


def test_partitioned_route_restores_url_order_and_provenance(monkeypatch):
    urls = ["u0", "u1", "u2", "u3"]
    cache = {"urls": urls, "records": [{"url": u, "title": "News"} for u in urls], "labels": None,
        "hashes": ["h0", "h1", "h2", "h3"],
        "features": pd.DataFrame({"html_text_length": [50, 200, 60, 300], "html_empty": [0]*4})}
    extra = dict(cache)
    base = {"base": {"base": {"config": {"aux_url_weight": .2}}}}
    def predict(model, subset, aux, url_frame):
        assert subset["urls"] == aux["urls"] == url_frame.url.tolist()
        weight = model["base"]["base"]["config"]["aux_url_weight"]
        values = np.array([int(u[-1])*.01+weight for u in subset["urls"]])
        return values, {"known_url": np.array([u == "u2" for u in subset["urls"]]),
                        "training_match": np.zeros(len(values), dtype=bool)}
    monkeypatch.setattr(v9_routing, "predict_v7", predict)
    p, hits = v9_routing.predict_v9({"base": base, "routing": {"selector": "short100_or_error", "url_weight": .5}},
                                   cache, extra, pd.DataFrame({"url": urls}))
    np.testing.assert_allclose(p, [.5, .21, .52, .23], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(hits["url_routed"], [True, False, True, False])
    np.testing.assert_array_equal(hits["known_url"], [False, False, True, False])
