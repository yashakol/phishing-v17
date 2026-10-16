"""Check learned-ensemble contracts that prevent silent inference mistakes."""

import joblib
import numpy as np
import pandas as pd
import pytest

from phishing_url import v11_bundle
from phishing_url.v11_meta import COMPONENTS, STRUCTURAL, LearnedEnsemble, ensemble_frame


def inputs(n=12):
    components = {name: np.linspace(.05, .95, n) for name in COMPONENTS}
    structural = pd.DataFrame({name: np.zeros(n) for name in STRUCTURAL})
    counts = {kind: np.zeros((n, 2), dtype=int) for kind in ("domain", "title")}
    return components, structural, np.zeros(n, dtype=bool), counts, .4


def test_unseen_training_evidence_is_neutral_and_does_not_need_labels():
    X = ensemble_frame(*inputs())
    for kind in ("domain", "title"):
        np.testing.assert_array_equal(X[kind+"_log_evidence"], 0)
        np.testing.assert_array_equal(X[kind+"_log_support"], 0)
        np.testing.assert_array_equal(X[kind+"_pure"], 0)


@pytest.mark.parametrize("bad", [float("nan"), -1, 2])
def test_invalid_component_values_are_rejected(bad):
    args = inputs()
    args[0]["svm"][0] = bad
    with pytest.raises(ValueError, match="probabilities"):
        ensemble_frame(*args)


def test_bad_alignment_of_training_evidence_is_rejected():
    args = inputs()
    args[3]["domain"] = np.zeros((11, 2), dtype=int)
    with pytest.raises(ValueError, match="align"):
        ensemble_frame(*args)


def test_saved_meta_model_preserves_probabilities_and_rejects_reordered_columns(tmp_path):
    X = ensemble_frame(*inputs())
    model = LearnedEnsemble().fit(X, np.array([0]*6+[1]*6))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))
    with pytest.raises(ValueError, match="schema"):
        restored.predict_proba(X.iloc[:, ::-1])


def test_class_missing_in_training_is_rejected():
    with pytest.raises(ValueError, match="binary"):
        LearnedEnsemble().fit(ensemble_frame(*inputs()), np.zeros(12))


@pytest.mark.parametrize("weight", [float("nan"), -.1, 1.1])
def test_invalid_ensemble_weight_rejected_before_loading_components(weight):
    with pytest.raises(ValueError, match="weight"):
        v11_bundle.predict_v11({"meta_weight": weight}, None, None, None)


def test_missing_model_rejected_and_base_probabilities_not_modified(monkeypatch):
    raw = np.array([.2, .8])
    monkeypatch.setattr(v11_bundle, "component_inputs", lambda *a: (None, raw, None, None))
    with pytest.raises(ValueError, match="trained"):
        v11_bundle.predict_v11({"meta_weight": .25, "base": {}, "meta": None}, None, None, None)
    np.testing.assert_array_equal(raw, [.2, .8])


def test_original_train_only_hard_rules_and_known_url_survive_meta(monkeypatch):
    from phishing_url.content_memory import fit_memory
    from phishing_url.soft_context import context_keys, fit_context
    records = [{"url": f"https://a{i}.example/", "title": "", "text": f"text{i}", "markup": ""}
               for i in range(3)]
    # Real fit/apply functions are exercised; only unrelated hierarchy is absent.
    memory = fit_memory(records, ["raw0", "raw1", "raw2"], [0, 1, 0],
                        minimum_support=1, domain_minimum_support=0)
    context = fit_context(context_keys(records), [0, 1, 0])
    v5 = {"memory": memory, "context": context, "context_config": {"alpha": 5},
          "base": {"known_urls": {records[2]["url"]: 1}}}
    base = {"base": {"base": {"config": {"hierarchy": False}, "base": {
        "base": v5, "config": {"extra_text_weight": .1, "preserve_training_rules": True}}}}}
    cache = {"records": records, "hashes": ["raw0", "raw1", "other"],
             "urls": [r["url"] for r in records], "features": pd.DataFrame({"html_text_length": [200]*3})}
    p, hits = v11_bundle.finish_probabilities(base, cache, np.array([.8, .2, .5]),
                                              np.array([.9, .1, .2]), np.zeros(3, bool))
    np.testing.assert_array_equal(p, [0, 1, 1])
    np.testing.assert_array_equal(hits["known_url"], [False, False, True])
