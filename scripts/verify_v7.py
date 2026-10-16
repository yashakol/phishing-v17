"""Replay selected supervised OOF, diagnostic holdout and final offline output."""

import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.v6.screen_extra import sha
from experiments.v7_astra_selection import main as replay_oof
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v7_bundle import predict_v7
from train_v6 import subset
from train_v7 import fit_added


def no_network():
    return (patch("socket.create_connection", side_effect=AssertionError("Network disabled")),
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")))


def same_added_models(bundle, checked, training_rows):
    text = bundle["replacement_text"]
    if text is not None:
        checkpoint = checked["replacement_text"]
        assert text.training_rows_ == text.vocabulary_fit_rows_ == training_rows
        assert text.get_params() == checkpoint.get_params()
        assert text.vectorizers_.keys() == checkpoint.vectorizers_.keys()
        np.testing.assert_array_equal(text.classifier_.coef_, checkpoint.classifier_.coef_)
        np.testing.assert_array_equal(text.classifier_.intercept_, checkpoint.classifier_.intercept_)
        for name, vectorizer in text.vectorizers_.items():
            assert vectorizer.vocabulary_ == checkpoint.vectorizers_[name].vocabulary_
            np.testing.assert_array_equal(vectorizer.idf_, checkpoint.vectorizers_[name].idf_)
    hierarchy = bundle["hierarchy"]
    if hierarchy is not None:
        assert hierarchy.training_rows_ == training_rows
        assert hierarchy.evidence_ == checked["hierarchy"].evidence_
        assert hierarchy.prior_ == checked["hierarchy"].prior_


def main():
    output = ROOT / "reports/v7"
    selection = json.loads((output / "selection.json").read_text())
    report, p = replay_oof(ROOT / selection["text_oof"], ROOT / selection["text_selection"], save=False)
    assert report == selection
    with np.load(output / "development_oof.npz") as oof:
        np.testing.assert_allclose(p, oof["probabilities"], atol=1e-14, rtol=0)
        assert measure(oof["labels"], p, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    metrics = json.loads((output / "metrics.json").read_text())
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    splits = aligned_splits(ROOT, cache)
    hold = joblib.load(ROOT / "artifacts/v7/holdout.joblib")
    assert hold["config"] == selection["selected"]["config"]
    assert hold["selection_sha256"] == sha(output / "selection.json")
    assert hold["training_rows"] == len(splits["development"])
    checked = fit_added(cache, splits["development"], "holdout", selection, False, reuse_only=True)
    same_added_models(hold, checked, len(splits["development"]))
    dev = splits["development"]
    assert hold["base"]["base"]["base"]["known_urls"] == dict(
        zip([cache["urls"][int(i)] for i in dev], cache["labels"][dev].tolist(), strict=True))
    va = splits["holdout"]
    a, b = no_network()
    with a, b:
        p, _ = predict_v7(hold, subset(cache, va), subset(extra, va), url["features"].iloc[va])
    saved = pd.read_csv(output / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), cache["labels"][va])
    np.testing.assert_allclose(saved.probability.to_numpy(), p, rtol=0, atol=2e-15)
    actual = measure(cache["labels"][va], p, hold["threshold"])
    for key in ("f1", "fp", "fn", "tp", "tn", "rows"):
        assert actual[key] == metrics["holdout"][key]
    training_labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    full_checked = fit_added(cache, np.arange(len(cache["urls"])), "full", selection, False, reuse_only=True)
    del cache, extra, url, hold, checked
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model_path = ROOT / "artifacts/v7/ensemble.joblib"
    model = joblib.load(model_path)
    assert model["selection_sha256"] == sha(output / "selection.json")
    assert model["config"] == selection["selected"]["config"]
    assert model["threshold"] == selection["selected"]["threshold"]
    assert model["training_rows"] == 63900
    same_added_models(model, full_checked, model["training_rows"])
    del full_checked
    learned = model["base"]["base"]["base"]
    assert learned["known_urls"] == training_labels
    assert all("external_reference" not in item for item in (model, model["base"], model["base"]["base"], learned))
    a, b = no_network()
    with a, b:
        p, hits = predict_v7(model, test, extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", output / "reproduced_submission.csv", (p >= model["threshold"]).astype(np.int8))
    assert (output / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v7.csv").read_bytes()
    result = {"oof_mixture_direct_replay": "passed", "oof_rows": 50977,
              "added_component_training_scope_guards": "passed", "holdout_model_replay": "passed", "holdout_rows": len(va),
              "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
              "submission": info, "model_sha256": sha(model_path), "external_test_labels_used": False,
              "known_training_urls": int(hits["known_url"].sum()), "new_astra_calls": 1}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
