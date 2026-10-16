"""Replay selected OOF, guarded coefficients and saved offline predictions."""

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
from experiments.v8_ensemble.search import main as search
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v8_bundle import predict_v8
from refit_v8_regularization import semantic_snapshot
from train_v6 import subset
from train_v8 import fitted_text


def network_blocked():
    return (patch("socket.create_connection", side_effect=AssertionError("Network disabled")),
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")))


def assert_bundle(model, selection):
    assert model["selection_sha256"] == sha(ROOT / "reports/v8/selection.json")
    assert model["v5_config"] == selection["selected"]["v5_config"]
    assert model["v6_config"] == selection["v6_config"]
    assert model["threshold"] == selection["selected"]["threshold"]
    assert model["resource_weight"] == 0 and model["resource_text"] is None


def main():
    out = ROOT / "reports/v8"
    selection = json.loads((out / "selection.json").read_text())
    assert selection["exploratory_text_selection_sha256"] == sha(ROOT / selection["exploratory_text_selection"])
    replay, p = search(ROOT / selection["text_oof"], selection["text_parameters"], selection["text_name"], save=False)
    for key in replay:
        assert replay[key] == selection[key], key
    with np.load(ROOT / f"experiments/v8_ensemble/{selection['text_name']}_development_oof.npz") as saved:
        np.testing.assert_allclose(saved["probabilities"], p, rtol=0, atol=1e-14)
        assert measure(saved["labels"], p, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    metrics = json.loads((out / "metrics.json").read_text())
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    splits = aligned_splits(ROOT, cache)
    hold = joblib.load(ROOT / "artifacts/v8/holdout.joblib")
    assert_bundle(hold, selection)
    assert hold["training_rows"] == len(splits["development"])
    checked = fitted_text(cache, splits["development"], "holdout", selection, reuse_only=True)
    assert semantic_snapshot(checked) == semantic_snapshot(hold["replacement_text"])
    va = splits["holdout"]
    a, b = network_blocked()
    with a, b:
        p, _ = predict_v8(hold, subset(cache, va), subset(extra, va), url["features"].iloc[va])
    saved = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), cache["labels"][va])
    np.testing.assert_allclose(saved.probability.to_numpy(), p, rtol=0, atol=2e-15)
    actual = measure(cache["labels"][va], p, hold["threshold"])
    for key in ("f1", "fp", "fn", "tp", "tn", "rows"):
        assert actual[key] == metrics["holdout"][key]
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    checked = fitted_text(cache, np.arange(len(cache["urls"])), "full", selection, reuse_only=True)
    checked_semantics = semantic_snapshot(checked)
    del checked, cache, extra, url, hold
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model = joblib.load(ROOT / "artifacts/v8/ensemble.joblib")
    assert_bundle(model, selection)
    assert model["training_rows"] == model["replacement_text"].training_rows_ == 63900
    assert semantic_snapshot(model["replacement_text"]) == checked_semantics
    original = model["base"]["base"]["base"]["base"]
    assert original["known_urls"] == labels
    assert all("external_reference" not in item for item in
               (model, model["base"], model["base"]["base"], model["base"]["base"]["base"], original))
    a, b = network_blocked()
    with a, b:
        p, hits = predict_v8(model, test, extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv", (p >= model["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v8.csv").read_bytes()
    result = {"oof_mixture_direct_replay": "passed", "oof_rows": 50977,
              "coefficient_training_guards": "passed", "holdout_model_replay": "passed", "holdout_rows": len(va),
              "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
              "submission": info, "known_training_urls": int(hits["known_url"].sum()),
              "model_sha256": sha(ROOT / "artifacts/v8/ensemble.joblib"), "external_test_labels_used": False}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
