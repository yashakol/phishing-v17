"""Replay selected OOF, frozen saved holdout and all 16000 offline outputs."""

import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.v6.screen_extra import sha
from experiments.v9_routing.search import main as replay_oof
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v9_routing import predict_v9, routing_mask, subset_cache


def blocked():
    return (patch("socket.create_connection", side_effect=AssertionError("Network disabled")),
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")))


def assert_bundle(model, selection):
    assert model["selection_sha256"] == sha(ROOT / "reports/v9/selection.json")
    assert model["routing"] == selection["selected"]["policy"]
    assert model["threshold"] == selection["selected"]["threshold"] == selection["baseline_v7"]["threshold"]
    assert model["train_cache_sha256"] == selection["train_cache_sha256"]


def main():
    out = ROOT / "reports/v9"
    selection = json.loads((out / "selection.json").read_text())
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_holdout.json").read_bytes()
    for key, path in (("routing_source_sha256", "phishing_url/v9_routing.py"),
        ("base_v7_model_sha256", "artifacts/v7/ensemble.joblib"),
        ("base_v7_holdout_model_sha256", "artifacts/v7/holdout.joblib"),
        ("train_cache_sha256", "data/content_train.joblib"),
        ("selection_source_sha256", selection["selection_source"]),
        ("oof_source_sha256", selection["oof_source"])):
        assert selection[key] == sha(ROOT / path), key
    result, oof = replay_oof(save=False)
    assert result["selected"] == selection["selected"] and result["gate_passed"]
    with np.load(ROOT / selection["oof_source"]) as saved:
        np.testing.assert_allclose(saved["probabilities"], oof, atol=1e-14, rtol=0)
        assert measure(saved["labels"], oof, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    metrics = json.loads((out / "metrics.json").read_text())
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    splits = aligned_splits(ROOT, cache)
    va = splits["holdout"]
    model = joblib.load(ROOT / "artifacts/v9/holdout.joblib")
    assert_bundle(model, selection)
    assert model["training_rows"] == len(splits["development"]) == 50977
    original = model["base"]["base"]["base"]["base"]
    tr = splits["development"]
    assert original["known_urls"] == dict(zip([cache["urls"][int(i)] for i in tr], cache["labels"][tr].tolist(), strict=True))
    a, b = blocked()
    with a, b:
        p, hits = predict_v9(model, subset_cache(cache, va), subset_cache(extra, va), url["features"].iloc[va])
    saved = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), cache["labels"][va])
    np.testing.assert_array_equal(saved.url_routed.to_numpy(), hits["url_routed"])
    np.testing.assert_allclose(saved.probability.to_numpy(), p, atol=2e-15, rtol=0)
    actual = measure(cache["labels"][va], p, model["threshold"])
    assert actual == metrics["holdout"]
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    del cache, extra, url, model, original
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model = joblib.load(ROOT / "artifacts/v9/ensemble.joblib")
    assert_bundle(model, selection)
    assert model["training_rows"] == model["base"]["training_rows"] == 63900
    original = model["base"]["base"]["base"]["base"]
    assert original["known_urls"] == labels
    assert all("external_reference" not in obj for obj in
               (model, model["base"], model["base"]["base"], model["base"]["base"]["base"], original))
    a, b = blocked()
    with a, b:
        p, hits = predict_v9(model, cache, extra, url["features"])
    np.testing.assert_array_equal(hits["url_routed"], routing_mask(cache, model["routing"]["selector"]))
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv", (p >= model["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v9.csv").read_bytes()
    report = {"oof_mixture_direct_replay": "passed", "oof_rows": 50977, "frozen_configuration": "passed",
        "holdout_model_replay": "passed", "holdout_rows": len(va),
        "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
        "submission": info, "known_training_urls": int(hits["known_url"].sum()),
        "url_routed_rows": int(hits["url_routed"].sum()), "model_sha256": sha(ROOT / "artifacts/v9/ensemble.joblib"),
        "external_test_labels_used": False}
    (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
