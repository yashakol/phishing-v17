"""Replay the exploratory development result and the complete saved models."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.v11_meta.search import prepare
from experiments.v8_astra_regularization.resume import scope
from experiments.v9_routing.search import finish
from phishing_url.content_checkpoints import guard, load_component
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v9_routing import routing_mask, subset_cache
from phishing_url.v10_training import sha
from phishing_url.v11_bundle import predict_v11


def check_oof(selection):
    inputs, cache, X, raw, baseline = prepare()
    dev, y = inputs["indices"], inputs["labels"]
    protocol_path = ROOT / "experiments/v11_meta/preregistered.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["model_source_sha256"] == sha(ROOT / "phishing_url/v11_meta.py")
    assert protocol["source_sha256"] == sha(ROOT / "experiments/v11_meta/search.py")
    with np.load(ROOT / "experiments/v11_meta/candidate_0.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], dev)
        np.testing.assert_array_equal(saved["labels"], y)
        p = saved["probabilities"].copy()
        fold_ids = saved["folds"].copy()
    for fold, (tr, va) in enumerate(inputs["folds"]):
        assert not set(dev[tr]) & set(dev[va])
        expected = scope(cache, dev[tr], {"parameters": selection["selected"]["parameters"],
            "source_sha256": protocol["model_source_sha256"], "frame_columns": list(X.columns),
            "protocol_sha256": sha(protocol_path)}, inputs["digest"])
        model = load_component(ROOT / f"experiments/v11_meta/models/candidate_0_fold_{fold}.joblib", expected)
        assert model.training_rows_ == len(tr)
        np.testing.assert_array_equal(fold_ids[va], fold)
        np.testing.assert_allclose(model.predict_proba(X.iloc[va]), p[va], rtol=0, atol=2e-15)
    weight = selection["selected"]["weight"]
    probability = finish(inputs, (1-weight)*raw+weight*p)
    assert measure(y, probability, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    np.savez_compressed(ROOT / "reports/v11/development_oof.npz", indices=dev, labels=y,
                        probabilities=probability, baseline=baseline, folds=fold_ids)
    return len(y)


def assert_bundle(model, selection, rows, known):
    assert model["selection_sha256"] == sha(ROOT / "reports/v11/selection.json")
    assert model["threshold"] == selection["selected"]["threshold"]
    assert model["meta_weight"] == selection["selected"]["weight"]
    assert model["train_cache_sha256"] == selection["train_cache_sha256"]
    assert model["training_rows"] == model["base"]["training_rows"] == rows
    assert model["meta_training_rows"] == model["meta"].training_rows_ == 50977
    assert model["meta"].get_params() == selection["selected"]["parameters"]
    node = model
    while isinstance(node, dict):
        assert "external_reference" not in node
        if "base" not in node:
            break
        node = node["base"]
    assert node["known_urls"] == known


def main():
    out = ROOT / "reports/v11"
    selection = json.loads((out / "selection.json").read_text())
    protocol = json.loads((out / "integration_protocol.json").read_text())
    assert selection["original_gate_passed"] is False
    assert selection["exploratory_followup"] is True
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    for name, digest in {**protocol["inputs_sha256"], **protocol["sources_sha256"]}.items():
        assert sha(ROOT / name) == digest, name
    oof_rows = check_oof(selection)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["original_screen_gate_passed"] is False and metrics["diagnostic_gate_passed"] is True
    cache = joblib.load(ROOT / "data/content_train.joblib")
    split = aligned_splits(ROOT, cache)
    dev, va = split["development"], split["holdout"]
    fit = json.loads((out / "meta_fit.json").read_text())
    stored = joblib.load(ROOT / "artifacts/v11/component_cache/development_meta.joblib")
    assert stored["guard"] == fit["guard"] == guard(cache, dev, fit["guard"]["config"], selection["train_cache_sha256"])
    assert fit["guard"]["config"]["frame_sha256"] == sha(ROOT / "experiments/v11_meta/training_frame.npz")
    assert stored["model"].training_rows_ == len(dev)
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    development_labels = dict(zip([cache["urls"][int(i)] for i in dev], cache["labels"][dev].tolist(), strict=True))
    extra, url = joblib.load(ROOT / "data/extra_html_train.joblib"), joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    model = joblib.load(ROOT / "artifacts/v11/holdout.joblib")
    assert_bundle(model, selection, len(dev), development_labels)
    p, _ = predict_v11(model, subset_cache(cache, va), subset_cache(extra, va), url["features"].iloc[va])
    saved = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), cache["labels"][va])
    np.testing.assert_allclose(saved.probability.to_numpy(), p, rtol=0, atol=2e-15)
    assert measure(cache["labels"][va], p, model["threshold"]) == metrics["holdout"]
    del cache, extra, url, model, stored
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model = joblib.load(ROOT / "artifacts/v11/ensemble.joblib")
    assert_bundle(model, selection, 63900, labels)
    p, hits = predict_v11(model, cache, extra, url["features"])
    np.testing.assert_array_equal(hits["url_routed"], routing_mask(cache, model["base"]["base"]["routing"]["selector"]))
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv",
                            (p >= model["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v11.csv").read_bytes()
    report = {"oof_mixture_direct_replay": "passed", "oof_rows": oof_rows,
        "meta_fold_model_scopes": "passed", "meta_training_scope": "passed", "meta_training_rows": len(dev),
        "original_screen_gate_passed": False, "frozen_exploratory_configuration": "passed",
        "holdout_model_replay": "passed", "holdout_rows": len(va),
        "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
        "submission": info, "model_sha256": sha(ROOT / "artifacts/v11/ensemble.joblib"),
        "external_test_labels_used": False}
    (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    with patch("socket.create_connection", side_effect=AssertionError("Network disabled")), \
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")), threadpool_limits(limits=2):
        main()
