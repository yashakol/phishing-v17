"""Replay resource specialist fold scopes, diagnostic and complete saved-model inference."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import hashlib
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import guard, load_component
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v9_routing import subset_cache
from phishing_url.v10_training import sha
from phishing_url.v15_bundle import predict_v15
from phishing_url.v15_resources import ResourceContext, add_resource_evidence
from phishing_url.v12_expert import blend_expert, eligibility, make_expert


def assert_bundle(model, selection, rows, known):
    assert model["selection_sha256"] == sha(ROOT / "reports/v15/selection.json")
    assert model["threshold"] == selection["selected"]["threshold"]
    assert model["expert_weight"] == selection["selected"]["weight"]
    assert model["train_cache_sha256"] == selection["train_cache_sha256"]
    assert model["training_rows"] == model["base"]["training_rows"] == rows
    assert model["expert_training_rows"] == selection["eligible_rows"] == 5885
    assert model["expert"].get_params() == make_expert().get_params()
    assert model["feature_columns"] == selection["feature_columns"]
    assert model["resource_context_training_rows"] == model["resource_context"].training_rows_ == rows
    node = model
    while isinstance(node, dict):
        assert "external_reference" not in node
        if "base" not in node:
            break
        node = node["base"]
    assert node["known_urls"] == known


def replay_training(cache, extra, selection):
    with np.load(ROOT / "experiments/v15_resources/training_frame.npz", allow_pickle=False) as data:
        dev, y = data["indices"].copy(), data["labels"].copy()
        X = pd.DataFrame(data["values"], columns=data["columns"].tolist())
        eligible = data["eligibility"].copy()
    np.testing.assert_array_equal(dev, aligned_splits(ROOT, cache)["development"])
    np.testing.assert_array_equal(y, cache["labels"][dev])
    np.testing.assert_array_equal(eligible, eligibility(X.baseline_probability.to_numpy()))
    protocol_path = ROOT / "experiments/v15_resources/preregistered.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["expert_source_sha256"] == sha(ROOT / "phishing_url/v15_resources.py")
    assert protocol["source_sha256"] == sha(ROOT / "experiments/v15_resources/search.py")
    for path, digest in protocol["input_sha256"].items():
        assert sha(ROOT / path) == digest, path
    with np.load(ROOT/"experiments/v13_static/training_frame.npz",allow_pickle=False) as part:
        np.testing.assert_array_equal(part["indices"],dev)
        static=pd.DataFrame(part["values"],columns=part["columns"].tolist())
    resource_features=extra["features"].iloc[dev].reset_index(drop=True)
    with np.load(ROOT / "experiments/v15_resources/development_expert.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["indices"], dev)
        np.testing.assert_array_equal(data["labels"], y)
        p, fold_ids = data["probabilities"].copy(), data["folds"].copy()
    baseline = X.baseline_probability.to_numpy()
    for fold, (tr, va) in enumerate(StratifiedKFold(**protocol["splitter"]).split(np.zeros(len(y)), y)):
        fit, query = tr[eligible[tr]], va[eligible[va]]
        assert not set(dev[fit]) & set(dev[va])
        config = {"parameters": make_expert().get_params(), "source_sha256": protocol["expert_source_sha256"],
                  "base_expert_source_sha256": protocol["base_expert_source_sha256"],
                  "protocol_sha256": sha(protocol_path), "frame_columns": list(X.columns),
                  "context_guard": guard(cache,dev[tr],{"source_sha256":protocol["expert_source_sha256"]},selection["train_cache_sha256"])}
        expected = guard(cache, dev[fit], config, selection["train_cache_sha256"])
        expected["ordered_training_indices_sha256"] = hashlib.sha256(dev[fit].tobytes()).hexdigest()
        model = load_component(ROOT / f"experiments/v15_resources/models/fold_{fold}.joblib", expected)
        np.testing.assert_array_equal(fold_ids[va], fold)
        reconstructed=ResourceContext().fit(resource_features.iloc[tr],y[tr])
        assert reconstructed.counts_==model["context"].counts_
        assert reconstructed.prior_==model["context"].prior_==float(y[tr].mean())
        assert model["context"].training_rows_==len(tr)
        calculated=add_resource_evidence(static.iloc[query],resource_features.iloc[query],model["context"])
        np.testing.assert_array_equal(calculated.to_numpy(),X.iloc[query].to_numpy())
        np.testing.assert_array_equal(model["classifier"].predict_proba(calculated,thread_count=2)[:,1],p[query])
    np.testing.assert_array_equal(p[~eligible], baseline[~eligible])
    probability = blend_expert(baseline, p, selection["selected"]["weight"])
    assert measure(y, probability, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    np.savez_compressed(ROOT / "reports/v15/development_oof.npz", indices=dev, labels=y,
                        probabilities=probability, baseline=baseline, folds=fold_ids)
    fit_report = json.loads((ROOT / "reports/v15/expert_fit.json").read_text())
    stored = joblib.load(ROOT / "artifacts/v15/component_cache/development_expert.joblib")
    assert fit_report["guard"] == stored["guard"] == guard(cache, dev[eligible], fit_report["guard"]["config"], selection["train_cache_sha256"])
    assert fit_report["guard"]["config"]["frame_sha256"] == sha(ROOT / "experiments/v15_resources/training_frame.npz")
    return len(y)


def main():
    out = ROOT / "reports/v15"
    selection, protocol, metrics = [json.loads((out / name).read_text()) for name in
                                    ("selection.json", "integration_protocol.json", "metrics.json")]
    assert selection["gate_passed"] is True
    assert metrics["diagnostic_gate_passed"] and metrics["submission"] is not None
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    for name, digest in {**protocol["inputs_sha256"], **protocol["sources_sha256"]}.items():
        assert sha(ROOT / name) == digest, name
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra=joblib.load(ROOT/"data/extra_html_train.joblib")
    assert cache["urls"]==extra["urls"]
    np.testing.assert_array_equal(cache["labels"],extra["labels"])
    oof_rows = replay_training(cache, extra, selection)
    split = aligned_splits(ROOT, cache)
    dev, va = split["development"], split["holdout"]
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    development_labels = dict(zip([cache["urls"][int(i)] for i in dev], cache["labels"][dev].tolist(), strict=True))
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    for stage,indices in (("development",dev),("full",np.arange(len(cache["urls"]),dtype=np.int64))):
        stored=joblib.load(ROOT/f"artifacts/v15/component_cache/{stage}_resource_context.joblib")
        assert stored["guard"]==guard(cache,indices,stored["guard"]["config"],selection["train_cache_sha256"])
        reconstructed=ResourceContext().fit(extra["features"].iloc[indices],cache["labels"][indices])
        assert stored["model"].counts_==reconstructed.counts_ and stored["model"].prior_==reconstructed.prior_
        assert stored["model"].training_rows_==len(indices)
    model = joblib.load(ROOT / "artifacts/v15/holdout.joblib")
    assert_bundle(model, selection, len(dev), development_labels)
    validation, validation_extra = subset_cache(cache, va), subset_cache(extra, va)
    validation_url = url["features"].iloc[va].reset_index(drop=True)
    validation_y = cache["labels"][va].copy()
    del cache, extra, url, stored, reconstructed
    gc.collect()
    print(json.dumps({"event": "v15_verify_diagnostic_started"}), flush=True)
    p, hits = predict_v15(model, validation, validation_extra, validation_url)
    saved = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), validation_y)
    np.testing.assert_allclose(saved.probability.to_numpy(), p, rtol=0, atol=2e-15)
    assert measure(validation_y, p, model["threshold"]) == metrics["holdout"]
    assert int(hits["expert_routed"].sum()) == metrics["diagnostic_expert_rows"]
    del validation, validation_extra, validation_url, validation_y, model
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra, url = joblib.load(ROOT / "data/extra_html_test.joblib"), joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model = joblib.load(ROOT / "artifacts/v15/ensemble.joblib")
    assert_bundle(model, selection, 63900, labels)
    print(json.dumps({"event": "v15_verify_full_test_started"}), flush=True)
    p, hits = predict_v15(model, cache, extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv", (p >= model["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v15.csv").read_bytes()
    report = {"oof_mixture_direct_replay": "passed", "oof_rows": oof_rows,
        "expert_fold_model_scopes": "passed", "expert_training_scope": "passed", "expert_training_rows": 5885,
        "original_screen_gate_passed": True, "frozen_configuration": "passed", "resource_context_training_scopes": "passed",
        "holdout_model_replay": "passed", "holdout_rows": len(va),
        "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
        "submission": info, "model_sha256": sha(ROOT / "artifacts/v15/ensemble.joblib"),
        "external_test_labels_used": False}
    (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    with patch("socket.create_connection", side_effect=AssertionError("Network disabled")), \
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")), threadpool_limits(limits=2):
        main()
