"""Replay specialist pair fold scopes, diagnostic and complete saved-model inference."""

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
from phishing_url.v14_bundle import predict_v14
from phishing_url.v12_expert import blend_expert, eligibility, make_expert


def assert_bundle(model, selection, rows, known):
    assert model["selection_sha256"] == sha(ROOT / "reports/v14/selection.json")
    assert model["threshold"] == selection["selected"]["threshold"]
    assert model["expert_weight"] == selection["selected"]["weight"]
    assert model["train_cache_sha256"] == selection["train_cache_sha256"]
    assert model["training_rows"] == model["base"]["training_rows"] == rows
    assert model["expert_training_rows"] == selection["eligible_rows"] == 5885
    assert model["structural_expert"].get_params() == model["static_expert"].get_params() == make_expert().get_params()
    assert model["structural_columns"] == selection["structural_columns"]
    assert model["static_columns"] == selection["static_columns"]
    node = model
    while isinstance(node, dict):
        assert "external_reference" not in node
        if "base" not in node:
            break
        node = node["base"]
    assert node["known_urls"] == known


def replay_family(cache, selection, folder):
    directory = ROOT / "experiments" / folder
    with np.load(directory / "training_frame.npz", allow_pickle=False) as data:
        dev, y = data["indices"].copy(), data["labels"].copy()
        X = pd.DataFrame(data["values"], columns=data["columns"].tolist())
        eligible = data["eligibility"].copy()
    np.testing.assert_array_equal(dev, aligned_splits(ROOT, cache)["development"])
    np.testing.assert_array_equal(y, cache["labels"][dev])
    np.testing.assert_array_equal(eligible, eligibility(X.baseline_probability.to_numpy()))
    protocol_path = directory / "preregistered.json"
    protocol = json.loads(protocol_path.read_text())
    source = "phishing_url/v12_expert.py" if folder == "v12_expert" else "phishing_url/v13_features.py"
    assert protocol["expert_source_sha256"] == sha(ROOT / source)
    assert protocol["source_sha256"] == sha(directory / "search.py")
    for path, digest in protocol["input_sha256"].items():
        assert sha(ROOT / path) == digest, path
    with np.load(directory / "development_expert.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["indices"], dev)
        np.testing.assert_array_equal(data["labels"], y)
        p, fold_ids = data["probabilities"].copy(), data["folds"].copy()
    baseline = X.baseline_probability.to_numpy()
    for fold, (tr, va) in enumerate(StratifiedKFold(**protocol["splitter"]).split(np.zeros(len(y)), y)):
        fit, query = tr[eligible[tr]], va[eligible[va]]
        assert not set(dev[fit]) & set(dev[va])
        config = {"parameters": make_expert().get_params(), "source_sha256": protocol["expert_source_sha256"],
                  "protocol_sha256": sha(protocol_path), "frame_columns": list(X.columns)}
        if folder == "v13_static":
            config["base_expert_source_sha256"] = protocol["base_expert_source_sha256"]
        expected = guard(cache, dev[fit], config, selection["train_cache_sha256"])
        expected["ordered_training_indices_sha256"] = hashlib.sha256(dev[fit].tobytes()).hexdigest()
        model = load_component(directory / f"models/fold_{fold}.joblib", expected)
        np.testing.assert_array_equal(fold_ids[va], fold)
        np.testing.assert_array_equal(model.predict_proba(X.iloc[query], thread_count=2)[:, 1], p[query])
    np.testing.assert_array_equal(p[~eligible], baseline[~eligible])
    return dev, y, p, baseline, fold_ids, eligible


def replay_training(cache, selection):
    dev, y, primary, baseline, folds, eligible = replay_family(cache, selection, "v12_expert")
    dev2, y2, secondary, baseline2, folds2, eligible2 = replay_family(cache, selection, "v13_static")
    for a,b in ((dev,dev2),(y,y2),(baseline,baseline2),(folds,folds2),(eligible,eligible2)):
        np.testing.assert_array_equal(a,b)
    probability = blend_expert(baseline, .5*primary+.5*secondary, selection["selected"]["weight"])
    assert measure(y, probability, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    with np.load(ROOT / "experiments/v14_pair/selected_oof.npz") as data:
        np.testing.assert_array_equal(data["probabilities"], probability)
    np.savez_compressed(ROOT / "reports/v14/development_oof.npz", indices=dev, labels=y,
                        probabilities=probability, baseline=baseline, folds=folds)
    for name, stage in (("v12", "development_expert"), ("v14", "development_static_expert")):
        fit_name = "expert_fit.json" if name=="v12" else "static_expert_fit.json"
        fit_report=json.loads((ROOT / f"reports/{name}/{fit_name}").read_text())
        stored=joblib.load(ROOT / f"artifacts/{name}/component_cache/{stage}.joblib")
        assert fit_report["guard"]==stored["guard"]==guard(cache,dev[eligible],fit_report["guard"]["config"],selection["train_cache_sha256"])
        frame_path="experiments/v12_expert/training_frame.npz" if name=="v12" else "experiments/v13_static/training_frame.npz"
        assert fit_report["guard"]["config"]["frame_sha256"]==sha(ROOT/frame_path)
    return len(y)


def main():
    out = ROOT / "reports/v14"
    selection, protocol, metrics = [json.loads((out / name).read_text()) for name in
                                    ("selection.json", "integration_protocol.json", "metrics.json")]
    assert selection["gate_passed"] is True
    assert metrics["diagnostic_gate_passed"] and metrics["submission"] is not None
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    for name, digest in {**protocol["inputs_sha256"], **protocol["sources_sha256"]}.items():
        assert sha(ROOT / name) == digest, name
    cache = joblib.load(ROOT / "data/content_train.joblib")
    oof_rows = replay_training(cache, selection)
    split = aligned_splits(ROOT, cache)
    dev, va = split["development"], split["holdout"]
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    development_labels = dict(zip([cache["urls"][int(i)] for i in dev], cache["labels"][dev].tolist(), strict=True))
    extra, url = joblib.load(ROOT / "data/extra_html_train.joblib"), joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    model = joblib.load(ROOT / "artifacts/v14/holdout.joblib")
    assert_bundle(model, selection, len(dev), development_labels)
    print(json.dumps({"event": "v14_verify_diagnostic_started"}), flush=True)
    p, hits = predict_v14(model, subset_cache(cache, va), subset_cache(extra, va), url["features"].iloc[va])
    saved = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), cache["labels"][va])
    np.testing.assert_allclose(saved.probability.to_numpy(), p, rtol=0, atol=2e-15)
    assert measure(cache["labels"][va], p, model["threshold"]) == metrics["holdout"]
    assert int(hits["expert_routed"].sum()) == metrics["diagnostic_expert_rows"]
    del cache, extra, url, model
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra, url = joblib.load(ROOT / "data/extra_html_test.joblib"), joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model = joblib.load(ROOT / "artifacts/v14/ensemble.joblib")
    assert_bundle(model, selection, 63900, labels)
    print(json.dumps({"event": "v14_verify_full_test_started"}), flush=True)
    p, hits = predict_v14(model, cache, extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv", (p >= model["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v14.csv").read_bytes()
    report = {"oof_mixture_direct_replay": "passed", "oof_rows": oof_rows,
        "expert_fold_model_scopes": "passed", "expert_training_scope": "passed", "expert_training_rows": 5885,
        "original_screen_gate_passed": True, "frozen_configuration": "passed",
        "holdout_model_replay": "passed", "holdout_rows": len(va),
        "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
        "submission": info, "model_sha256": sha(ROOT / "artifacts/v14/ensemble.joblib"),
        "external_test_labels_used": False}
    (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    with patch("socket.create_connection", side_effect=AssertionError("Network disabled")), \
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")), threadpool_limits(limits=2):
        main()
