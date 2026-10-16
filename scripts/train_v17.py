"""Refit the validated v15 specialist on all eligible own-train rows, without retuning."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.v9_routing import subset_cache
from phishing_url.v10_training import sha
from phishing_url.v12_expert import blend_expert, eligibility, make_expert
from phishing_url.v15_bundle import resource_inputs
from phishing_url.v15_resources import ResourceContext

PHASE = "own_train"


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline own-train refit")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / ".venv"):
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(("external/", "reports/v2/")) or "reference" in path.name.lower():
        raise PermissionError("Published labels and reference code are forbidden")
    if PHASE == "own_train" and ("test" in path.name.lower() or rel.startswith("submissions/")):
        raise PermissionError("No test input before frozen own-train refit")
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
              (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing and not (path.is_relative_to(ROOT / "reports/v17") or path.is_relative_to(ROOT / "artifacts/v17") or
                        (PHASE == "export" and path == ROOT / "submissions/submission_content_v17.csv")):
        raise PermissionError("Refit write outside active version")


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def assert_original(bundle, cache, indices, parameters):
    assert bundle["training_rows"] == bundle["base"]["training_rows"] == len(indices)
    assert bundle["expert"].get_params() == parameters
    node = bundle
    while "base" in node:
        assert "external_reference" not in node
        node = node["base"]
    assert node["known_urls"] == {cache["urls"][int(i)]: int(cache["labels"][i]) for i in indices}


def main():
    global PHASE
    out, artifacts = ROOT / "reports/v17", ROOT / "artifacts/v17"
    out.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    sys.addaudithook(audit)
    assert not (out / "preregistered.json").exists(), "No unrecorded rerun or retuning"
    started = time.perf_counter()
    inputs = ("data/content_train.joblib", "data/content_url_train.joblib", "data/extra_html_train.joblib",
              "experiments/v15_resources/training_frame.npz", "reports/v15/selection.json",
              "reports/v15/metrics.json", "reports/v15/verification.json", "reports/v10/holdout_predictions.csv",
              "artifacts/v15/holdout.joblib", "artifacts/v15/ensemble.joblib")
    selection = json.loads((ROOT / "reports/v15/selection.json").read_text())
    historical = json.loads((ROOT / "reports/v15/metrics.json").read_text())
    verified = json.loads((ROOT / "reports/v15/verification.json").read_text())
    assert selection["gate_passed"] and historical["diagnostic_gate_passed"] and verified["offline_full_inference_byte_identical"]
    assert sha(ROOT / "artifacts/v15/ensemble.joblib") == verified["model_sha256"]
    protocol = {"action": "Refit validated v15 family on all uncertain organizer train rows",
        "parameters": make_expert().get_params(), "weight": selection["selected"]["weight"],
        "threshold": selection["selected"]["threshold"], "development_eligible_rows": selection["eligible_rows"],
        "fit_scope": "Development eligible rows use frozen OOF frame; remaining eligible own-train rows use frozen base/context fitted exclusively on development. No query row label enters its training features",
        "feature_policy": "Identical 354-column v15 numeric/resource schema; no new parameters, weights or threshold selection",
        "validation_status": "Historical v15 OOF/diagnostic scores retained as family evidence. Final refit includes diagnostic train rows; no new independent F1 is claimed",
        "test_policy": "Only unlabeled test after refit; no test probabilities used to adapt parameters",
        "source_sha256": sha(Path(__file__)),
        "sources_sha256": {name: sha(ROOT/name) for name in ("phishing_url/v12_expert.py", "phishing_url/v15_bundle.py", "phishing_url/v15_resources.py", "phishing_url/v12_bundle.py")},
        "inputs_sha256": {name: sha(ROOT/name) for name in inputs}, "external_test_labels_used": False}
    write(out / "preregistered.json", protocol)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    split = aligned_splits(ROOT, cache)
    dev, diagnostic, added = split["development"], split["holdout"], split["added"]
    assert len(cache["urls"]) == 63900 and len(dev) == 50977 and len(diagnostic) == 12745 and len(added) == 178
    with np.load(ROOT / "experiments/v15_resources/training_frame.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["indices"], dev)
        np.testing.assert_array_equal(stored["labels"], cache["labels"][dev])
        columns = stored["columns"].tolist()
        assert columns == selection["feature_columns"]
        dev_mask = stored["eligibility"].copy()
        dev_baseline = stored["values"][:, columns.index("baseline_probability")].copy()
        dev_values = stored["values"][dev_mask].copy()
    np.testing.assert_array_equal(dev_mask, eligibility(dev_baseline))
    assert int(dev_mask.sum()) == selection["eligible_rows"] == 5885
    previous = pd.read_csv(ROOT / "reports/v10/holdout_predictions.csv")
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), diagnostic)
    np.testing.assert_array_equal(previous.label.to_numpy(), cache["labels"][diagnostic])
    assert previous.url.tolist() == [cache["urls"][int(i)] for i in diagnostic]
    diagnostic_mask = eligibility(previous.probability.to_numpy())
    query_indices = np.concatenate([diagnostic[diagnostic_mask], added])
    assert not set(query_indices) & set(dev)
    extra, url = joblib.load(ROOT / "data/extra_html_train.joblib"), joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(extra["labels"], cache["labels"])
    query, query_extra = subset_cache(cache, query_indices), subset_cache(extra, query_indices)
    query["labels"] = query_extra["labels"] = None
    query_url = url["features"].iloc[query_indices].reset_index(drop=True)
    full_context = ResourceContext().fit(extra["features"], cache["labels"])
    dev_context = ResourceContext().fit(extra["features"].iloc[dev], cache["labels"][dev])
    small = {"urls": cache["urls"], "labels": cache["labels"]}
    expected_known = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    del cache, extra, url
    gc.collect()
    v15 = joblib.load(ROOT / "artifacts/v15/holdout.joblib")
    assert_original(v15, small, dev, protocol["parameters"])
    assert v15["resource_context"].counts_ == dev_context.counts_
    assert v15["resource_context"].prior_ == dev_context.prior_
    print(json.dumps({"event": "v17_own_train_features_started", "eligible_diagnostic_rows": int(diagnostic_mask.sum()), "added_rows": len(added)}), flush=True)
    frame, baseline, hits = resource_inputs(v15["base"], v15["resource_context"], query, query_extra, query_url)
    assert frame.columns.tolist() == columns
    assert not hits["known_url"].any()
    np.testing.assert_allclose(baseline[:int(diagnostic_mask.sum())], previous.probability.to_numpy()[diagnostic_mask], rtol=0, atol=2e-15)
    query_mask = eligibility(baseline)
    assert query_mask[:int(diagnostic_mask.sum())].all()
    indices = np.concatenate([dev[dev_mask], query_indices[query_mask]])
    values = np.concatenate([dev_values, frame.iloc[query_mask].to_numpy()])
    labels = small["labels"][indices]
    assert len(indices) == len(set(indices.tolist())) and values.shape == (len(indices), len(columns))
    assert np.isfinite(values).all()
    np.savez_compressed(out / "full_training_frame.npz", indices=indices, labels=labels, values=values,
                        columns=np.asarray(columns, dtype=str), development_indices=dev,
                        development_mask=dev_mask, query_indices=query_indices, query_eligibility=query_mask,
                        query_baseline=baseline)
    np.savez_compressed(out / "new_training_features.npz", indices=query_indices, labels=small["labels"][query_indices],
                        values=frame.to_numpy(), columns=np.asarray(columns, dtype=str), baseline=baseline)
    del v15, query, query_extra, query_url, frame, dev_values, dev_context
    gc.collect()
    config = {"parameters": protocol["parameters"], "protocol_sha256": sha(out/"preregistered.json"),
              "frame_sha256": sha(out/"full_training_frame.npz"), "source_sha256": protocol["source_sha256"],
              "feature_columns": columns}
    expected = guard(small, indices, config, protocol["inputs_sha256"]["data/content_train.joblib"])
    expected["ordered_training_indices_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
    print(json.dumps({"event": "v17_refit_started", "full_eligible_rows": len(indices), "development_eligible_rows": int(dev_mask.sum()),
                      "diagnostic_eligible_rows": int(diagnostic_mask.sum()), "added_eligible_rows": int(query_mask.sum()-diagnostic_mask.sum())}), flush=True)
    X = pd.DataFrame(values, columns=columns)
    expert = make_expert().fit(X, labels)
    path = artifacts / "component_cache/full_expert.joblib"
    assert not path.exists()
    save_component(path, expert, expected)
    restored = load_component(path, expected)
    np.testing.assert_array_equal(expert.predict_proba(X.iloc[:128], thread_count=2), restored.predict_proba(X.iloc[:128], thread_count=2))
    write(out / "expert_fit.json", {"training_rows": len(indices), "guard": expected, "parameters": expert.get_params(),
        "feature_columns": columns, "new_classifier_fits": 1, "training_rows_by_source": {"development": int(dev_mask.sum()),
        "diagnostic": int(diagnostic_mask.sum()), "added": int(query_mask.sum()-diagnostic_mask.sum())},
        "serialization_replay_rows": 128, "no_test_or_external_labels": True})
    del restored, X, values
    gc.collect()
    PHASE = "export"
    original = joblib.load(ROOT / "artifacts/v15/ensemble.joblib")
    assert_original(original, small, np.arange(len(small["urls"]), dtype=np.int64), protocol["parameters"])
    assert original["resource_context"].counts_ == full_context.counts_
    assert original["resource_context"].prior_ == full_context.prior_
    model = {**original, "expert": expert, "expert_training_rows": len(indices),
             "selection_sha256": sha(out/"preregistered.json"), "refit_version": "v17",
             "previous_v15_model_sha256": verified["model_sha256"]}
    assert model["threshold"] == protocol["threshold"] and model["expert_weight"] == protocol["weight"]
    joblib.dump(model, artifacts / "ensemble.joblib", compress=3)
    del original, small, full_context, expected_known
    gc.collect()
    report = {"version": "v17", "previous_user_reported_public_f1": 0.995082, "candidate_public_f1": None,
              "historical_v15_development": selection["selected"]["metrics"], "historical_v15_diagnostic": historical["holdout"],
              "new_independent_quality_evaluation": False, "own_train_rows": 63900, "expert_training_rows": len(indices),
              "previous_v15_expert_training_rows": 5885, "new_classifier_fits": 1, "threshold_and_weight_unchanged": True,
              "external_test_labels_used": False, "submission": None}
    write(out / "metrics.json", report)
    test, extra, url = [joblib.load(ROOT/name) for name in ("data/content_test.joblib", "data/extra_html_test.joblib", "data/content_url_test.joblib")]
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    print(json.dumps({"event": "v17_full_test_started", "rows": len(test["urls"])}), flush=True)
    frame, baseline, hits = resource_inputs(model["base"], model["resource_context"], test, extra, url["features"])
    previous_baseline = pd.read_csv(ROOT / "reports/v10/test_prediction_provenance.csv")
    np.testing.assert_allclose(baseline, previous_baseline.probability.to_numpy(), rtol=0, atol=2e-15)
    assert previous_baseline.url.tolist() == test["urls"]
    mask, q = eligibility(baseline), baseline.copy()
    q[mask] = expert.predict_proba(frame.iloc[mask], thread_count=2)[:, 1]
    probability = blend_expert(baseline, q, model["expert_weight"])
    predictions = (probability >= model["threshold"]).astype(np.int8)
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v17.csv", predictions)
    info["prediction_changes_vs_v15"] = int((predictions != pd.read_csv(ROOT / "submissions/submission_content_v15.csv").Predicted.to_numpy()).sum())
    info["prediction_changes_vs_v10"] = int((predictions != pd.read_csv(ROOT / "submissions/submission_content_v10.csv").Predicted.to_numpy()).sum())
    info["expert_routed_rows"] = int(mask.sum())
    pd.DataFrame({"Id": np.arange(len(predictions)), "url": test["urls"], "probability": probability,
                  "expert_routed": mask}).to_csv(out / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started)
    write(out / "metrics.json", report)
    print(json.dumps({"event": "v17_complete", "submission": info, "expert_training_rows": len(indices)}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
