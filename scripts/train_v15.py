"""Fit the fixed resource-evidence specialist and export after the frozen diagnostic gate."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
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
from phishing_url.evaluation import measure
from phishing_url.v9_routing import subset_cache
from phishing_url.v10_training import sha
from phishing_url.v15_bundle import resource_inputs
from phishing_url.v12_expert import blend_expert, eligibility, make_expert
from phishing_url.v15_resources import ResourceContext

PHASE = "diagnostic"


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline own-training specialist")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("external/", "reports/v2/")) or "reference" in path.name.lower():
            raise PermissionError("External labels and reference artifacts are forbidden")
        if PHASE == "diagnostic" and ("test" in path.name.lower() or rel.startswith("submissions/")):
            raise PermissionError("No test or submission inputs before the diagnostic gate")


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def make_bundle(base, expert, context, selection, rows):
    return {"base": base, "expert": expert, "resource_context": context, "resource_context_training_rows": context.training_rows_, "expert_weight": selection["selected"]["weight"],
        "threshold": selection["selected"]["threshold"], "training_rows": rows,
        "expert_training_rows": selection["eligible_rows"], "feature_columns": selection["feature_columns"],
        "selection_sha256": sha(ROOT / "reports/v15/selection.json"),
        "train_cache_sha256": selection["train_cache_sha256"]}


def main():
    global PHASE
    sys.addaudithook(audit)
    started = time.perf_counter()
    out, artifacts = ROOT / "reports/v15", ROOT / "artifacts/v15"
    artifacts.mkdir(exist_ok=True)
    assert not (out / "metrics.json").exists(), "No adaptive diagnostic reruns"
    selection = json.loads((out / "selection.json").read_text())
    protocol = json.loads((out / "integration_protocol.json").read_text())
    assert selection["gate_passed"] is True and selection["selected"]["weight"] == .75
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    assert protocol["export_gate"] == "diagnostic F1 > v10 and at least 3 fewer diagnostic errors; no parameter adaptation"
    for name, digest in {**protocol["inputs_sha256"], **protocol["sources_sha256"]}.items():
        assert sha(ROOT / name) == digest, name
    cache = joblib.load(ROOT / "data/content_train.joblib")
    split = aligned_splits(ROOT, cache)
    dev, va = split["development"], split["holdout"]
    assert len(dev) == 50977 and len(va) == 12745 and not set(dev) & set(va)
    with np.load(ROOT / "experiments/v15_resources/training_frame.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], dev)
        np.testing.assert_array_equal(saved["labels"], cache["labels"][dev])
        assert saved["columns"].tolist() == selection["feature_columns"]
        frame = pd.DataFrame(saved["values"], columns=saved["columns"].tolist())
        mask = saved["eligibility"].copy()
    np.testing.assert_array_equal(mask, eligibility(frame.baseline_probability.to_numpy()))
    fit_indices = dev[mask]
    assert len(fit_indices) == selection["eligible_rows"] == 5885
    config = {"parameters": make_expert().get_params(), "selection_sha256": sha(out / "selection.json"),
        "frame_sha256": sha(ROOT / "experiments/v15_resources/training_frame.npz"),
        "expert_source_sha256": sha(ROOT / "phishing_url/v15_resources.py")}
    expected = guard(cache, fit_indices, config, selection["train_cache_sha256"])
    assert not (artifacts / "component_cache/development_expert.joblib").exists()
    expert = make_expert().fit(frame.iloc[mask], cache["labels"][fit_indices])
    path = artifacts / "component_cache/development_expert.joblib"
    save_component(path, expert, expected)
    restored = load_component(path, expected)
    sample = frame.iloc[np.flatnonzero(mask)[:128]]
    np.testing.assert_array_equal(expert.predict_proba(sample, thread_count=2), restored.predict_proba(sample, thread_count=2))
    try:
        load_component(path, dict(expected, training_rows=-1))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid specialist scope accepted")
    write(out / "expert_fit.json", {"training_rows": len(fit_indices), "development_rows": len(dev),
        "new_classifier_fits": 1, "parameters": expert.get_params(), "guard": expected,
        "feature_columns": list(frame.columns), "serialization_replay_rows": 128,
        "wrong_guard_rejected": True, "no_holdout_or_added_rows_fitted": True})
    del frame, restored, sample
    gc.collect()
    extra, url = joblib.load(ROOT / "data/extra_html_train.joblib"), joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    context = ResourceContext().fit(extra["features"].iloc[dev], cache["labels"][dev])
    context_scope = guard(cache, dev, {"source_sha256": sha(ROOT/"phishing_url/v15_resources.py"),
        "extra_cache_sha256": sha(ROOT/"data/extra_html_train.joblib")}, selection["train_cache_sha256"])
    save_component(artifacts/"component_cache/development_resource_context.joblib",context,context_scope)
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    validation, validation_extra = subset_cache(cache, va), subset_cache(extra, va)
    base = joblib.load(ROOT / "artifacts/v10/holdout.joblib")
    assert base["training_rows"] == len(dev)
    model = make_bundle(base, expert, context, selection, len(dev))
    print(json.dumps({"event": "v15_diagnostic_components_started", "rows": len(va)}), flush=True)
    X, baseline, hits = resource_inputs(base, context, validation, validation_extra, url["features"].iloc[va])
    assert list(X.columns) == selection["feature_columns"]
    previous = pd.read_csv(ROOT / "reports/v10/holdout_predictions.csv")
    y = cache["labels"][va]
    assert previous.url.tolist() == validation["urls"]
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(previous.label.to_numpy(), y)
    np.testing.assert_allclose(baseline, previous.probability.to_numpy(), rtol=0, atol=1e-14)
    eligible, q = eligibility(baseline), baseline.copy()
    q[eligible] = expert.predict_proba(X.iloc[eligible], thread_count=2)[:, 1]
    p = blend_expert(baseline, q, model["expert_weight"])
    before, after = previous.prediction.to_numpy(), p >= model["threshold"]
    metrics = measure(y, p, model["threshold"])
    old = protocol["baseline_v10_diagnostic"]
    passed = metrics["f1"] > old["f1"] and metrics["fp"]+metrics["fn"] <= old["fp"]+old["fn"]-3
    report = {"selection": selection["selected"], "original_screen_gate_passed": True,
        "exploratory_followup": False, "holdout": metrics, "previous_v10_holdout": old,
        "holdout_fixed_errors": int(((before != y) & (after == y)).sum()),
        "holdout_new_errors": int(((before == y) & (after != y)).sum()),
        "diagnostic_gate_passed": bool(passed), "external_test_labels_used": False,
        "holdout_note": "Reused diagnostic split; candidate, threshold and export criterion frozen before scoring. Not an independent model-selection evaluation.",
        "new_active_classifier_fits": 1, "training_rows": len(cache["urls"]),
        "expert_training_rows": len(fit_indices), "resource_context_training_rows": len(dev), "diagnostic_expert_rows": int(eligible.sum()),
        "baseline_replay_max_difference": float(np.max(np.abs(baseline-previous.probability.to_numpy()))),
        "submission": None}
    pd.DataFrame({"cache_index": va, "url": validation["urls"], "label": y, "probability": p,
        "prediction": after, "previous_prediction": before}).to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump(model, artifacts / "holdout.joblib", compress=3)
    write(out / "metrics.json", report)
    print(json.dumps({"event": "v15_diagnostic_complete", "holdout": metrics, "gate_passed": bool(passed),
        "fixed_errors": report["holdout_fixed_errors"], "new_errors": report["holdout_new_errors"]}), flush=True)
    full_context = None
    if passed:
        full_context = ResourceContext().fit(extra["features"], cache["labels"])
        full_scope = guard(cache, np.arange(len(cache["urls"]),dtype=np.int64), context_scope["config"], selection["train_cache_sha256"])
        save_component(artifacts/"component_cache/full_resource_context.joblib",full_context,full_scope)
    del base, model, validation, validation_extra, X, extra, url
    gc.collect()
    if not passed:
        report["seconds"] = time.perf_counter()-started
        write(out / "metrics.json", report)
        return
    PHASE = "export"
    base = joblib.load(ROOT / "artifacts/v10/ensemble.joblib")
    assert base["training_rows"] == len(cache["urls"]) == 63900
    original = base["base"]["base"]["base"]["base"]["base"]
    assert original["known_urls"] == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    model = make_bundle(base, expert, full_context, selection, len(cache["urls"]))
    joblib.dump(model, artifacts / "ensemble.joblib", compress=3)
    write(artifacts / "config.json", selection)
    del base, original, cache
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    extra, url = joblib.load(ROOT / "data/extra_html_test.joblib"), joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    X, baseline, hits = resource_inputs(model["base"], model["resource_context"], test, extra, url["features"])
    assert list(X.columns) == selection["feature_columns"]
    previous = pd.read_csv(ROOT / "reports/v10/test_prediction_provenance.csv")
    assert previous.url.tolist() == test["urls"]
    np.testing.assert_allclose(baseline, previous.probability.to_numpy(), rtol=0, atol=1e-14)
    eligible, q = eligibility(baseline), baseline.copy()
    q[eligible] = expert.predict_proba(X.iloc[eligible], thread_count=2)[:, 1]
    p = blend_expert(baseline, q, model["expert_weight"])
    predicted = (p >= model["threshold"]).astype(np.int8)
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v15.csv", predicted)
    info.update({f"prediction_changes_vs_v{v}": int((predicted != pd.read_csv(
        ROOT / f"submissions/submission_content_v{v}.csv").Predicted.to_numpy()).sum()) for v in (7, 8, 9, 10)})
    info.update(known_training_urls=int(hits["known_url"].sum()), training_rule_matches=int(hits["training_match"].sum()),
                url_routed_rows=int(hits["url_routed"].sum()), expert_routed_rows=int(eligible.sum()))
    pd.DataFrame({"Id": np.arange(len(p)), "url": test["urls"], "probability": p,
        "url_routed": hits["url_routed"], "expert_routed": eligible}).to_csv(out / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started, full_resource_context_training_rows=model["resource_context"].training_rows_)
    write(out / "metrics.json", report)
    print(json.dumps({"event": "v15_complete", "submission": info, "holdout": metrics}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
