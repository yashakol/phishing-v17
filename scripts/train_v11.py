"""One explicitly exploratory fixed follow-up after the original screen failed."""

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
from phishing_url.v11_bundle import component_inputs, finish_probabilities
from phishing_url.v11_meta import LearnedEnsemble

PHASE = "diagnostic"


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline own-training ensemble")
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


def make_bundle(base, meta, selection, rows):
    return {"base": base, "meta": meta, "meta_weight": selection["selected"]["weight"],
            "threshold": selection["selected"]["threshold"], "training_rows": rows,
            "meta_training_rows": meta.training_rows_, "selection_sha256": sha(ROOT / "reports/v11/selection.json"),
            "train_cache_sha256": selection["train_cache_sha256"]}


def main():
    global PHASE
    sys.addaudithook(audit)
    started = time.perf_counter()
    out, artifacts = ROOT / "reports/v11", ROOT / "artifacts/v11"
    artifacts.mkdir(exist_ok=True)
    assert not (out / "metrics.json").exists(), "No adaptive diagnostic reruns"
    selection = json.loads((out / "selection.json").read_text())
    protocol = json.loads((out / "integration_protocol.json").read_text())
    assert selection["original_gate_passed"] is False
    assert selection["selected"]["parameters"]["kind"] == "linear" and selection["selected"]["weight"] == .25
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    assert protocol["export_gate"] == "diagnostic F1 > v10 and at least 3 fewer diagnostic errors; no parameter adaptation"
    for path, digest in protocol["inputs_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in protocol["sources_sha256"].items():
        assert sha(ROOT / path) == digest, path
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    dev, va = splits["development"], splits["holdout"]
    assert len(dev) == 50977 and len(va) == 12745 and not set(dev) & set(va)
    with np.load(ROOT / "experiments/v11_meta/training_frame.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], dev)
        np.testing.assert_array_equal(saved["labels"], cache["labels"][dev])
        columns = json.loads((ROOT / "experiments/v11_meta/selection.json").read_text())["frame_columns"]
        assert saved["values"].shape == (len(dev), len(columns))
        frame = pd.DataFrame(saved["values"], columns=columns)
    expected = guard(cache, dev, {"parameters": selection["selected"]["parameters"],
        "selection_sha256": sha(out / "selection.json"),
        "frame_sha256": sha(ROOT / "experiments/v11_meta/training_frame.npz"),
        "source_sha256": sha(ROOT / "phishing_url/v11_meta.py")}, selection["train_cache_sha256"])
    assert not (artifacts / "component_cache/development_meta.joblib").exists()
    meta = LearnedEnsemble(**selection["selected"]["parameters"]).fit(frame, cache["labels"][dev])
    save_component(artifacts / "component_cache/development_meta.joblib", meta, expected)
    restored = load_component(artifacts / "component_cache/development_meta.joblib", expected)
    np.testing.assert_array_equal(meta.predict_proba(frame.iloc[:128]), restored.predict_proba(frame.iloc[:128]))
    try:
        load_component(artifacts / "component_cache/development_meta.joblib", dict(expected, training_rows=-1))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid meta training scope accepted")
    write(out / "meta_fit.json", {"training_rows": len(dev), "new_classifier_fits": 1,
        "feature_columns": list(frame.columns), "parameters": meta.get_params(),
        "guard": expected, "serialization_replay_rows": 128, "wrong_guard_rejected": True,
        "no_holdout_or_added_rows_in_meta_fit": True})
    del frame, restored
    gc.collect()
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    validation, validation_extra = subset_cache(cache, va), subset_cache(extra, va)
    base = joblib.load(ROOT / "artifacts/v10/holdout.joblib")
    assert base["training_rows"] == len(dev)
    model = make_bundle(base, meta, selection, len(dev))
    print(json.dumps({"event": "v11_diagnostic_components_started", "rows": len(va)}), flush=True)
    X, raw, extra_p, routed = component_inputs(base, validation, validation_extra, url["features"].iloc[va])
    baseline, _ = finish_probabilities(base, validation, raw, extra_p, routed)
    previous = pd.read_csv(ROOT / "reports/v10/holdout_predictions.csv")
    y = cache["labels"][va]
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(previous.label.to_numpy(), y)
    assert previous.url.tolist() == validation["urls"]
    np.testing.assert_allclose(baseline, previous.probability.to_numpy(), rtol=0, atol=1e-14)
    w = model["meta_weight"]
    p, hits = finish_probabilities(base, validation, (1-w)*raw+w*meta.predict_proba(X), extra_p, routed)
    before, after = previous.prediction.to_numpy(), p >= model["threshold"]
    metrics = measure(y, p, model["threshold"])
    old = protocol["baseline_v10_diagnostic"]
    passed = metrics["f1"] > old["f1"] and metrics["fp"]+metrics["fn"] <= old["fp"]+old["fn"]-3
    report = {"selection": selection["selected"], "original_screen_gate_passed": False,
        "exploratory_followup": True, "holdout": metrics, "previous_v10_holdout": old,
        "holdout_fixed_errors": int(((before != y) & (after == y)).sum()),
        "holdout_new_errors": int(((before == y) & (after != y)).sum()),
        "diagnostic_gate_passed": bool(passed), "external_test_labels_used": False,
        "holdout_note": "Reused diagnostic split; candidate, threshold and export criterion frozen before scoring. This is not independent model-selection evaluation.",
        "new_active_classifier_fits": 1, "training_rows": len(cache["urls"]),
        "meta_training_rows": len(dev), "baseline_replay_max_difference": float(np.max(np.abs(baseline-previous.probability.to_numpy()))),
        "submission": None}
    pd.DataFrame({"cache_index": va, "url": validation["urls"], "label": y, "probability": p,
        "prediction": after, "previous_prediction": before}).to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump(model, artifacts / "holdout.joblib", compress=3)
    write(out / "metrics.json", report)
    print(json.dumps({"event": "v11_diagnostic_complete", "holdout": metrics, "gate_passed": bool(passed),
        "fixed_errors": report["holdout_fixed_errors"], "new_errors": report["holdout_new_errors"]}), flush=True)
    del base, model, validation, validation_extra, X, raw, extra_p, baseline, extra, url
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
    model = make_bundle(base, meta, selection, len(cache["urls"]))
    joblib.dump(model, artifacts / "ensemble.joblib", compress=3)
    write(artifacts / "config.json", selection)
    del base, original, cache
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    X, raw, extra_p, routed = component_inputs(model["base"], test, extra, url["features"])
    old_p, _ = finish_probabilities(model["base"], test, raw, extra_p, routed)
    previous = pd.read_csv(ROOT / "reports/v10/test_prediction_provenance.csv")
    assert previous.url.tolist() == test["urls"]
    np.testing.assert_allclose(old_p, previous.probability.to_numpy(), rtol=0, atol=1e-14)
    p, hits = finish_probabilities(model["base"], test, (1-w)*raw+w*meta.predict_proba(X), extra_p, routed)
    predicted = (p >= model["threshold"]).astype(np.int8)
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v11.csv", predicted)
    info.update({f"prediction_changes_vs_v{v}": int((predicted != pd.read_csv(
        ROOT / f"submissions/submission_content_v{v}.csv").Predicted.to_numpy()).sum()) for v in (7, 8, 9, 10)})
    info.update(known_training_urls=int(hits["known_url"].sum()),
        training_rule_matches=int(hits["training_match"].sum()), url_routed_rows=int(hits["url_routed"].sum()))
    pd.DataFrame({"Id": np.arange(len(p)), "url": test["urls"], "probability": p,
        "url_routed": routed}).to_csv(out / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started)
    write(out / "metrics.json", report)
    print(json.dumps({"event": "v11_complete", "submission": info, "holdout": metrics}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
