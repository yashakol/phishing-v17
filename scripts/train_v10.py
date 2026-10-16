"""Evaluate the frozen calibrated-text candidate and export only after gates."""

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

from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v10_bundle import predict_v10
from phishing_url.v10_training import fit_margin, sha
from phishing_url.v9_routing import subset_cache


def bundle(base, margin, selection, rows):
    return {"base": base, "margin": margin, "text_replacement": selection["selected"]["text_replacement"],
            "threshold": selection["selected"]["threshold"], "training_rows": rows,
            "selection_sha256": sha(ROOT / "reports/v10/selection.json"),
            "train_cache_sha256": selection["train_cache_sha256"]}


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline own-training solution")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("external/", "reports/v2/")) or "reference" in path.name.lower():
            raise PermissionError("External labels and reference artifacts are forbidden")


def main():
    sys.addaudithook(audit)
    started = time.perf_counter()
    out, artifacts = ROOT / "reports/v10", ROOT / "artifacts/v10"
    artifacts.mkdir(exist_ok=True)
    assert not (out / "metrics.json").exists(), "No adaptive diagnostic reruns"
    selection = json.loads((out / "selection.json").read_text())
    assert selection["gate_passed"] is True
    protocol = json.loads((out / "integration_protocol.json").read_text())
    assert protocol["export_gate"] == "diagnostic F1 >= v9 and diagnostic errors <= v9; no parameter adaptation"
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    for path, digest in protocol["sources_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in protocol["base_files_sha256"].items():
        assert sha(ROOT / path) == digest, path
    assert sha(ROOT / "data/content_train.joblib") == selection["train_cache_sha256"]
    assert sha(ROOT / "phishing_url/v9_margin.py") == selection["model_source_sha256"]
    assert sha(ROOT / "experiments/v10_margin/development_oof.npz") == selection["text_oof_sha256"]
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["development"], splits["holdout"]
    assert len(tr) == 50977 and len(va) == 12745 and not set(tr) & set(va)
    margin = fit_margin(ROOT, cache, tr, "holdout", selection)
    base = joblib.load(ROOT / "artifacts/v9/holdout.joblib")
    assert base["training_rows"] == len(tr)
    model = bundle(base, margin, selection, len(tr))
    validation, validation_extra = subset_cache(cache, va), subset_cache(extra, va)
    p, hits = predict_v10(model, validation, validation_extra, url["features"].iloc[va])
    y = cache["labels"][va]
    previous = pd.read_csv(ROOT / "reports/v9/holdout_predictions.csv")
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(previous.label.to_numpy(), y)
    assert previous.url.tolist() == validation["urls"]
    before, after = previous.prediction.to_numpy(), p >= model["threshold"]
    metrics = measure(y, p, model["threshold"])
    old_metrics = protocol["baseline_v9_diagnostic"]
    passed = (metrics["f1"] >= old_metrics["f1"] and
              metrics["fp"]+metrics["fn"] <= old_metrics["fp"]+old_metrics["fn"])
    report = {"selection": selection["selected"], "holdout": metrics, "previous_v9_holdout": old_metrics,
              "holdout_fixed_errors": int(((before != y) & (after == y)).sum()),
              "holdout_new_errors": int(((before == y) & (after != y)).sum()),
              "diagnostic_gate_passed": bool(passed), "external_test_labels_used": False,
              "holdout_note": "Reused diagnostic split, candidate and export criterion frozen before scoring; not independent model-selection evaluation",
              "new_active_classifier_fits": 4, "training_rows": len(cache["urls"]),
              "submission": None}
    pd.DataFrame({"cache_index": va, "url": validation["urls"], "label": y, "probability": p,
                  "prediction": after, "previous_prediction": before}).to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump(model, artifacts / "holdout.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"event": "v10_diagnostic_complete", "holdout": metrics,
                      "gate_passed": bool(passed), "fixed_errors": report["holdout_fixed_errors"],
                      "new_errors": report["holdout_new_errors"]}), flush=True)
    del margin, base, model, validation, validation_extra
    gc.collect()
    if not passed:
        report["seconds"] = time.perf_counter()-started
        (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
        return
    tr = np.arange(len(cache["urls"]), dtype=np.int64)
    margin = fit_margin(ROOT, cache, tr, "full", selection)
    base = joblib.load(ROOT / "artifacts/v9/ensemble.joblib")
    assert base["training_rows"] == len(tr) == 63900
    original = base["base"]["base"]["base"]["base"]
    assert original["known_urls"] == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    model = bundle(base, margin, selection, len(tr))
    joblib.dump(model, artifacts / "ensemble.joblib", compress=3)
    (artifacts / "config.json").write_text(json.dumps(selection, indent=2)+"\n")
    del cache, extra, url, base, margin, original
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    p, hits = predict_v10(model, test, extra, url["features"])
    predicted = (p >= model["threshold"]).astype(np.int8)
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v10.csv", predicted)
    info.update({f"prediction_changes_vs_v{v}": int((predicted != pd.read_csv(
        ROOT / f"submissions/submission_content_v{v}.csv").Predicted.to_numpy()).sum()) for v in (7, 8, 9)})
    info.update(known_training_urls=int(hits["known_url"].sum()),
                training_rule_matches=int(hits["training_match"].sum()),
                url_routed_rows=int(hits["url_routed"].sum()))
    pd.DataFrame({"Id": np.arange(len(p)), "url": test["urls"], "probability": p,
                  "url_routed": hits["url_routed"]}).to_csv(out / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started, new_active_classifier_fits=8)
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"event": "v10_complete", "submission": info, "holdout": metrics,
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
