"""Exactly one calibrated-margin screen; original inner train/validation only."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.v8_astra_regularization.resume import checked_base, prediction, scope, sha, semantic_hash
from phishing_url.content_checkpoints import data_digest, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v9_margin import CalibratedMarginText
import joblib


def metrics(y, p):
    cut, _ = best_threshold(y, p)
    result = measure(y, p, cut)
    result["errors"] = result["fp"]+result["fn"]
    return result


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "preregistered.json").exists(), "No silent rescreening"
    parameters = CalibratedMarginText().get_params()
    protocol = {"parameters": parameters, "configurations": 1, "classifier_fits": 4,
        "calibration": "Three internal training-only margin fits plus one fit on complete outer training side",
        "representation": "Frozen guarded C16 inner train vocabulary/IDF, no vocabulary or validation fitting",
        "baseline": "Half guarded C16 text plus half guarded deep numeric",
        "blends": [0., .25, .5, 1.],
        "gate": "At least two net fewer inner mixture errors at both frozen baseline threshold and selected F1 threshold",
        "scope": "Original development inner_train and inner_validation only; no holdout, added rows or test",
        "limitation": "Reused inner screening split. Internal calibration vocabulary sees complete outer training side; outer validation never used to fit representation or coefficients.",
        "source_sha256": sha(Path(__file__)), "model_source_sha256": sha(ROOT / "phishing_url/v9_margin.py")}
    (OUT / "preregistered.json").write_text(json.dumps(protocol, indent=2)+"\n")
    digest = data_digest(ROOT)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    assert len(tr) == 34663 and len(va) == 6118 and not set(tr) & set(va)
    assert set(tr) | set(va) <= set(splits["development"])
    base, origin = checked_base(cache, tr, "inner_full_vocabulary", digest)
    old = prediction("experiments/v7_astra/inner_full_vocabulary.npz", va, cache["labels"][va], tr)
    cat = prediction("experiments/v5/cat_inner_1.npz", va, cache["labels"][va])
    y = cache["labels"][va]
    baseline = metrics(y, .5*old+.5*cat)
    assert baseline["errors"] == 51
    before = (.5*old+.5*cat) >= baseline["threshold"]
    print(json.dumps({"event": "margin_screen_started", "baseline": baseline, "train_rows": len(tr)}), flush=True)
    fingerprint = semantic_hash(base)
    X = base._matrix(records_at(cache, tr))
    V = base._matrix(records_at(cache, va))
    np.testing.assert_array_equal(base.classifier_.predict_proba(V)[:, 1], old)
    print(json.dumps({"event": "margin_representations_ready", "shape": list(X.shape), "nnz": X.nnz}), flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CalibratedMarginText(**parameters).fit_representation(base, X, cache["labels"][tr])
    p = model.classifier_.predict_proba(V)[:, 1]
    expected = scope(cache, tr, {"parameters": parameters, "representation_origin": origin,
        "model_source_sha256": protocol["model_source_sha256"], "screen_source_sha256": protocol["source_sha256"],
        "protocol_sha256": sha(OUT / "preregistered.json")}, digest)
    save_component(OUT / "inner.joblib", model, expected)
    restored = load_component(OUT / "inner.joblib", expected)
    np.testing.assert_array_equal(p[:32], restored.predict_proba(records_at(cache, va[:32])))
    assert semantic_hash(base) == fingerprint
    rows = []
    for weight in protocol["blends"]:
        mixed = .5*((1-weight)*old+weight*p)+.5*cat
        candidate = metrics(y, mixed)
        pred = mixed >= candidate["threshold"]
        frozen = measure(y, mixed, baseline["threshold"])
        net_frozen = baseline["errors"]-frozen["fp"]-frozen["fn"]
        rows.append({"text_replacement": weight, "metrics": candidate,
            "metrics_at_baseline_threshold": frozen, "frozen_net": net_frozen,
            "fixed_errors": int(((before != y) & (pred == y)).sum()),
            "new_errors": int(((before == y) & (pred != y)).sum()),
            "gate_passed": bool(weight > 0 and candidate["errors"] <= 49 and net_frozen >= 2 and candidate["f1"] > baseline["f1"])})
    eligible = [r for r in rows if r["gate_passed"]]
    selected = max(eligible if eligible else rows, key=lambda r: r["metrics"]["f1"])
    report = {"gate_passed": bool(eligible), "baseline": baseline, "standalone": metrics(y, p),
        "selected": selected, "candidates": rows, "parameters": parameters, "protocol": protocol,
        "classifier_fits": 4, "warnings": [str(w.message) for w in caught], "serialization_rows": 32,
        "data_sha256": digest, "origin": origin, "seconds": time.perf_counter()-start,
        "external_test_labels_used": False}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2)+"\n")
    (OUT / "inner_scope.json").write_text(json.dumps(expected, indent=2)+"\n")
    np.savez_compressed(OUT / "inner_predictions.npz", train_indices=tr, indices=va, labels=y, probabilities=p)
    print(json.dumps({"event": "margin_screen_complete", **report}), flush=True)
    del X, V, model, restored, base
    gc.collect()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
