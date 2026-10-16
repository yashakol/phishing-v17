"""Validate the fixed resource representation on the original three folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.v8_resources.screen import CONFIG, scope
from experiments.v6.screen_extra import sha
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v8_bundle import mix_resources
from phishing_url.v8_resource_text import ResourceCharacterTextModel


def main():
    out = ROOT / "experiments/v8_resources"
    selected = json.loads((out / "selection.json").read_text())
    assert selected["useful"], "Resource representation did not pass its frozen inner gate"
    assert selected["model_parameters"] == CONFIG
    assert selected["guard"]["config"] == scope()
    cache = joblib.load(ROOT / "data/extra_html_train.joblib")
    dev = aligned_splits(ROOT, cache)["development"]
    y = cache["labels"][dev]
    with np.load(ROOT / "reports/v7/development_oof.npz") as baseline:
        np.testing.assert_array_equal(baseline["indices"], dev)
        np.testing.assert_array_equal(baseline["labels"], y)
        base = baseline["probabilities"].copy()
    config = scope()
    digest = sha(ROOT / "data/content_train.joblib")
    p = np.empty(len(dev))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(dev, y))
    fold_ids = np.full(len(dev), -1, dtype=np.int8)
    counts = np.zeros(len(dev), dtype=np.int8)
    checks = []
    for fold, (tr, va) in enumerate(folds):
        started = time.perf_counter()
        expected = guard(cache, dev[tr], config, digest)
        path = out / f"fold_{fold}.joblib"
        if path.exists():
            model = load_component(path, expected)
        else:
            print(json.dumps({"event": "v8_resource_fold_fit", "fold": fold, "training_rows": len(tr)}), flush=True)
            model = ResourceCharacterTextModel(**CONFIG).fit(records_at(cache, dev[tr]), y[tr])
            assert model.vocabulary_fit_rows_ == len(tr)
            save_component(path, model, expected)
        records = records_at(cache, dev[va])
        p[va] = model.predict_proba(records)
        restored = load_component(path, expected)
        np.testing.assert_array_equal(p[va[:32]], restored.predict_proba(records[:32]))
        np.savez_compressed(out / f"fold_{fold}.npz", train_indices=dev[tr], indices=dev[va], labels=y[va],
                            probabilities=p[va], scope_json=json.dumps(expected, sort_keys=True))
        check = {"fold": fold, "training_rows": len(tr), "validation_rows": len(va),
                 "training_guard": expected, "model_sha256": sha(path), "serialization_rows": 32,
                 "seconds": time.perf_counter()-started}
        (out / f"fold_{fold}.json").write_text(json.dumps(check, indent=2) + "\n")
        checks.append(check)
        fold_ids[va] = fold
        counts[va] += 1
        print(json.dumps({"event": "v8_resource_fold_complete", "fold": fold, "seconds": check["seconds"]}), flush=True)
        del model, restored, records
        gc.collect()
    np.testing.assert_array_equal(counts, np.ones(len(dev), dtype=np.int8))
    v7 = json.loads((ROOT / "reports/v7/selection.json").read_text())["selected"]
    before = base >= v7["threshold"]
    rows, predictions = [], {}
    for weight in (0., .1, .25):
        combined = mix_resources(base, p, weight)
        threshold, _ = best_threshold(y, combined)
        after = combined >= threshold
        row = {"resource_weight": weight, "threshold": threshold, "metrics": measure(y, combined, threshold),
               "fixed_errors": int(np.sum((before != y) & (after == y))),
               "new_errors": int(np.sum((before == y) & (after != y))),
               "fold_comparison": [{"fold": f, "baseline_errors": int(np.sum(before[va] != y[va])),
                   "candidate_errors": int(np.sum(after[va] != y[va]))} for f, (_, va) in enumerate(folds)]}
        rows.append(row)
        predictions[weight] = combined
    chosen = max(rows, key=lambda row: (row["metrics"]["f1"], -row["resource_weight"]))
    result = {"selected": chosen, "baseline_v7": v7["metrics"], "candidates": rows, "model_parameters": CONFIG,
              "useful": chosen["metrics"]["f1"] > v7["metrics"]["f1"] and chosen["fixed_errors"] > chosen["new_errors"],
              "training_checks": checks, "train_cache_sha256": digest, "scope": config,
              "external_test_labels_used": False,
              "limitation": "OOF weight/threshold selection and repeated inner screening are optimistic; no holdout/test read."}
    np.savez_compressed(out / "development_oof.npz", indices=dev, labels=y, probabilities=p, folds=fold_ids,
                        mixture=predictions[chosen["resource_weight"]], baseline=base)
    (out / "oof_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "v8_resource_oof_selected", "selected": chosen, "useful": result["useful"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
