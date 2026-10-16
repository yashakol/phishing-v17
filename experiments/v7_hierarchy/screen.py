"""Select URL conditional evidence on inner development, then test original OOF."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from phishing_url.content_checkpoints import guard, save_component
from phishing_url.content_memory import fit_memory, apply_memory
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v7_hierarchy import HierarchicalEvidence, hierarchy_keys
from experiments.v6.screen_extra import sha


FAMILIES = ["host_path1", "host_path2", "domain_path1", "domain_path2", "host_query"]
POLICIES = [
    dict(families=FAMILIES, minimum_support=3, alpha=5., weight=.6, confidence_limit=.98),
    dict(families=FAMILIES + ["host"], minimum_support=5, alpha=5., weight=.6, confidence_limit=.98),
    dict(families=FAMILIES + ["global_path"], minimum_support=3, alpha=5., weight=.6, confidence_limit=.98),
]


def main():
    started = time.perf_counter()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    keys = [hierarchy_keys(url) for url in cache["urls"]]
    text = np.load(ROOT / "experiments/html_astra/inner_predictions.npz")
    tree = np.load(ROOT / "experiments/v5/cat_inner_1.npz")
    for values in (text, tree):
        np.testing.assert_array_equal(values["indices"], va)
        np.testing.assert_array_equal(values["labels"], cache["labels"][va])
    p = .5 * text["probabilities"] + .5 * tree["probabilities"]
    memory = fit_memory(records_at(cache, tr), [cache["hashes"][int(i)] for i in tr], cache["labels"][tr], 3, 5)
    base, _ = apply_memory(memory, records_at(cache, va), [cache["hashes"][int(i)] for i in va], p)
    y = cache["labels"][va]
    cut, _ = best_threshold(y, base)
    baseline = measure(y, base, cut)
    rows = []
    digest = sha(ROOT / "data/content_train.joblib")
    for number, config in enumerate(POLICIES):
        model = HierarchicalEvidence(**config).fit_keys([keys[int(i)] for i in tr], cache["labels"][tr])
        p, hits = model.predict_keys([keys[int(i)] for i in va], base)
        threshold, _ = best_threshold(y, p)
        row = {"number": number, "config": config, "metrics": measure(y, p, threshold), "matches": int(hits.sum()),
               "metrics_at_baseline_threshold": measure(y, p, cut),
               "fixed_errors": int(np.sum((base >= cut) != y) - np.sum((p >= threshold) != y))}
        rows.append(row)
        np.savez_compressed(OUT / f"inner_{number}.npz", indices=va, labels=y, probabilities=p)
        print(json.dumps({"event": "hierarchy_inner_complete", **row}), flush=True)
    chosen = max(rows, key=lambda row: row["metrics"]["f1"])
    useful = chosen["metrics"]["f1"] > baseline["f1"] + 1e-12
    selection = {"baseline": baseline, "selected": chosen, "candidates": rows, "useful": useful,
                 "protocol": "Selected on original inner development only before opening OOF predictions.",
                 "source_sha256": sha(ROOT / "phishing_url/v7_hierarchy.py"), "data_sha256": digest}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    if useful:
        oof = np.load(ROOT / "reports/v6/development_oof.npz")
        dev = splits["development"]
        np.testing.assert_array_equal(oof["indices"], dev)
        np.testing.assert_array_equal(oof["labels"], cache["labels"][dev])
        base = oof["probabilities"]
        y = cache["labels"][dev]
        p = np.empty(len(dev))
        folds = np.empty(len(dev), dtype=np.int8)
        hits = np.zeros(len(dev), dtype=bool)
        for fold, (training, validation) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(dev, y)):
            model = HierarchicalEvidence(**chosen["config"]).fit_keys([keys[int(i)] for i in dev[training]], y[training])
            p[validation], hits[validation] = model.predict_keys([keys[int(i)] for i in dev[validation]], base[validation])
            folds[validation] = fold
            save_component(OUT / f"fold_{fold}.joblib", model,
                           guard(cache, dev[training], {**chosen["config"], "source_sha256": selection["source_sha256"]}, digest))
        baseline_threshold = json.loads((ROOT / "reports/v6/metrics.json").read_text())["selection"]["threshold"]
        threshold, _ = best_threshold(y, p)
        result = {"selected": chosen, "baseline": measure(y, base, baseline_threshold), "oof": measure(y, p, threshold),
                  "at_frozen_baseline_threshold": measure(y, p, baseline_threshold), "matches": int(hits.sum()),
                  "fixed_errors": int(np.sum(((base >= baseline_threshold) != y) & ((p >= threshold) == y))),
                  "new_errors": int(np.sum(((base >= baseline_threshold) == y) & ((p >= threshold) != y))),
                  "folds": [{"fold": f, "baseline": measure(y[folds==f], base[folds==f], baseline_threshold),
                             "candidate": measure(y[folds==f], p[folds==f], threshold)} for f in range(3)],
                  "protocol": "Fold-local training counts. Policy selected inner; threshold optimized on development OOF, no holdout/test labels.",
                  "limitation": "Development OOF participates in model/threshold selection; not a fresh independent score.",
                  "external_test_labels_used": False}
        np.savez_compressed(OUT / "development_oof.npz", indices=dev, labels=y, probabilities=p, base=base, folds=folds, matches=hits)
        (OUT / "oof_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": "hierarchy_oof_complete", **result}), flush=True)
    print(json.dumps({"event": "hierarchy_complete", "useful": useful, "baseline": baseline, "selected": chosen,
                      "seconds": time.perf_counter()-started}), flush=True)


if __name__ == "__main__":
    main()

