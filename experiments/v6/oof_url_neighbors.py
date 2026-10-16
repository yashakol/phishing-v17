"""Quick training-only URL neighbor OOF, including confident learned predictions."""

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

from experiments.v6.screen_extra import sha
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.neighbors import NeighborModel, adjust_neighbors


def main():
    cache = joblib.load(ROOT / "data/content_train.joblib")
    dev = aligned_splits(ROOT, cache)["development"]
    old = np.load(ROOT / "reports/v5/development_oof.npz")
    np.testing.assert_array_equal(old["indices"], dev)
    y = cache["labels"][dev]
    np.testing.assert_array_equal(old["labels"], y)
    config = dict(view="url", max_features=180000, max_neighbors=20, threads=2)
    scoped = {**config, "source_sha256": sha(ROOT / "phishing_url/neighbors.py")}
    digest = sha(ROOT / "data/content_train.joblib")
    output = ROOT / "experiments/v6/url_neighbors_oof"
    output.mkdir(exist_ok=True)
    similarities = np.zeros((len(dev), 20), dtype=np.float32)
    labels = np.full((len(dev), 20), -1, dtype=np.int8)
    prior = np.zeros(len(dev))
    folds = np.zeros(len(dev), dtype=np.int8)
    for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(dev, y)):
        started = time.perf_counter()
        print(json.dumps({"event": "url_neighbors_oof_started", "fold": fold}), flush=True)
        expected = guard(cache, dev[tr], scoped, digest)
        path = output / f"fold_{fold}.joblib"
        if path.exists():
            model = load_component(path, expected)
        else:
            model = NeighborModel(**config).fit(records_at(cache, dev[tr]), y[tr])
            save_component(path, model, expected)
        similarities[va], labels[va] = model.lookup(records_at(cache, dev[va]))
        prior[va], folds[va] = model.prior_, fold
        np.savez_compressed(output / f"fold_{fold}.npz", indices=dev[va], labels=y[va],
                            similarities=similarities[va], neighbor_labels=labels[va],
                            prior=prior[va], scope_json=json.dumps(expected, sort_keys=True))
        print(json.dumps({"event": "url_neighbors_oof_complete", "fold": fold,
                          "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()
    np.savez_compressed(ROOT / "experiments/v6/url_neighbors_development_oof.npz",
                        indices=dev, labels=y, similarities=similarities,
                        neighbor_labels=labels, prior=prior, folds=folds)
    rows = []
    for minimum in (.7, .85, .95):
        for weight in (.15, .35, .6):
            for support in (1, 2, 3):
                adjustment = dict(minimum_similarity=minimum, weight=weight, minimum_support=support,
                                  neighbors=10, power=4, smoothing=.5)
                p = adjust_neighbors(old["probabilities"], similarities, labels, {**adjustment, "prior": prior})
                cut, _ = best_threshold(y, p)
                pred = p >= cut
                oldpred = old["probabilities"] >= .47131719980380204
                rows.append({"adjustment": adjustment, "threshold": cut, "metrics": measure(y, p, cut),
                             "fixed_v5_errors": int(np.sum((oldpred != y) & (pred == y))),
                             "new_errors": int(np.sum((oldpred == y) & (pred != y)))})
    selected = max(rows, key=lambda row: row["metrics"]["f1"])
    report = {"selected": selected, "candidates": rows, "model_parameters": config,
              "train_cache_sha256": digest, "external_test_labels_used": False,
              "reason_for_reopening": "URL variant is inexpensive to validate on full OOF, and some residual errors are very confident learned answers on domains with similar labeled URLs."}
    (ROOT / "experiments/v6/url_neighbor_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "url_neighbor_selection", "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
