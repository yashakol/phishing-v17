"""Compare page-neighbor representations on the prior inner development split."""

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import data_digest, guard, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.neighbors import NeighborModel, adjust_neighbors


def main():
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    digest = data_digest(ROOT)
    table = np.load(ROOT / "experiments/html_astra/inner_predictions.npz")
    np.testing.assert_array_equal(table["indices"], va)
    text = table["probabilities"]
    # Frozen numeric inner model, already fitted only on inner_train.
    tree = np.load(ROOT / "experiments/v5/cat_inner_1.npz")
    np.testing.assert_array_equal(tree["indices"], va)
    base = .5 * text + .5 * tree["probabilities"]
    y = cache["labels"][va]
    cut, _ = best_threshold(y, base)
    baseline = measure(y, base, cut)
    out = ROOT / "experiments/v6"
    results = []
    for view in ("content", "markup", "combined", "url"):
        output = out / f"neighbors_inner_{view}.npz"
        config = dict(view=view, max_features=180000, max_neighbors=20, threads=2)
        if output.exists():
            saved = np.load(output)
            np.testing.assert_array_equal(saved["indices"], va)
            similarities, labels = saved["similarities"], saved["neighbor_labels"]
        else:
            started = time.perf_counter()
            print(json.dumps({"event": "neighbor_inner_started", "view": view}), flush=True)
            model = NeighborModel(**config).fit(records_at(cache, tr), cache["labels"][tr])
            similarities, labels = model.lookup(records_at(cache, va))
            save_component(out / f"neighbors_inner_{view}.joblib", model, guard(cache, tr, config, digest))
            np.savez_compressed(output, indices=va, labels=y, similarities=similarities,
                                neighbor_labels=labels, base=base)
            print(json.dumps({"event": "neighbor_inner_ready", "view": view,
                              "seconds": time.perf_counter() - started}), flush=True)
            del model
            gc.collect()
        for minimum in (.7, .85, .95):
            for weight in (.15, .35, .6):
                for support in (1, 2, 3):
                    adjustment = dict(minimum_similarity=minimum, weight=weight, minimum_support=support,
                                      neighbors=10, power=4, smoothing=.5,
                                      prior=float(cache["labels"][tr].mean()))
                    p = adjust_neighbors(base, similarities, labels, adjustment)
                    cut, _ = best_threshold(y, p)
                    results.append({"view": view, "model_parameters": config,
                                    "adjustment": adjustment, "threshold": cut,
                                    "metrics": measure(y, p, cut)})
        best = max((row for row in results if row["view"] == view), key=lambda row: row["metrics"]["f1"])
        print(json.dumps({"event": "neighbor_inner_scored", **best}), flush=True)
    selected = max(results, key=lambda row: row["metrics"]["f1"])
    report = {"selected": selected, "baseline": baseline, "candidates": results,
              "train_cache_sha256": digest,
              "scope": "Original inner development only; no holdout/test labels or added rows."}
    (out / "neighbor_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "neighbor_screen_complete", "baseline": baseline, "selected": selected}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
