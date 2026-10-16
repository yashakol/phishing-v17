"""Numeric HTML + URL model; all parameter selection uses development only."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from phishing_url.content_validation import aligned_splits
from phishing_url.features import categorical_columns
from phishing_url.evaluation import measure, best_threshold


def load_frame(cache, split_name="train"):
    url = joblib.load(ROOT / f"data/content_url_{split_name}.joblib")
    if cache["urls"] != url["urls"]:
        raise ValueError("HTML/URL feature order differs")
    return pd.concat([url["features"].reset_index(drop=True),
                      cache["features"].reset_index(drop=True)], axis=1)


def fit(frame, labels, config, evaluation=None, seed=2026, threads=4):
    model = CatBoostClassifier(**config, loss_function="Logloss", random_seed=seed,
                               thread_count=threads, verbose=False, allow_writing_files=False)
    kwargs = {"cat_features": categorical_columns(frame)}
    if evaluation is not None:
        kwargs.update(eval_set=evaluation, early_stopping_rounds=120, use_best_model=True)
    model.fit(frame, labels, **kwargs)
    return model


def main():
    out = ROOT / "experiments/content"
    out.mkdir(parents=True, exist_ok=True)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    frame = load_frame(cache)
    y = cache["labels"]
    splits = aligned_splits(ROOT, cache)
    # Numeric fitting does not need the large text fields retained in memory.
    del cache["records"], cache["hashes"]
    np.savez_compressed(ROOT / "reports/v3/split_indices.npz", **splits)
    tr, va = splits["inner_train"], splits["inner_validation"]
    results = []
    candidates = [(7, .055, 5), (8, .045, 5)]
    for depth, learning_rate, regularization in candidates:
        config = dict(iterations=2200, depth=depth, learning_rate=learning_rate, l2_leaf_reg=regularization)
        print("Inner development tuning:", config, flush=True)
        started = time.perf_counter()
        model = fit(frame.iloc[tr], y[tr], config, (frame.iloc[va], y[va]))
        p = model.predict_proba(frame.iloc[va])[:, 1]
        threshold, _ = best_threshold(y[va], p)
        fitted_config = {**config, "iterations": int(model.tree_count_)}
        result = {"config": fitted_config, "threshold": threshold,
                  "metrics": measure(y[va], p, threshold), "seconds": time.perf_counter() - started}
        result["inner_predictions_file"] = f"inner_depth_{depth}.npz"
        np.savez_compressed(out / result["inner_predictions_file"], indices=va, labels=y[va], probabilities=p)
        results.append(result)
        print("Measured:", result, flush=True)
    selected = max(results, key=lambda row: row["metrics"]["f1"])
    (out / "selection.json").write_text(json.dumps({"selected": selected, "candidates": results,
                                                   "selected_on": "inner development validation only"}, indent=2))
    tr, va = splits["screening_train"], splits["screening_validation"]
    print("Frozen candidate screening:", selected["config"], flush=True)
    started = time.perf_counter()
    model = fit(frame.iloc[tr], y[tr], selected["config"])
    p = model.predict_proba(frame.iloc[va])[:, 1]
    metrics = measure(y[va], p, selected["threshold"])
    result = {"metrics": metrics, "config": selected["config"], "seconds": time.perf_counter() - started,
              "selection": "parameters and threshold frozen on inner validation; outer development screening only"}
    (out / "screen.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "screen_predictions.npz", indices=va, labels=y[va], probabilities=p)
    # Reuse predictions of the selected fitted inner model instead of refitting it.
    inner_p = np.load(out / selected["inner_predictions_file"])["probabilities"]
    np.savez_compressed(out / "inner_predictions.npz", indices=splits["inner_validation"],
                        labels=y[splits["inner_validation"]], probabilities=inner_p)
    joblib.dump(model, out / "screen_model.joblib", compress=3)
    print("HTML boost screening:", result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
