"""Development-only screening. Run from repo root with .venv/bin/python.

Uses the shared split/label integrity checks, then models only original
development records. Holdout and newly added rows never enter model selection.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits, threadpool_info

from phishing_url.evaluation import best_threshold, measure
from phishing_url.html_text import ContentTextModel
from phishing_url.content_validation import aligned_splits, records_at


def write_json(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def aligned_development(cache_path):
    cache = joblib.load(cache_path)
    try:
        splits = aligned_splits(ROOT, cache)
    except KeyError as error:
        raise ValueError("original URL missing from cache; no partial comparison") from error
    cache_indices = splits["development"]
    reverse_mapping = {int(index): old for old, index in enumerate(splits["old_to_cache"])}
    development = np.asarray([reverse_mapping[int(index)] for index in cache_indices])
    positions = {int(index): position for position, index in enumerate(cache_indices)}
    split_positions = {name: np.asarray([positions[int(i)] for i in splits[name]])
                       for name in ("screening_train", "screening_validation", "inner_train", "inner_validation")}
    labels = np.asarray(cache["labels"])[cache_indices]
    urls = np.asarray([cache["urls"][i] for i in cache_indices], dtype=str)
    records = records_at(cache, cache_indices)
    if any(record.get("url") != url for record, url in zip(records, urls)):
        raise ValueError("record URL order disagrees with cache URL order")
    source = cache.get("source", {})
    total = len(cache["urls"])
    del cache
    gc.collect()
    return records, labels, urls, development, cache_indices, source, total, split_positions


def fit_matrix(prototype, matrix, labels, config):
    model = copy.copy(prototype)
    model.c, model.nb, model.include_url = config["c"], config["nb"], config["include_url"]
    model.vectorizers_ = dict(prototype.vectorizers_)
    if not model.include_url:
        model.vectorizers_.pop("url", None)
    model.ratio_ = None
    if model.nb:
        pos = np.asarray(matrix[labels == 1].sum(axis=0)).ravel() + 1
        neg = np.asarray(matrix[labels == 0].sum(axis=0)).ravel() + 1
        model.ratio_ = np.log((pos / pos.sum()) / (neg / neg.sum())).astype(np.float32)
        matrix = matrix.multiply(model.ratio_).tocsr()
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.classifier_ = LogisticRegression(C=model.c, solver="liblinear",
                                               max_iter=model.max_iter,
                                               random_state=model.random_state).fit(matrix, labels)
    model.classes_ = model.classifier_.classes_
    model.n_features_ = model.n_features = matrix.shape[1]
    model.fit_seconds_ = model.fit_seconds = time.perf_counter() - started
    return model, [str(w.message) for w in caught]


def save_predictions(name, positions, p, labels, urls, original, cache_indices, threshold):
    payload = dict(clean_indices=original[positions], indices=cache_indices[positions],
                   cache_indices=cache_indices[positions], urls=urls[positions],
                   labels=labels[positions], probabilities=p, threshold=threshold)
    np.savez_compressed(OUT / f"{name}.npz", **payload)
    alias = {"tune_predictions": "inner_predictions", "validation_predictions": "screen_predictions"}.get(name)
    if alias:
        np.savez_compressed(OUT / f"{alias}.npz", **payload)
    with open(OUT / f"{name}.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clean_index", "cache_index", "url", "label", "probability"])
        writer.writerows(zip(original[positions], cache_indices[positions], urls[positions],
                             labels[positions], p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/content_train.joblib")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while not args.cache.exists():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(args.cache)
        print(json.dumps({"event": "waiting_for_cache", "path": str(args.cache)}), flush=True)
        time.sleep(30)
    # The extractor writes joblib directly, so existence alone is not readiness.
    previous = None
    stable = 0
    while stable < 3:
        state = (args.cache.stat().st_size, args.cache.stat().st_mtime_ns)
        stable = stable + 1 if state == previous and state[0] > 0 else 0
        previous = state
        if time.monotonic() >= deadline:
            raise TimeoutError("cache did not finish writing")
        time.sleep(5)
    started = time.perf_counter()
    records, labels, urls, original, cache_indices, source, total, splits = aligned_development(args.cache)
    outer, validation = splits["screening_train"], splits["screening_validation"]
    train, tune = splits["inner_train"], splits["inner_validation"]
    np.savez_compressed(OUT / "screen_split.npz", development_indices=original,
                        train_indices=original[train], tune_indices=original[tune],
                        validation_indices=original[validation], outer_train_indices=original[outer],
                        development_cache_indices=cache_indices,
                        inner_train=cache_indices[train], inner_validation=cache_indices[tune],
                        screening_train=cache_indices[outer], screening_validation=cache_indices[validation])
    alignment = {"scope": "original development only; new archive rows excluded; holdout not evaluated",
                 "development_rows": len(labels), "cache_rows": total,
                 "train_rows": len(train), "tune_rows": len(tune), "validation_rows": len(validation),
                 "source": source, "cache_bytes": args.cache.stat().st_size,
                 "clean_train_sha256": digest(ROOT / "reports/clean_train.csv"),
                 "split_sha256": digest(ROOT / "reports/split_indices.npz"),
                 "seeds": [20261006, 18428]}
    write_json("alignment.json", alignment)
    print(json.dumps({"event": "aligned", **alignment}, default=str), flush=True)
    take = lambda indices: [records[i] for i in indices]
    prototype = ContentTextModel(verbose=True)
    feature_started = time.perf_counter()
    train_x = prototype._matrix(take(train), fit=True)
    tune_x = prototype._matrix(take(tune))
    feature_seconds = time.perf_counter() - feature_started
    dimensions = {name: len(v.vocabulary_) for name, v in prototype.vectorizers_.items()}
    print(json.dumps({"event": "features_ready", "seconds": feature_seconds,
                      "dimensions": dimensions, "nnz": train_x.nnz}), flush=True)
    content_columns = train_x.shape[1] - dimensions.get("url", 0)
    configurations = [dict(name="content_url_c4", c=4., nb=False, include_url=True),
                      dict(name="content_url_c16", c=16., nb=False, include_url=True),
                      dict(name="content_url_nb4", c=4., nb=True, include_url=True),
                      dict(name="content_only_c4", c=4., nb=False, include_url=False)]
    rows, predictions = [], {}
    best_model = None
    for config in configurations:
        x, tx = ((train_x, tune_x) if config["include_url"] else
                 (train_x[:, :content_columns], tune_x[:, :content_columns]))
        model, warning_messages = fit_matrix(prototype, x, labels[train], config)
        if model.ratio_ is not None:
            tx = tx.multiply(model.ratio_).tocsr()
        p = model.classifier_.predict_proba(tx)[:, 1]
        threshold, f1 = best_threshold(labels[tune], p)
        row = {"config": config, "threshold": threshold, "tune_f1": f1,
               "tune_metrics": measure(labels[tune], p, threshold),
               "fit_seconds": model.fit_seconds_, "warnings": warning_messages,
               "iterations": model.classifier_.n_iter_.tolist()}
        if not rows or f1 > max(r["tune_f1"] for r in rows):
            best_model = model
        rows.append(row)
        predictions[config["name"]] = p
        write_json("tuning.json", rows)
        print(json.dumps({"event": "candidate", **row}), flush=True)
        del x, tx
    selected = max(rows, key=lambda row: row["tune_f1"])
    parameters = best_model.get_params()
    selection = {"config": selected["config"], "model_parameters": parameters,
                 "threshold": selected["threshold"], "selection_metric": "inner tune F1",
                 "validation_protocol": "refit fixed parameters on outer train; frozen inner threshold",
                 "full_3fold_oof": "not run; requires explicit cost review"}
    write_json("selection.json", selection)  # Frozen before validation records are transformed.
    np.savez_compressed(OUT / "all_tune_predictions.npz", clean_indices=original[tune], cache_indices=cache_indices[tune],
                        urls=urls[tune], labels=labels[tune], **predictions)
    save_predictions("tune_predictions", tune, predictions[selected["config"]["name"]],
                     labels, urls, original, cache_indices, selection["threshold"])
    joblib.dump(best_model, OUT / "inner_selected.joblib", compress=3)
    del train_x, tune_x, prototype, best_model, model
    gc.collect()
    model = ContentTextModel(**parameters).fit(take(outer), labels[outer])
    predict_started = time.perf_counter()
    p = model.predict_proba(take(validation))
    predict_seconds = time.perf_counter() - predict_started
    metrics = measure(labels[validation], p, selection["threshold"])
    save_predictions("validation_predictions", validation, p, labels, urls, original,
                     cache_indices, selection["threshold"])
    path = OUT / "screen_selected.joblib"
    joblib.dump(model, path, compress=3)
    restored = joblib.load(path)
    np.testing.assert_array_equal(p[:32], restored.predict_proba(take(validation[:32])))
    result = {"alignment": alignment, "selection": selection, "tuning": rows,
              "validation": metrics, "validation_at_05": measure(labels[validation], p),
              "feature_seconds": feature_seconds, "dimensions": dimensions,
              "outer_refit_seconds": model.fit_seconds_, "validation_predict_seconds": predict_seconds,
              "serialized_bytes": path.stat().st_size, "serialization_verified_rows": min(32, len(p)),
              "total_seconds": time.perf_counter() - started,
              "environment": {name: importlib.metadata.version(name)
                              for name in ("numpy", "scipy", "scikit-learn", "joblib")},
              "threadpools": threadpool_info(),
              "model_source_sha256": digest(Path(sys.modules[ContentTextModel.__module__].__file__)),
              "screen_script_sha256": digest(Path(__file__)),
              "cost_note": "3-fold OOF not run. Measured outer refit includes TF-IDF fitting."}
    write_json("results.json", result)
    print(json.dumps({"event": "finished", **result}, default=str), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
