"""Resumable original-development OOF with frozen screening parameters.

No holdout, new archive rows, external reference data, or test labels are used.
Run after screening, or leave --wait-seconds at its default to wait for it.
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
import gc
import hashlib
import importlib.metadata
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info

from phishing_url.html_text import ContentTextModel
from phishing_url.evaluation import measure
from experiments.html_astra.screen import aligned_development, digest


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def validated_checkpoint(directory, fold, guard_digest, expected_train, expected_valid, expected_labels):
    meta_path = directory / f"fold_{fold}.json"
    path = directory / f"fold_{fold}.npz"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    if metadata["guard_digest"] != guard_digest or metadata["fold"] != fold:
        raise ValueError("OOF checkpoint config/source guard mismatch")
    if not path.exists() or digest(path) != metadata["predictions_sha256"]:
        raise ValueError("OOF checkpoint prediction file missing or changed")
    with np.load(path, allow_pickle=False) as checkpoint:
        if (not np.array_equal(checkpoint["indices"], expected_valid) or
                not np.array_equal(checkpoint["train_indices"], expected_train) or
                not np.array_equal(checkpoint["labels"], expected_labels)):
            raise ValueError("OOF checkpoint index/label alignment mismatch")
        p = checkpoint["probabilities"].copy()
    if p.shape != expected_labels.shape or not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("OOF checkpoint invalid probabilities")
    return p, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/content_train.joblib")
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while not (OUT / "results.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("screening must finish before OOF starts")
        print(json.dumps({"event": "waiting_for_screening"}), flush=True)
        time.sleep(30)
    selection = json.loads((OUT / "selection.json").read_text())
    screen = json.loads((OUT / "results.json").read_text())
    if screen["selection"] != selection:
        raise ValueError("screening result does not match frozen selection")
    model_source_sha256 = digest(Path(sys.modules[ContentTextModel.__module__].__file__))
    if model_source_sha256 != screen["model_source_sha256"]:
        raise ValueError("model source changed since completed screening")
    started = time.perf_counter()
    records, labels, urls, original, indices, source, _, _ = aligned_development(args.cache)
    parameters = selection["model_parameters"]
    environment = {name: importlib.metadata.version(name)
                   for name in ("numpy", "scipy", "scikit-learn", "joblib")}
    guard = dict(format_version=1, model_parameters=parameters,
                 selection_sha256=digest(OUT / "selection.json"),
                 cache_sha256=digest(args.cache), source=source,
                 clean_train_sha256=digest(ROOT / "reports/clean_train.csv"),
                 split_sha256=digest(ROOT / "reports/split_indices.npz"),
                 model_source_sha256=model_source_sha256, script_sha256=digest(Path(__file__)),
                 cache_indices_sha256=canonical_hash(indices.tolist()),
                 labels_sha256=canonical_hash(labels.tolist()),
                 splitter=dict(n_splits=3, shuffle=True, random_state=2026), environment=environment)
    guard_digest = canonical_hash(guard)
    directory = OUT / "oof"
    directory.mkdir(exist_ok=True)
    guard_path = directory / "run_guard.json"
    if guard_path.exists() and json.loads(guard_path.read_text()) != guard:
        raise ValueError("existing OOF checkpoints use a different config/source/index guard")
    atomic_json(guard_path, guard)
    p = np.full(len(labels), np.nan)
    folds = np.full(len(labels), -1, dtype=np.int8)
    summaries = []
    take = lambda positions: [records[int(i)] for i in positions]
    split = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    for fold, (train, valid) in enumerate(split.split(np.arange(len(labels)), labels)):
        restored = validated_checkpoint(directory, fold, guard_digest, indices[train], indices[valid], labels[valid])
        if restored is not None:
            p[valid], metadata = restored
            folds[valid] = fold
            summaries.append(metadata)
            print(json.dumps({"event": "fold_restored", "fold": fold, "rows": len(valid)}), flush=True)
            continue
        print(json.dumps({"event": "fold_started", "fold": fold, "train_rows": len(train),
                          "validation_rows": len(valid), "threads": 2}), flush=True)
        fold_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = ContentTextModel(**parameters).fit(take(train), labels[train])
        prediction_started = time.perf_counter()
        fold_p = model.predict_proba(take(valid))
        prediction_seconds = time.perf_counter() - prediction_started
        if not (np.isfinite(fold_p).all() and ((fold_p >= 0) & (fold_p <= 1)).all()):
            raise ValueError("non-finite or out-of-range fold probabilities")
        model_path = directory / f"fold_{fold}.joblib"
        temporary_model = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, temporary_model, compress=3)
        temporary_model.replace(model_path)
        path = directory / f"fold_{fold}.npz"
        atomic_npz(path, indices=indices[valid], cache_indices=indices[valid],
                   clean_indices=original[valid], urls=urls[valid], labels=labels[valid],
                   probabilities=fold_p, train_indices=indices[train], fold=fold)
        metadata = dict(fold=fold, guard_digest=guard_digest,
                        predictions_sha256=digest(path), train_rows=len(train), validation_rows=len(valid),
                        fit_seconds=model.fit_seconds_, view_fit_seconds=model.view_fit_seconds_,
                        predict_seconds=prediction_seconds, total_seconds=time.perf_counter() - fold_started,
                        features=model.n_features_, iterations=model.classifier_.n_iter_.tolist(),
                        warnings=[str(w.message) for w in caught], model_bytes=model_path.stat().st_size,
                        metrics_frozen_screen_threshold=measure(labels[valid], fold_p, selection["threshold"]))
        atomic_json(directory / f"fold_{fold}.json", metadata)
        p[valid], folds[valid] = fold_p, fold
        summaries.append(metadata)
        print(json.dumps({"event": "fold_finished", **metadata}), flush=True)
        del model, fold_p
        gc.collect()
    if not np.isfinite(p).all() or (folds < 0).any():
        raise AssertionError("OOF must cover every development row exactly once")
    atomic_npz(OUT / "development_oof.npz", indices=indices, cache_indices=indices,
               clean_indices=original, urls=urls, labels=labels, y=labels, probabilities=p, folds=folds)
    report = dict(scope="original development only; no holdout/new rows/test labels", rows=len(labels),
                  guard_digest=guard_digest, splitter=guard["splitter"], selected_parameters=parameters,
                  folds=summaries, metrics_frozen_screen_threshold=measure(labels, p, selection["threshold"]),
                  metrics_at_05=measure(labels, p), total_seconds_this_run=time.perf_counter() - started,
                  measured_fold_seconds=sum(item["total_seconds"] for item in summaries),
                  threadpools=threadpool_info(), environment=environment,
                  output_sha256=digest(OUT / "development_oof.npz"))
    atomic_json(OUT / "oof_results.json", report)
    print(json.dumps({"event": "oof_finished", **report}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
