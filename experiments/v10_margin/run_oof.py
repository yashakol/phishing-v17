"""One frozen calibrated-SVM follow-up, three original development folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "2"

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info

from experiments.v8_astra_regularization.resume import checked_base, scope, semantic_hash, sha
from phishing_url.content_checkpoints import data_digest, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.v9_margin import CalibratedMarginText

READS = set()


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline development experiment")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT):
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(".venv/"):
        return
    mode, flags = args[1:3]
    writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
               (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
    if writing and not path.is_relative_to(OUT):
        raise PermissionError(f"Write outside experiment: {rel}")
    if (rel.startswith(("external/", "reports/v2/", "submissions/")) or
            "reference" in path.name.lower() or "holdout" in rel.lower() or
            "test" in path.name.lower()):
        raise PermissionError(f"Forbidden experiment input: {rel}")
    if not writing:
        READS.add(rel)


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def main():
    sys.addaudithook(audit)
    assert not (OUT / "preregistered.json").exists(), "No silent rerun or refit"
    prior = json.loads((ROOT / "experiments/v9_margin/selection.json").read_text())
    assert prior["gate_passed"] is False
    assert prior["standalone"]["errors"] == 61 and prior["baseline"]["errors"] == 51
    parameters = CalibratedMarginText().get_params()
    protocol = {
        "reason": "Explore full existing ensemble after standalone inner SVM improved 68 to 61 errors; original half-mixture gate failed and remains false",
        "parameters": parameters, "new_classifier_fits": 12,
        "previous_inner_classifier_fits": 4, "total_margin_branch_classifier_fits": 16,
        "outer_splitter": {"n_splits": 3, "shuffle": True, "random_state": 2026},
        "development_rows": 50977, "calibration": "Three internal train-only folds and one complete outer-train SVM fit per outer fold",
        "representation": "Exact guarded original C16 fold vocabulary/IDF; no vocabulary fitting or hyperparameter changes",
        "integration_weights": [0.25, 0.5, 1.0],
        "integration_fixed": "Original v9 weak-HTML routing, v7 remaining mixture, train-only memory, soft context, extra HTML and URL hierarchy",
        "integration_gate": "At least 5 fewer development errors at both original frozen and tuned F1 thresholds, higher F1, at least 2 improved folds and no fold worsens by more than 2",
        "selection_scope": "Reused original development only; no holdout, added rows or test",
        "limitation": "Exploratory model selection is not an independent performance estimate. Internal calibration uses outer-train vocabulary/IDF, including internal calibration rows; outer validation remains excluded",
        "model_source_sha256": sha(ROOT / "phishing_url/v9_margin.py"),
        "run_source_sha256": sha(Path(__file__)),
        "original_inner_selection_sha256": sha(ROOT / "experiments/v9_margin/selection.json"),
        "public_feedback": "User reports another 0.995080; association to v9 pending clarification; no row labels received",
    }
    write("preregistered.json", protocol)
    protocol_sha = sha(OUT / "preregistered.json")
    digest = data_digest(ROOT)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    dev = splits["development"]
    y = cache["labels"][dev]
    assert len(dev) == 50977 and len(np.unique(dev)) == len(dev)
    assert not set(dev) & (set(splits["holdout"]) | set(splits["added"]))
    fold_indices = list(StratifiedKFold(**protocol["outer_splitter"]).split(dev, y))
    origins = []
    for fold, (ti, vi) in enumerate(fold_indices):
        tr, va = dev[ti], dev[vi]
        assert not set(tr) & set(va) and set(tr) | set(va) == set(dev)
        gp = ROOT / f"experiments/v7_astra/guards/fold_{fold}.json"
        mp = ROOT / f"experiments/v7_astra/models/fold_{fold}.joblib"
        g = json.loads(gp.read_text())
        assert g == scope(cache, tr, g["config"], digest)
        for path, value in g["config"]["sources"].items():
            assert sha(ROOT / path) == value, path
        origins.append({"fold": fold, "train_indices_sha256": array_sha(tr),
                        "validation_indices_sha256": array_sha(va),
                        "model_sha256": sha(mp), "guard_sha256": sha(gp)})
    write("fold_scopes_before_fitting.json", {"origins": origins, "data_sha256": digest,
                                           "development_indices_sha256": array_sha(dev)})
    p = np.full(len(dev), np.nan)
    folds = np.full(len(dev), -1, dtype=np.int8)
    coverage = np.zeros(len(dev), dtype=np.int8)
    start = time.perf_counter()
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as old:
        np.testing.assert_array_equal(old["indices"], dev)
        np.testing.assert_array_equal(old["labels"], y)
        old_p = old["probabilities"].copy()
    for fold, (ti, vi) in enumerate(fold_indices):
        started_fold = time.perf_counter()
        tr, va = dev[ti], dev[vi]
        base, origin = checked_base(cache, tr, f"fold_{fold}", digest)
        assert origin["model_sha256"] == origins[fold]["model_sha256"]
        assert origin["guard_sha256"] == origins[fold]["guard_sha256"]
        fingerprint = semantic_hash(base)
        emit("margin_fold_transform_started", fold=fold, train_rows=len(tr), validation_rows=len(va))
        X = base._matrix(records_at(cache, tr))
        V = base._matrix(records_at(cache, va))
        np.testing.assert_array_equal(base.classifier_.predict_proba(V)[:, 1], old_p[vi])
        emit("margin_fold_fit_started", fold=fold, shape=list(X.shape), nnz=int(X.nnz))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = CalibratedMarginText(**parameters).fit_representation(base, X, cache["labels"][tr])
        values = model.classifier_.predict_proba(V)[:, 1]
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        expected = scope(cache, tr, {"parameters": parameters, "representation_origin": origin,
            "model_source_sha256": protocol["model_source_sha256"],
            "run_source_sha256": protocol["run_source_sha256"], "protocol_sha256": protocol_sha}, digest)
        target = OUT / f"models/fold_{fold}.joblib"
        save_component(target, model, expected)
        restored = load_component(target, expected)
        np.testing.assert_array_equal(values[:32], restored.predict_proba(records_at(cache, va[:32])))
        try:
            load_component(target, dict(expected, training_rows=-1))
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong training guard accepted")
        assert clone(model).get_params() == parameters
        assert semantic_hash(base) == fingerprint
        np.testing.assert_array_equal(base.predict_proba(records_at(cache, va[:32])), old_p[vi[:32]])
        assert sha(ROOT / origin["model"]) == origin["model_sha256"]
        assert sha(ROOT / origin["guard"]) == origin["guard_sha256"]
        write(f"guards/fold_{fold}.json", expected)
        write(f"fold_{fold}_execution.json", {"fold": fold, "classifier_fits": model.classifier_fits_,
            "train_rows": len(tr), "validation_rows": len(va), "features": model.n_features_,
            "model_sha256": sha(target), "training_nnz": int(X.nnz), "validation_nnz": int(V.nnz),
            "warnings": [str(item.message) for item in caught], "serialization_replay_rows": 32,
            "base_model_unchanged": True, "wrong_guard_rejected": True,
            "seconds": time.perf_counter() - started_fold})
        np.savez_compressed(OUT / f"fold_{fold}.npz", train_indices=tr, indices=va,
                            labels=cache["labels"][va], probabilities=values)
        p[vi], folds[vi] = values, fold
        coverage[vi] += 1
        emit("margin_fold_complete", fold=fold, seconds=time.perf_counter()-started_fold)
        del X, V, model, restored, base
        gc.collect()
    assert np.all(coverage == 1) and np.isfinite(p).all() and np.all(folds >= 0)
    np.savez_compressed(OUT / "development_oof.npz", indices=dev, labels=y, probabilities=p, folds=folds)
    assert data_digest(ROOT) == digest
    assert sha(OUT / "preregistered.json") == protocol_sha
    assert sha(Path(__file__)) == protocol["run_source_sha256"]
    assert sha(ROOT / "phishing_url/v9_margin.py") == protocol["model_source_sha256"]
    pools = threadpool_info()
    assert all(pool["num_threads"] <= 2 for pool in pools)
    report = {"rows": len(dev), "new_classifier_fits": 12, "total_margin_branch_classifier_fits": 16,
              "coverage_exactly_once": True, "protocol_sha256": protocol_sha,
              "oof_sha256": sha(OUT / "development_oof.npz"), "data_sha256": digest,
              "seconds": time.perf_counter()-start, "threadpools": pools,
              "project_files_read": sorted(READS), "original_inner_gate_remains_false": True,
              "external_test_labels_used": False, "sparse_matrices_saved": False}
    write("execution.json", report)
    emit("margin_oof_complete", rows=len(dev), seconds=report["seconds"], new_classifier_fits=12)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
