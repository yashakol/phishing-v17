"""Check completed real-data OOF alignment, checkpoints and offline inference."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gc
import json
import socket
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from experiments.html_astra.screen import aligned_development, digest
from experiments.html_astra.run_oof import canonical_hash, validated_checkpoint, atomic_json


def main():
    records, labels, urls, original, indices, _, _, _ = aligned_development(ROOT / "data/content_train.joblib")
    guard = json.loads((OUT / "oof/run_guard.json").read_text())
    guard_digest = canonical_hash(guard)
    report = json.loads((OUT / "oof_results.json").read_text())
    assert report["guard_digest"] == guard_digest
    assert guard["cache_sha256"] == digest(ROOT / "data/content_train.joblib")
    assert guard["model_source_sha256"] == digest(ROOT / "phishing_url/html_text.py")
    assert guard["selection_sha256"] == digest(OUT / "selection.json")
    assert report["output_sha256"] == digest(OUT / "development_oof.npz")
    with np.load(OUT / "development_oof.npz", allow_pickle=False) as output:
        np.testing.assert_array_equal(output["indices"], indices)
        np.testing.assert_array_equal(output["cache_indices"], indices)
        np.testing.assert_array_equal(output["clean_indices"], original)
        np.testing.assert_array_equal(output["labels"], labels)
        np.testing.assert_array_equal(output["y"], labels)
        np.testing.assert_array_equal(output["urls"], urls)
        probabilities, folds = output["probabilities"].copy(), output["folds"].copy()
    assert len(np.unique(indices)) == len(indices)
    assert np.isfinite(probabilities).all()
    expected_folds = np.full(len(labels), -1, dtype=np.int8)
    checked = []
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled")), \
         patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS disabled")):
        for fold, (train, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(indices, labels)):
            expected_folds[valid] = fold
            assert not np.intersect1d(indices[train], indices[valid]).size
            restored = validated_checkpoint(OUT / "oof", fold, guard_digest, indices[train], indices[valid], labels[valid])
            assert restored is not None
            p, metadata = restored
            np.testing.assert_array_equal(probabilities[valid], p)
            model = joblib.load(OUT / "oof" / f"fold_{fold}.joblib")
            assert model.get_params() == guard["model_parameters"]
            subset = valid[:32]
            np.testing.assert_array_equal(model.predict_proba([records[i] for i in subset]), p[:32])
            checked.append(dict(fold=fold, rows=len(valid), offline_serialization_rows=len(subset)))
            del model
            gc.collect()
    np.testing.assert_array_equal(folds, expected_folds)
    result = dict(status="passed", rows=len(labels), aligned_original_development=True,
                  original_three_fold_membership_verified=True, checkpoint_guards_verified=True,
                  offline_serialization_checked=checked, output_sha256=report["output_sha256"])
    atomic_json(OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
