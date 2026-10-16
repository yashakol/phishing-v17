"""Three-fold development OOF of frozen selected configuration, then dev fit + test."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import gc
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.data import load_train, sha256
from phishing_url.features import group_for_url
from phishing_url.evaluation import measure
from url_model import URLModel
from predict_test import predict


def main():
    start = time.perf_counter()
    selected = json.loads((OUT / "selection.json").read_text())["selected"]
    urls, labels, _ = load_train(ROOT / "data/train.csv")
    indices = np.load(ROOT / "reports/split_indices.npz")["development"]
    urls = np.asarray(urls, dtype=object)[indices]
    y = labels[indices]
    del labels
    p = np.full(len(y), np.nan)
    fold = np.full(len(y), -1, dtype=np.int8)
    unseen = np.zeros(len(y), dtype=bool)
    timings = []
    for f, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(urls, y)):
        fold_start = time.perf_counter()
        model = URLModel(selected["representation"], selected["c"], selected["nb"])
        model.fit(urls[fit].tolist(), y[fit])
        prediction_start = time.perf_counter()
        p[valid] = model.predict_proba(urls[valid].tolist())
        predict_seconds = time.perf_counter() - prediction_start
        fold[valid] = f
        seen = set(group_for_url(u) for u in urls[fit])
        unseen[valid] = [group_for_url(u) not in seen for u in urls[valid]]
        row = {"fold": f, "fit_seconds": model.fit_seconds, "predict_seconds": predict_seconds,
               "total_seconds": time.perf_counter() - fold_start, "n_features": model.n_features,
               "train_rows": len(fit), "valid_rows": len(valid),
               "metrics": measure(y[valid], p[valid], selected["threshold"])}
        timings.append(row)
        np.savez_compressed(OUT / "oof_partial.npz", indices=indices, y=y, probabilities=p, fold=fold,
                            unseen_domain=unseen, parameters=json.dumps(selected))
        print(json.dumps({"event": "fold_complete", **row}), flush=True)
        del model
        gc.collect()
    assert np.isfinite(p).all() and (fold >= 0).all()
    np.savez_compressed(OUT / "development_oof.npz", indices=indices, y=y, probabilities=p, fold=fold,
                        unseen_domain=unseen, parameters=json.dumps(selected))
    (OUT / "oof_partial.npz").unlink()
    result = {"scope": "Development-only OOF; hyperparameters chosen within development, not an untouched score.",
              "selected": selected, "split": {"class": "StratifiedKFold", "n_splits": 3,
                                                "shuffle": True, "random_state": 2026},
              "data_sha256": sha256(ROOT / "data/train.csv"),
              "split_sha256": sha256(ROOT / "reports/split_indices.npz"),
              "folds": timings, "oof": measure(y, p, selected["threshold"]),
              "oof_at_05": measure(y, p),
              "oof_unseen_domain": measure(y[unseen], p[unseen], selected["threshold"]),
              "oof_seconds": time.perf_counter() - start}
    (OUT / "oof_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "oof_complete", **result}), flush=True)
    final_model = URLModel(selected["representation"], selected["c"], selected["nb"])
    final_model.fit(urls.tolist(), y)
    joblib.dump(final_model, OUT / "development_model.joblib", compress=3)
    result["development_fit_seconds"] = final_model.fit_seconds
    result["test"] = predict(final_model, selected["threshold"])
    result["total_seconds"] = time.perf_counter() - start
    result["threadpools"] = threadpool_info()
    (OUT / "oof_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "all_complete", **result}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
