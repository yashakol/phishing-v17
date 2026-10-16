"""Three-fold HTML boosting predictions using the original development folds."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from boost import load_frame, fit
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import measure, best_threshold


def main():
    out = ROOT / "experiments/content"
    config = json.loads((out / "selection.json").read_text())["selected"]["config"]
    cache = joblib.load(ROOT / "data/content_train.joblib")
    frame = load_frame(cache)
    dev = aligned_splits(ROOT, cache)["development"]
    labels = cache["labels"][dev]
    del cache["records"], cache["hashes"]
    probabilities = np.zeros(len(dev), dtype=np.float64)
    folds = np.full(len(dev), -1, dtype=np.int8)
    for fold, (tr, va) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=2026).split(dev, labels)):
        saved = out / f"fold_{fold}.npz"
        if saved.exists():
            prior = np.load(saved)
            if not np.array_equal(prior["indices"], dev[va]):
                raise ValueError("Saved fold indexing differs")
            if str(prior["config_json"]) != json.dumps(config, sort_keys=True):
                raise ValueError("Saved fold configuration differs")
            p = prior["probabilities"]
        else:
            started = time.perf_counter()
            print("Training HTML boost OOF fold", fold, config, flush=True)
            model = fit(frame.iloc[dev[tr]], labels[tr], config)
            p = model.predict_proba(frame.iloc[dev[va]])[:, 1]
            np.savez_compressed(saved, indices=dev[va], probabilities=p, config_json=json.dumps(config, sort_keys=True))
            print("Fold complete", fold, "seconds", round(time.perf_counter()-started, 1), flush=True)
        probabilities[va] = p
        folds[va] = fold
    threshold, _ = best_threshold(labels, probabilities)
    metrics = measure(labels, probabilities, threshold)
    np.savez_compressed(out / "development_oof.npz", indices=dev, labels=labels,
                        probabilities=probabilities, folds=folds)
    (out / "oof_metrics.json").write_text(json.dumps({"config": config, "metrics": metrics,
         "protocol": "Original development split; fresh HTML/URL model trained separately per fold; threshold optimized on OOF."}, indent=2))
    print("HTML boost development OOF:", metrics, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
