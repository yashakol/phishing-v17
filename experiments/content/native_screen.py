"""Try native text boosting on the same inner/outer development splits."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.content.boost import load_frame
from phishing_url.content_catboost import NativeContentBoost
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure


def main():
    cache = joblib.load(ROOT / "data/content_train.joblib")
    frame = load_frame(cache)
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    y = cache["labels"]
    started = time.perf_counter()
    print("Native text boost inner tuning", flush=True)
    model = NativeContentBoost().fit(frame.iloc[tr], records_at(cache, tr), y[tr],
                    (frame.iloc[va], records_at(cache, va), y[va]))
    p = model.predict_proba(frame.iloc[va], records_at(cache, va))
    threshold, _ = best_threshold(y[va], p)
    result = {"config": model.configuration(), "threshold": threshold,
              "inner": measure(y[va], p, threshold), "inner_seconds": time.perf_counter()-started,
              "native_text_documentation": "https://catboost.ai/docs/en/references/text-processing__feature_calcers"}
    folder = ROOT / "experiments/content"
    (folder / "native_selection.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(folder / "native_inner.npz", indices=va, labels=y[va], probabilities=p)
    print("Native text inner:", result, flush=True)
    tr, va = splits["screening_train"], splits["screening_validation"]
    started = time.perf_counter()
    model = NativeContentBoost(**result["config"]).fit(frame.iloc[tr], records_at(cache, tr), y[tr])
    p = model.predict_proba(frame.iloc[va], records_at(cache, va))
    result["screen"] = measure(y[va], p, threshold)
    result["screen_seconds"] = time.perf_counter()-started
    (folder / "native_results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(folder / "native_screen.npz", indices=va, labels=y[va], probabilities=p)
    joblib.dump(model, folder / "native_screen.joblib", compress=3)
    print("Native text screen:", result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
