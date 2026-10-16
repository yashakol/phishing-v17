"""Predict the untouched test rows using a saved Astra URLModel. No network calls."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import argparse
import json
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from phishing_url.data import load_test, write_submission
import url_model  # Required for joblib's importable custom estimator.


def predict(model, threshold, output_prefix="test"):
    if Path(output_prefix).name != output_prefix:
        raise ValueError("Output prefix must be a plain filename within experiments/astra")
    start = time.perf_counter()
    urls = load_test(ROOT / "data/test.csv")
    proba = model.predict_proba(urls)
    assert proba.shape == (len(urls),) and np.isfinite(proba).all()
    np.savez_compressed(OUT / f"{output_prefix}_predictions.npz", indices=np.arange(len(urls)),
                        probabilities=proba, threshold=threshold)
    submission = write_submission(ROOT / "data/sample.csv", OUT / f"{output_prefix}_submission.csv",
                                  (proba >= threshold).astype(int))
    return {"rows": len(urls), "positive": int((proba >= threshold).sum()),
            "threshold": threshold, "predict_seconds": time.perf_counter() - start,
            "submission": submission}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=OUT / "development_model.joblib")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output-prefix", default="test")
    args = parser.parse_args()
    threshold = args.threshold
    if threshold is None:
        threshold = json.loads((OUT / "selection.json").read_text())["selected"]["threshold"]
    with threadpool_limits(limits=2):
        print(json.dumps(predict(joblib.load(args.model), threshold, args.output_prefix), indent=2))
