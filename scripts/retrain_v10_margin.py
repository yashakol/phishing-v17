"""Repeat new SVM training in a separate file without replacing saved models."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.v10_training import representation_snapshot, sha
from phishing_url.v9_margin import CalibratedMarginText


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("full", "holdout"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output file")
    selection = json.loads((ROOT / "reports/v10/selection.json").read_text())
    assert selection["model_source_sha256"] == sha(ROOT / "phishing_url/v9_margin.py")
    assert selection["train_cache_sha256"] == sha(ROOT / "data/content_train.joblib")
    cache = joblib.load(ROOT / "data/content_train.joblib")
    indices = (np.arange(len(cache["urls"]), dtype=np.int64) if args.stage == "full"
               else aligned_splits(ROOT, cache)["development"])
    c16 = json.loads((ROOT / "reports/v7/selection.json").read_text())["text_parameters"]
    config = {**c16, "source_sha256": sha(ROOT / "phishing_url/v7_text.py"),
              "html_text_source_sha256": sha(ROOT / "phishing_url/html_text.py")}
    expected = guard(cache, indices, config, selection["train_cache_sha256"])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    origin = ROOT / f"artifacts/v7/component_cache/{args.stage}_text_{tag}.joblib"
    base = load_component(origin, expected)
    before = representation_snapshot(base)
    with threadpool_limits(limits=2):
        X = base._matrix(records_at(cache, indices))
        model = CalibratedMarginText(**selection["margin_parameters"]).fit_representation(base, X, cache["labels"][indices])
        del X
    assert representation_snapshot(base) == before
    new_guard = guard(cache, indices, {"parameters": selection["margin_parameters"],
        "base_guard": expected, "base_component_sha256": sha(origin),
        "model_source_sha256": sha(ROOT / "phishing_url/v9_margin.py"),
        "retrain_source_sha256": sha(Path(__file__))}, selection["train_cache_sha256"])
    save_component(args.output, model, new_guard)
    restored = load_component(args.output, new_guard)
    sample = records_at(cache, indices[:32])
    np.testing.assert_array_equal(model.predict_proba(sample), restored.predict_proba(sample))
    original = joblib.load(ROOT / f"artifacts/v10/component_cache/{args.stage}_margin.joblib")
    assert original["guard"] == guard(cache, indices, original["guard"]["config"], selection["train_cache_sha256"])
    difference = float(np.max(np.abs(original["model"].predict_proba(sample)-model.predict_proba(sample))))
    print(json.dumps({"output": str(args.output.resolve()), "training_rows": len(indices),
        "classifier_fits": model.classifier_fits_, "serialization_replay_rows": 32,
        "max_difference_from_original_on_32_rows": difference, "sha256": sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
