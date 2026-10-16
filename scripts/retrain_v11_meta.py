"""Fit the same development-only meta classifier into a separate guarded file."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import save_component
from phishing_url.v10_training import sha
from phishing_url.v11_meta import LearnedEnsemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace an existing trained model")
    fit = json.loads((ROOT / "reports/v11/meta_fit.json").read_text())
    guard = fit["guard"]
    for path, key in (("experiments/v11_meta/training_frame.npz", "frame_sha256"),
                      ("reports/v11/selection.json", "selection_sha256"),
                      ("phishing_url/v11_meta.py", "source_sha256")):
        if sha(ROOT / path) != guard["config"][key]:
            raise ValueError("Frozen own-training input changed: "+path)
    with np.load(ROOT / "experiments/v11_meta/training_frame.npz", allow_pickle=False) as data:
        X = pd.DataFrame(data["values"], columns=fit["feature_columns"])
        y = data["labels"].copy()
    if len(y) != guard["training_rows"]:
        raise ValueError("Meta training rows differ")
    with threadpool_limits(limits=2):
        model = LearnedEnsemble(**fit["parameters"]).fit(X, y)
    save_component(args.output, model, guard)
    print(json.dumps({"output": str(args.output), "training_rows": len(y), "new_classifier_fits": 1,
                      "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
