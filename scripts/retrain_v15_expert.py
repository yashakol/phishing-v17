"""Refit the new development-only classifier into a separate guarded checkpoint."""

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
from phishing_url.v12_expert import make_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace an existing trained model")
    fit = json.loads((ROOT / "reports/v15/expert_fit.json").read_text())
    scope = fit["guard"]
    for path, key in (("experiments/v15_resources/training_frame.npz", "frame_sha256"),
                      ("reports/v15/selection.json", "selection_sha256"),
                      ("phishing_url/v15_resources.py", "expert_source_sha256")):
        if sha(ROOT / path) != scope["config"][key]:
            raise ValueError("Frozen own-training input changed: "+path)
    if make_expert().get_params() != fit["parameters"]:
        raise ValueError("Frozen classifier parameters differ")
    with np.load(ROOT / "experiments/v15_resources/training_frame.npz", allow_pickle=False) as data:
        X = pd.DataFrame(data["values"], columns=data["columns"].tolist())
        mask, y = data["eligibility"].copy(), data["labels"].copy()
    if int(mask.sum()) != scope["training_rows"] or list(X.columns) != fit["feature_columns"]:
        raise ValueError("Own-training rows or schema differ")
    with threadpool_limits(limits=2):
        model = make_expert().fit(X.iloc[mask], y[mask])
    save_component(args.output, model, scope)
    print(json.dumps({"output": str(args.output), "training_rows": int(mask.sum()),
                      "new_classifier_fits": 1, "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
