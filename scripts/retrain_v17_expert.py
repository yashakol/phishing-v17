"""Refit the frozen full-data specialist into a separate checkpoint."""

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
    assert not args.output.exists(), "Refusing to overwrite a trained model"
    out = ROOT/"reports/v17"
    fit = json.loads((out/"expert_fit.json").read_text())
    scope = fit["guard"]
    assert sha(out/"preregistered.json") == scope["config"]["protocol_sha256"]
    assert sha(out/"full_training_frame.npz") == scope["config"]["frame_sha256"]
    assert make_expert().get_params() == fit["parameters"]
    with np.load(out/"full_training_frame.npz", allow_pickle=False) as stored:
        frame = pd.DataFrame(stored["values"], columns=stored["columns"].tolist())
        labels = stored["labels"].copy()
    assert len(frame) == len(labels) == scope["training_rows"] and frame.columns.tolist() == fit["feature_columns"]
    with threadpool_limits(limits=2):
        expert = make_expert().fit(frame, labels)
    save_component(args.output, expert, scope)
    print(json.dumps({"output": str(args.output), "training_rows": len(frame), "new_classifier_fits": 1,
                      "sha256": sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
