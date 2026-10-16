"""Compare a separate, actually refitted specialist with the active component."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import load_component
from phishing_url.v10_training import sha
from phishing_url.v12_expert import make_expert


def main():
    report_dir = ROOT / "reports/v15"
    fit = json.loads((report_dir / "expert_fit.json").read_text())
    scope = fit["guard"]
    frame_path = ROOT / "experiments/v15_resources/training_frame.npz"
    assert sha(frame_path) == scope["config"]["frame_sha256"]
    original_path = ROOT / "artifacts/v15/component_cache/development_expert.joblib"
    refitted_path = report_dir / "retrained_expert.joblib"
    original = load_component(original_path, scope)
    refitted = load_component(refitted_path, scope)
    assert original.get_params() == refitted.get_params() == make_expert().get_params() == fit["parameters"]
    with np.load(frame_path, allow_pickle=False) as data:
        frame = pd.DataFrame(data["values"], columns=data["columns"].tolist())
        eligibility = data["eligibility"].copy()
    assert frame.columns.tolist() == fit["feature_columns"]
    assert int(eligibility.sum()) == scope["training_rows"]
    with threadpool_limits(limits=2):
        before = original.predict_proba(frame, thread_count=2)
        after = refitted.predict_proba(frame, thread_count=2)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert np.array_equal(before, after), "Repeated training produced different probabilities"
    result = {"status": "passed", "training_rows": int(eligibility.sum()),
              "checked_prediction_rows": len(frame), "new_classifier_fits": 1,
              "training_scope_exactly_equal": True, "parameters_exactly_equal": True,
              "predictions_exactly_equal": True, "max_probability_difference": float(np.max(np.abs(before-after))),
              "source_sha256": sha(ROOT / "scripts/retrain_v15_expert.py"),
              "original_checkpoint_sha256": sha(original_path),
              "refitted_checkpoint_sha256": sha(refitted_path),
              "frame_sha256": sha(frame_path),
              "environment": "Current pinned local environment; no external or test labels"}
    (report_dir / "retraining_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
