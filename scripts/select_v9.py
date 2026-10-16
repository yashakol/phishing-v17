"""Freeze the accepted route before any holdout/test predictions are read."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v6.screen_extra import sha
from experiments.v9_routing.search import ERROR_TITLE as SCREEN_ERROR_TITLE
from phishing_url.v9_routing import ERROR_TITLE


def main():
    out = ROOT / "reports/v9"
    assert not (out / "selection.json").exists(), "Do not silently revise frozen selection"
    selection = json.loads((ROOT / "experiments/v9_routing/selection.json").read_text())
    assert selection["gate_passed"] and selection["selected"]["gate_passed"]
    assert selection["protocol"]["source_sha256"] == sha(ROOT / "experiments/v9_routing/search.py")
    assert ERROR_TITLE.pattern == SCREEN_ERROR_TITLE.pattern
    assert selection["selected"]["threshold"] == selection["baseline"]["threshold"]
    report = {"selected": selection["selected"], "baseline_v7": selection["baseline"],
        "protocol": selection["protocol"], "selector_breakdown": selection["selector_breakdown"],
        "selection_source": "experiments/v9_routing/selection.json",
        "selection_source_sha256": sha(ROOT / "experiments/v9_routing/selection.json"),
        "oof_source": "experiments/v9_routing/development_oof.npz",
        "oof_source_sha256": sha(ROOT / "experiments/v9_routing/development_oof.npz"),
        "train_cache_sha256": selection["train_cache_sha256"],
        "base_v7_model_sha256": sha(ROOT / "artifacts/v7/ensemble.joblib"),
        "base_v7_holdout_model_sha256": sha(ROOT / "artifacts/v7/holdout.joblib"),
        "routing_source_sha256": sha(ROOT / "phishing_url/v9_routing.py"),
        "external_test_labels_used": False, "new_classifier_fits_in_active_solution": 0,
        "public_feedback": json.loads((out / "baseline_public_feedback.json").read_text()),
        "note": "Active classifiers are own previously trained v7 models. Only conditional URL weight changes; original threshold preserved."}
    payload = json.dumps(report, indent=2)+"\n"
    (out / "selection.json").write_text(payload)
    (out / "frozen_before_holdout.json").write_text(payload)
    print(json.dumps(report["selected"]), flush=True)


if __name__ == "__main__":
    main()
