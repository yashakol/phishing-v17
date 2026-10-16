"""Diagnose the already-screened fold-consistent alternative, without refitting.

This follow-up was initiated after the text replacement failed diagnostic
holdout. It is explicitly exploratory; the holdout is not an independent test.
The alternative is fixed from development OOF before its holdout/test replay.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from phishing_url.evaluation import measure
from phishing_url.v7_hierarchy import hierarchy_keys


def main():
    selection = json.loads((ROOT / "reports/v7/selection.json").read_text())
    stable = [row for row in selection["candidates"]
              if all(fold["candidate_errors"] <= fold["baseline_errors"] for fold in row["fold_comparison"])]
    chosen = max(stable, key=lambda row: row["metrics"]["f1"])
    assert chosen["config"] == {"text_replacement": 0.0, "hierarchy": True}
    assert chosen["threshold"] == selection["baseline_v6"]["threshold"]
    out = ROOT / "experiments/v7_hierarchy/conservative_diagnostic.json"
    report = {"chosen_before_alternative_holdout_test": chosen,
              "protocol": "Exploratory follow-up after v7 text holdout; choose fold-consistent existing OOF alternative. No new fit or parameters.",
              "independent_holdout": False, "external_test_labels_used": False}
    out.write_text(json.dumps(report, indent=2) + "\n")
    def load_hierarchy(stage, rows):
        paths = list((ROOT / "artifacts/v7/component_cache").glob(f"{stage}_hierarchy_*.joblib"))
        assert len(paths) == 1
        stored = joblib.load(paths[0])
        assert stored["guard"]["training_rows"] == stored["model"].training_rows_ == rows
        assert stored["guard"]["data_sha256"] == selection["train_cache_sha256"]
        return stored["model"]
    model = load_hierarchy("holdout", 50977)
    hold = pd.read_csv(ROOT / "reports/v6/holdout_predictions.csv")
    p, hits = model.predict_keys([hierarchy_keys(url) for url in hold.url], hold.probability.to_numpy())
    y = hold.label.to_numpy()
    before, after = hold.prediction.to_numpy(), p >= chosen["threshold"]
    report.update(holdout=measure(y, p, chosen["threshold"]),
                  holdout_fixed_errors=int(np.sum((before != y) & (after == y))),
                  holdout_new_errors=int(np.sum((before == y) & (after != y))),
                  holdout_hierarchy_matches=int(hits.sum()))
    model = load_hierarchy("full", 63900)
    test = pd.read_csv(ROOT / "reports/v6/test_prediction_provenance.csv")
    assert test.url.tolist() == pd.read_csv(ROOT / "data/test.csv").url.tolist()
    p, hits = model.predict_keys([hierarchy_keys(url) for url in test.url], test.probability.to_numpy())
    before = pd.read_csv(ROOT / "submissions/submission_content_v6.csv").Predicted.to_numpy()
    after = p >= chosen["threshold"]
    report.update(test_prediction_changes_vs_v6=int(np.sum(before != after)),
                  test_hierarchy_matches=int(hits.sum()), test_rows=len(test))
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
