"""Select only the three predeclared SVM mixtures against frozen v9 OOF."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np

from experiments.v8_ensemble.search import training_inputs
from experiments.v9_routing.search import routed_base, finish
from experiments.v8_astra_regularization.resume import sha
from phishing_url.evaluation import best_threshold, measure


def main():
    assert not (OUT / "selection.json").exists(), "No silent mixture rescreening"
    protocol = json.loads((OUT / "preregistered.json").read_text())
    execution = json.loads((OUT / "execution.json").read_text())
    assert sha(OUT / "preregistered.json") == execution["protocol_sha256"]
    assert sha(OUT / "development_oof.npz") == execution["oof_sha256"]
    assert protocol["integration_weights"] == [0.25, 0.5, 1.0]
    assert sha(ROOT / "phishing_url/v9_margin.py") == protocol["model_source_sha256"]
    inputs = training_inputs()
    assert inputs["digest"] == execution["data_sha256"]
    frozen = json.loads((ROOT / "reports/v9/selection.json").read_text())
    policy, cut = frozen["selected"]["policy"], frozen["selected"]["threshold"]
    assert policy == {"selector": "short100_or_error", "url_weight": 0.5}
    y, indices = inputs["labels"], inputs["indices"]
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as old:
        np.testing.assert_array_equal(old["indices"], indices)
        np.testing.assert_array_equal(old["labels"], y)
        original = old["probabilities"].copy()
    with np.load(OUT / "development_oof.npz") as svm:
        np.testing.assert_array_equal(svm["indices"], indices)
        np.testing.assert_array_equal(svm["labels"], y)
        margin = svm["probabilities"].copy()
        for fold, (_, va) in enumerate(inputs["folds"]):
            np.testing.assert_array_equal(svm["folds"][va], np.full(len(va), fold))
    with np.load(ROOT / "experiments/v9_routing/development_oof.npz") as v9:
        np.testing.assert_array_equal(v9["indices"], indices)
        np.testing.assert_array_equal(v9["labels"], y)
        mask = v9[policy["selector"]].copy()
        baseline = v9["probabilities"].copy()

    def integrated(text):
        standard = routed_base(inputs, text, inputs["v5"]["config"]["aux_url_weight"])
        routed = routed_base(inputs, text, policy["url_weight"])
        return finish(inputs, np.where(mask, routed, standard))

    np.testing.assert_allclose(integrated(original), baseline, rtol=0, atol=1e-14)
    before = baseline >= cut
    baseline_scores = measure(y, baseline, cut)
    assert baseline_scores == frozen["selected"]["metrics"]
    rows, probabilities = [], []
    for weight in protocol["integration_weights"]:
        p = integrated((1-weight)*original + weight*margin)
        threshold, _ = best_threshold(y, p)
        after, fixed = p >= threshold, p >= cut
        scores = measure(y, p, threshold)
        net = int((before != y).sum() - (after != y).sum())
        frozen_net = int((before != y).sum() - (fixed != y).sum())
        folds = [{"fold": fold, "before_errors": int((before[va] != y[va]).sum()),
                  "after_errors": int((after[va] != y[va]).sum())}
                 for fold, (_, va) in enumerate(inputs["folds"])]
        passed = (net >= 5 and frozen_net >= 5 and scores["f1"] > baseline_scores["f1"] and
                  sum(item["after_errors"] < item["before_errors"] for item in folds) >= 2 and
                  all(item["after_errors"] <= item["before_errors"]+2 for item in folds))
        rows.append({"text_replacement": weight, "threshold": threshold, "metrics": scores,
                     "metrics_at_baseline_threshold": measure(y, p, cut), "net_errors": net,
                     "frozen_net": frozen_net, "fixed_errors": int(((before != y) & (after == y)).sum()),
                     "new_errors": int(((before == y) & (after != y)).sum()),
                     "fold_comparison": folds, "gate_passed": bool(passed)})
        probabilities.append(p)
    eligible = [row for row in rows if row["gate_passed"]]
    winner = max(eligible if eligible else rows,
                 key=lambda row: (row["metrics"]["f1"], row["net_errors"], -row["text_replacement"]))
    pos = rows.index(winner)
    standalone_cut, _ = best_threshold(y, margin)
    report = {"gate_passed": bool(eligible), "selected": winner, "candidates": rows,
              "baseline_v9": baseline_scores, "baseline_threshold": cut,
              "standalone_svm": measure(y, margin, standalone_cut),
              "routing": policy, "margin_parameters": protocol["parameters"],
              "protocol_sha256": execution["protocol_sha256"],
              "text_oof_sha256": execution["oof_sha256"], "train_cache_sha256": inputs["digest"],
              "v9_selection_sha256": sha(ROOT / "reports/v9/selection.json"),
              "selection_source_sha256": sha(Path(__file__)),
              "model_source_sha256": protocol["model_source_sha256"],
              "external_test_labels_used": False,
              "limitation": protocol["limitation"], "original_inner_gate_remains_false": True}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT / "selected_oof.npz", indices=indices, labels=y,
                        probabilities=probabilities[pos], baseline=baseline)
    print(json.dumps({"event": "v10_margin_selected", "gate_passed": bool(eligible),
                      "selected": winner, "baseline_v9": baseline_scores,
                      "standalone_svm": report["standalone_svm"],
                      "all_candidates": [{"weight": row["text_replacement"],
                         "errors": row["metrics"]["fp"]+row["metrics"]["fn"],
                         "frozen_net": row["frozen_net"], "f1": row["metrics"]["f1"]} for row in rows]}), flush=True)


if __name__ == "__main__":
    main()
