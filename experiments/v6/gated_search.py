"""Coarse expert gating based on OOF confidence and visible-text availability."""

import json
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.v6.screen_extra import sha
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v6_bundle import mix_extra, gate_extra


def main():
    old5 = np.load(ROOT / "reports/v5/development_oof.npz")
    old3 = np.load(ROOT / "reports/v3/development_oof.npz")
    extra = np.load(ROOT / "experiments/v6/extra_text_development_oof.npz")
    y, indices = old5["labels"], old5["indices"]
    for table in (old3, extra):
        np.testing.assert_array_equal(table["indices"], indices)
        np.testing.assert_array_equal(table["labels"], y)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    length = cache["features"].html_text_length.to_numpy()[indices]
    del cache
    bases = {3: old3["probabilities"], 5: old5["probabilities"]}
    oldcut = json.loads((ROOT / "reports/v5/selection.json").read_text())["selected"]["threshold"]
    before = bases[5] >= oldcut
    folds = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(indices, y))
    rows, probabilities = [], {}
    for version, confidence, max_text, weight in product((3, 5), (None, .9, .975), (None, 80), (.1, .25, .4)):
        gate = np.ones(len(y), dtype=bool)
        if confidence:
            gate &= (extra["probabilities"] >= confidence) | (extra["probabilities"] <= 1 - confidence)
        if max_text is not None:
            gate &= length <= max_text
        config = {"base_version": version, "extra_text_weight": weight, "extra_boost_weight": 0.,
                  "preserve_training_rules": True, "neighbor_adjustment": None,
                  "extra_confidence": confidence, "maximum_visible_text": max_text}
        p = mix_extra(bases[version], extra["probabilities"], None, config)
        p = gate_extra(bases[version], p, extra["probabilities"], length, config)
        cut, _ = best_threshold(y, p)
        pred = p >= cut
        row = {"config": config, "threshold": cut, "metrics": measure(y, p, cut),
               "fixed_v5_errors": int(np.sum((before != y) & (pred == y))),
               "new_errors": int(np.sum((before == y) & (pred != y))), "gate_rows": int(gate.sum()),
               "fold_comparison": [{"fold": n, "v5_f1": measure(y[va], bases[5][va], oldcut)["f1"],
                                    "v6_f1": measure(y[va], p[va], cut)["f1"],
                                    "v5_errors": int(np.sum(before[va] != y[va])),
                                    "v6_errors": int(np.sum(pred[va] != y[va]))}
                                   for n, (_, va) in enumerate(folds)]}
        rows.append(row)
        probabilities[json.dumps(config, sort_keys=True)] = p
    selected = max(rows, key=lambda row: (row["metrics"]["f1"], -row["new_errors"]))
    ordinary = json.loads((ROOT / "reports/v6/selection.json").read_text())
    report = {**ordinary, "selected": selected, "candidates": rows,
              "ungated_selection": ordinary["selected"],
              "protocol": "Original OOF, two frozen baselines, coarse confidence/short-text gating and three text weights. No holdout/test/added-row labels used. Gating motivated by the prior residual-error audit."}
    (ROOT / "reports/v6/gated_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(ROOT / "reports/v6/gated_development_oof.npz", indices=indices, labels=y,
                        probabilities=probabilities[json.dumps(selected["config"], sort_keys=True)],
                        base_v3=bases[3], base_v5=bases[5], extra_text=extra["probabilities"],
                        extra_boost=np.zeros(len(y)), visible_text_length=length)
    print(json.dumps({"event": "gated_oof_selection", "selected": selected,
                      "ungated": ordinary["selected"]}), flush=True)


if __name__ == "__main__":
    main()
