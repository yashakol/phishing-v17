"""Select extra-HTML mixture weights from original development OOF only."""

import json
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.v6.screen_extra import sha
from phishing_url.evaluation import best_threshold, measure
from phishing_url.neighbors import adjust_neighbors
from phishing_url.v6_bundle import mix_extra


def main():
    old3 = np.load(ROOT / "reports/v3/development_oof.npz")
    old5 = np.load(ROOT / "reports/v5/development_oof.npz")
    extra_text = np.load(ROOT / "experiments/v6/extra_text_development_oof.npz")
    boost_path = ROOT / "experiments/v6/extra_boost_development_oof.npz"
    extra_boost = np.load(boost_path) if boost_path.exists() else None
    neighbors = np.load(ROOT / "experiments/v6/url_neighbors_development_oof.npz")
    y, indices = old5["labels"], old5["indices"]
    for table in (old3, extra_text, neighbors, extra_boost):
        if table is None:
            continue
        np.testing.assert_array_equal(indices, table["indices"])
        np.testing.assert_array_equal(y, table["labels"])
    thresholds = {3: json.loads((ROOT / "reports/v3/ensemble_selection.json").read_text())["selected"]["threshold"],
                  5: json.loads((ROOT / "reports/v5/selection.json").read_text())["selected"]["threshold"]}
    bases = {3: old3["probabilities"], 5: old5["probabilities"]}
    previous = bases[5] >= thresholds[5]
    candidates, probabilities = [], {}
    folds = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(indices, y))
    chosen_neighbor = json.loads((ROOT / "experiments/v6/url_neighbor_selection.json").read_text())["selected"]["adjustment"]
    neighbor_configs = [None, chosen_neighbor,
                        {**chosen_neighbor, "minimum_similarity": .85, "weight": .35, "minimum_support": 2}]
    boost_weights = (0., .1, .25, .4) if extra_boost is not None else (0.,)
    for version, tw, bw, neighbor in product((3, 5), (0., .1, .25, .4), boost_weights, neighbor_configs):
        if tw + bw > .6:
            continue
        config = {"base_version": version, "extra_text_weight": tw, "extra_boost_weight": bw,
                  "preserve_training_rules": True, "neighbor_adjustment": neighbor}
        p = mix_extra(bases[version], extra_text["probabilities"],
                      extra_boost["probabilities"] if extra_boost is not None else None, config)
        if neighbor:
            p = adjust_neighbors(p, neighbors["similarities"], neighbors["neighbor_labels"],
                                 {**neighbor, "prior": neighbors["prior"]})
        cut, _ = best_threshold(y, p)
        pred = p >= cut
        result = {"config": config, "threshold": cut, "metrics": measure(y, p, cut),
                  "fixed_v5_errors": int(np.sum((previous != y) & (pred == y))),
                  "new_errors": int(np.sum((previous == y) & (pred != y))),
                  "fold_comparison": [{"fold": f, "v5_f1": measure(y[va], bases[5][va], thresholds[5])["f1"],
                                       "v6_f1": measure(y[va], p[va], cut)["f1"],
                                       "v5_errors": int(np.sum(previous[va] != y[va])),
                                       "v6_errors": int(np.sum(pred[va] != y[va]))}
                                      for f, (_, va) in enumerate(folds)]}
        candidates.append(result)
        probabilities[json.dumps(config, sort_keys=True)] = p
    chosen = max(candidates, key=lambda row: (row["metrics"]["f1"], -row["new_errors"],
                  -row["config"]["extra_text_weight"] - row["config"]["extra_boost_weight"]))
    report = {"selected": chosen, "baseline_v3": measure(y, bases[3], thresholds[3]),
              "baseline_v5": measure(y, bases[5], thresholds[5]), "candidates": candidates,
              "train_cache_sha256": sha(ROOT / "data/content_train.joblib"),
              "extra_cache_sha256": sha(ROOT / "data/extra_html_train.joblib"),
              "protocol": "Two frozen learned baselines, a coarse extra-HTML mixture grid and three optional URL neighbor policies, preserved training rules. Weights/threshold selected using original development OOF only; no holdout/test/added-row labels.",
              "external_test_labels_used": False, "new_astra_calls": 0}
    out = ROOT / "reports/v6"
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "development_oof.npz", indices=indices, labels=y,
                        probabilities=probabilities[json.dumps(chosen["config"], sort_keys=True)],
                        base_v3=bases[3], base_v5=bases[5], extra_text=extra_text["probabilities"],
                        extra_boost=extra_boost["probabilities"] if extra_boost is not None else np.zeros(len(y)))
    print(json.dumps({"event": "v6_oof_selection_complete", "selected": chosen,
                      "baseline_v5": report["baseline_v5"]}), flush=True)


if __name__ == "__main__":
    main()
