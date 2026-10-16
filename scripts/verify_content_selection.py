"""Replay selected OOF rules using the ordinary inference implementation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from phishing_url.content_memory import fit_memory, apply_memory
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import measure


def main():
    reports = ROOT / "reports/v3"
    selected = json.loads((reports / "ensemble_selection.json").read_text())["selected"]
    saved = np.load(reports / "development_oof.npz")
    cache = joblib.load(ROOT / "data/content_train.joblib")
    dev = aligned_splits(ROOT, cache)["development"]
    np.testing.assert_array_equal(dev, saved["indices"])
    labels = cache["labels"][dev]
    weight, auxiliary = selected["text_weight"], selected["aux_url_weight"]
    base = (1-auxiliary) * (weight*saved["text_probabilities"] + (1-weight)*saved["numeric_probabilities"]) + auxiliary*saved["url_probabilities"]
    p = base.copy()
    if selected["memory_minimum_support"] or selected["domain_minimum_support"]:
        for tr, va in StratifiedKFold(n_splits=3, shuffle=True, random_state=2026).split(dev, labels):
            memory = fit_memory(records_at(cache, dev[tr]), [cache["hashes"][int(i)] for i in dev[tr]], labels[tr],
                                selected["memory_minimum_support"], selected["domain_minimum_support"])
            p[va], _ = apply_memory(memory, records_at(cache, dev[va]), [cache["hashes"][int(i)] for i in dev[va]], base[va])
    np.testing.assert_array_equal(p, saved["probabilities"])
    metrics = measure(labels, p, selected["threshold"])
    for key in ("f1", "precision", "recall", "tn", "fp", "fn", "tp", "rows"):
        assert np.isclose(metrics[key], selected[key], rtol=0, atol=1e-12)
    result = {"ordinary_rules_equal_vectorized_selection": True, "rows": len(dev),
              "folds": 3, "parameters_frozen_without_holdout": True, "metrics": metrics}
    (reports / "selection_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
