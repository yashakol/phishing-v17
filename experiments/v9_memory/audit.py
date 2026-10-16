"""Audit frozen development OOF decisions, without holdout/test inputs."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from experiments.v8_ensemble.search import predict_oof, training_inputs
from experiments.v5.select_ensemble import apply_values
from phishing_url.v5_bundle import mix_probabilities


def main():
    inputs = training_inputs()
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as saved:
        text = saved["probabilities"].copy()
    config = inputs["v5"]["config"]
    old = inputs["old"]
    base = mix_probabilities(text, old["numeric"], old["url"], None, old["cat"], config, old["hashed"])
    hard = apply_values(base, inputs["memory"])
    p = predict_oof(inputs, text, config)
    np.testing.assert_allclose(p, inputs["baseline"]["probabilities"], rtol=0, atol=1e-14)
    cut = inputs["v7"]["selected"]["threshold"]
    y = inputs["labels"]
    rule = hard != base
    page = (inputs["memory"]["raw"] >= 0) | (inputs["memory"]["text"] >= 0)
    domain = inputs["memory"]["domain"] >= 0
    error = (p >= cut) != y
    groups = {}
    for name, mask in {
        "all": np.ones(len(y), dtype=bool), "hard_rule": rule,
        "hard_page_rule": rule & page, "hard_domain_only": rule & ~page,
        "no_hard_rule": ~rule, "known_pure_domain": domain,
        "confident_learned": ~rule & ((base < .05) | (base > .95)),
    }.items():
        groups[name] = {"rows": int(mask.sum()), "errors": int((mask & error).sum()),
                        "fp": int((mask & error & (y == 0)).sum()), "fn": int((mask & error & (y == 1)).sum())}
    urls = joblib.load(ROOT / "experiments/v7_signatures/training_keys.joblib")["urls"]
    table = pd.DataFrame({"cache_index": inputs["indices"], "url": [urls[int(i)] for i in inputs["indices"]],
        "label": y, "probability": p, "learned_probability": base, "hard_probability": hard,
        "hard_rule": rule, "page_evidence": page, "domain_evidence": domain,
        "domain_zero": inputs["stats"]["domain_counts"][:, 0], "domain_one": inputs["stats"]["domain_counts"][:, 1],
        "raw_label": inputs["memory"]["raw"], "text_label": inputs["memory"]["text"], "domain_label": inputs["memory"]["domain"]})
    table.loc[error].to_csv(Path(__file__).resolve().parent / "development_errors.csv", index=False)
    report = {"baseline": inputs["v7"]["selected"]["metrics"], "groups": groups,
              "scope": "Frozen v7 development OOF only. No holdout/test labels or predictions.",
              "external_test_labels_used": False}
    (Path(__file__).resolve().parent / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
