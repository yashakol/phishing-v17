"""Small frozen rule-policy search on original fold-local development counts."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from experiments.v6.screen_extra import sha
from experiments.v8_ensemble.search import training_inputs
from phishing_url.evaluation import best_threshold, measure
from phishing_url.soft_context import apply_count_context
from phishing_url.v5_bundle import mix_probabilities
from phishing_url.v6_bundle import mix_extra
from phishing_url.v9_memory import DEFAULT_POLICY, MemoryEvidence, apply_evidence, pure_values


CHANGES = [{}, {"page_support": 2}, {"page_support": 5}, {"domain_support": 3},
           {"domain_support": 10}, {"domain_confidence": .99},
           {"page_support": 2, "domain_confidence": .99},
           {"page_support": 2, "domain_support": 3, "domain_confidence": .99},
           {"veto_support": 1}, {"veto_support": 2},
           {"page_support": 2, "veto_support": 1}]
POLICIES = [{**DEFAULT_POLICY, **change} for change in CHANGES]


def fold_counts(inputs):
    data = joblib.load(ROOT / "experiments/v5/development_memory.joblib")
    result = {}
    for kind, keys in data["keys"].items():
        encoded, unique = pd.factorize(np.asarray(keys, dtype=object), sort=False)
        counts = np.zeros((len(keys), 2), dtype=np.int32)
        for tr, va in inputs["folds"]:
            for label in (0, 1):
                groups = np.bincount(encoded[tr], weights=inputs["labels"][tr] == label, minlength=len(unique))
                counts[va, label] = groups[encoded[va]].astype(np.int32)
            # Independently compare counts to direct partition-fitted lookup.
            if kind == "raw":
                train = {k: [v[int(i)] for i in tr] for k, v in data["keys"].items()}
                valid = {k: [v[int(i)] for i in va] for k, v in data["keys"].items()}
                independent = MemoryEvidence(DEFAULT_POLICY).fit_keys(train, inputs["labels"][tr]).lookup(valid)
                assert np.array_equal(counts[va], independent[kind])
        result[kind] = counts
        support = DEFAULT_POLICY["domain_support"] if kind == "domain" else DEFAULT_POLICY["page_support"]
        np.testing.assert_array_equal(pure_values(counts, support), inputs["memory"][kind])
    return result


def replay(inputs, learned, counts, policy):
    hard, matched = apply_evidence(learned, counts, policy)
    # Legacy v5 preserves matches; exact 0/1 learned coincidences are irrelevant
    # to context (.05-.95 only), but use the old mask for exact baseline replay.
    changed = hard != learned
    contextual = np.empty(len(learned))
    for tr, va in inputs["folds"]:
        contextual[va] = apply_count_context(learned[va], {
            "domain": inputs["stats"]["domain_counts"][va], "title": inputs["stats"]["title_counts"][va]},
            float(inputs["labels"][tr].mean()), inputs["v5"]["context_config"])
    contextual[changed] = hard[changed]
    p = mix_extra(contextual, inputs["extra"]["extra_text"], None, inputs["v6"]["config"])
    for fold, (_, va) in enumerate(inputs["folds"]):
        p[va], _ = inputs["hierarchy_models"][fold].predict_keys(inputs["hierarchy_keys"][fold], p[va])
    return p, matched


def main(save=True):
    protocol = {"policies": POLICIES, "configurations": len(POLICIES),
        "baseline": "Frozen v7 C16 mixture, context, extra HTML and hierarchy unchanged",
        "scope": "Original development OOF only; fit evidence separately in each original training fold",
        "selection": "Require >=5 net fewer errors, higher F1, at least two improved folds, no fold worsening by >2",
        "gate_at_frozen_baseline_threshold": True,
        "limitation": "Exploratory policy search on previously used development labels; no independent public/private guarantee",
        "source_sha256": sha(Path(__file__)), "evidence_source_sha256": sha(ROOT / "phishing_url/v9_memory.py")}
    if save:
        assert not (OUT / "preregistered.json").exists(), "No silent reselection"
        (OUT / "preregistered.json").write_text(json.dumps(protocol, indent=2) + "\n")
    inputs = training_inputs()
    counts = fold_counts(inputs)
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as text:
        np.testing.assert_array_equal(text["indices"], inputs["indices"])
        np.testing.assert_array_equal(text["labels"], inputs["labels"])
        config = inputs["v5"]["config"]
        learned = mix_probabilities(text["probabilities"], inputs["old"]["numeric"], inputs["old"]["url"],
            None, inputs["old"]["cat"], config, inputs["old"]["hashed"])
    baseline, _ = replay(inputs, learned, counts, DEFAULT_POLICY)
    np.testing.assert_allclose(baseline, inputs["baseline"]["probabilities"], rtol=0, atol=1e-14)
    cut = inputs["v7"]["selected"]["threshold"]
    y = inputs["labels"]
    before = baseline >= cut
    base_errors = int((before != y).sum())
    rows, probabilities = [], []
    for number, policy in enumerate(POLICIES):
        p, hits = replay(inputs, learned, counts, policy)
        selected_cut, _ = best_threshold(y, p)
        pred = p >= selected_cut
        folds = [{"fold": f, "before_errors": int((before[va] != y[va]).sum()),
                  "after_errors": int((pred[va] != y[va]).sum())} for f, (_, va) in enumerate(inputs["folds"])]
        errors = int((pred != y).sum())
        fixed_pred = p >= cut
        frozen_net = int((before != y).sum() - (fixed_pred != y).sum())
        passes = (base_errors-errors >= 5 and frozen_net >= 5 and
            measure(y, p, selected_cut)["f1"] > inputs["v7"]["selected"]["metrics"]["f1"] and
            sum(r["after_errors"] < r["before_errors"] for r in folds) >= 2 and
            all(r["after_errors"] <= r["before_errors"]+2 for r in folds))
        row = {"number": number, "policy": policy, "threshold": selected_cut,
            "metrics": measure(y, p, selected_cut), "metrics_at_frozen_baseline_threshold": measure(y, p, cut),
            "frozen_net_fewer_errors": frozen_net, "fixed_errors": int(((before != y) & (pred == y)).sum()),
            "new_errors": int(((before == y) & (pred != y)).sum()), "evidence_rows": int(hits.sum()),
            "fold_comparison": folds, "gate_passed": bool(passes)}
        rows.append(row)
        probabilities.append(p)
    eligible = [r for r in rows if r["gate_passed"]]
    chosen = max(eligible if eligible else rows, key=lambda r: r["metrics"]["f1"])
    report = {"selected": chosen, "gate_passed": bool(eligible), "baseline": inputs["v7"]["selected"]["metrics"],
        "candidates": rows, "protocol": protocol, "train_cache_sha256": inputs["digest"],
        "external_test_labels_used": False}
    if save:
        (OUT / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(OUT / "development_oof.npz", indices=inputs["indices"], labels=y,
            probabilities=probabilities[chosen["number"]], baseline=baseline)
        print(json.dumps({"event": "v9_policy_selected", "gate_passed": bool(eligible), "selected": chosen,
            "all_candidates": [{"number":r["number"], "errors":r["metrics"]["fp"]+r["metrics"]["fn"],
                "frozen_net":r["frozen_net_fewer_errors"], "f1":r["metrics"]["f1"],"passes":r["gate_passed"]} for r in rows]}), flush=True)
    return report, probabilities[chosen["number"]]


if __name__ == "__main__":
    main()
