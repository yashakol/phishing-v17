"""Route weak/blocked HTML toward the existing URL model, using honest OOF."""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from experiments.v6.screen_extra import sha
from experiments.v8_ensemble.search import training_inputs
from experiments.v5.select_ensemble import apply_values
from phishing_url.evaluation import best_threshold, measure
from phishing_url.soft_context import apply_count_context
from phishing_url.v5_bundle import mix_probabilities
from phishing_url.v6_bundle import mix_extra


ERROR_TITLE = re.compile(r"(?:\b(?:403|404|410|500|502|503|forbidden|unavailable)\b|access denied|not found|bad gateway|"
                         r"internal server error|just a moment|attention required|checking your browser|domain (?:expired|parked))", re.I)
POLICIES = [{"selector": selector, "url_weight": weight} for selector in
            ("empty", "short100", "error_title", "short100_or_error") for weight in (.35, .5, .7)]


def routed_base(inputs, text, url_weight):
    old, config = inputs["old"], inputs["v5"]["config"]
    varied = dict(config, aux_url_weight=url_weight)
    return mix_probabilities(text, old["numeric"], old["url"], None, old["cat"], varied, old["hashed"])


def finish(inputs, learned):
    hard = apply_values(learned, inputs["memory"])
    matched = hard != learned
    contextual = np.empty(len(learned))
    for tr, va in inputs["folds"]:
        contextual[va] = apply_count_context(learned[va], {
            "domain": inputs["stats"]["domain_counts"][va], "title": inputs["stats"]["title_counts"][va]},
            float(inputs["labels"][tr].mean()), inputs["v5"]["context_config"])
    contextual[matched] = hard[matched]
    p = mix_extra(contextual, inputs["extra"]["extra_text"], None, inputs["v6"]["config"])
    for fold, (_, va) in enumerate(inputs["folds"]):
        p[va], _ = inputs["hierarchy_models"][fold].predict_keys(inputs["hierarchy_keys"][fold], p[va])
    return p


def main(save=True):
    protocol = {"policies": POLICIES, "configurations": len(POLICIES), "baseline": "Frozen v7 C16, URL weight .2",
        "error_title_regex": ERROR_TITLE.pattern, "fit_classifier_count": 0,
        "fixed": "All remaining weights, training-only hard rules/context, extra HTML and hierarchy unchanged",
        "gate": ">=5 fewer errors at frozen and tuned thresholds, higher F1, >=2 improved folds, no fold worsening>2",
        "scope": "Original development OOF; no holdout/added/test rows for selection",
        "limitation": "Exploratory selection on reused development OOF, not an independent performance estimate",
        "source_sha256": sha(Path(__file__))}
    if save:
        OUT.mkdir(exist_ok=True)
        assert not (OUT / "preregistered.json").exists(), "No silent rescreening"
        (OUT / "preregistered.json").write_text(json.dumps(protocol, indent=2)+"\n")
    inputs = training_inputs()
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as text:
        np.testing.assert_array_equal(text["indices"], inputs["indices"])
        text_p = text["probabilities"].copy()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    dev = inputs["indices"]
    np.testing.assert_array_equal(cache["labels"][dev], inputs["labels"])
    length = cache["features"].iloc[dev].html_text_length.to_numpy()
    empty = cache["features"].iloc[dev].html_empty.to_numpy().astype(bool)
    error_title = np.asarray([bool(ERROR_TITLE.search(cache["records"][int(i)]["title"])) for i in dev])
    selectors = {"empty": empty, "short100": length < 100, "error_title": error_title,
                 "short100_or_error": (length < 100) | error_title}
    del cache
    learned = routed_base(inputs, text_p, .2)
    baseline = finish(inputs, learned)
    np.testing.assert_allclose(baseline, inputs["baseline"]["probabilities"], rtol=0, atol=1e-14)
    cut = inputs["v7"]["selected"]["threshold"]
    y = inputs["labels"]
    before = baseline >= cut
    routed = {weight: routed_base(inputs, text_p, weight) for weight in (.35, .5, .7)}
    rows, predictions = [], []
    breakdown = {name: {"rows": int(mask.sum()), "errors": int((mask & (before != y)).sum())}
                 for name, mask in selectors.items()}
    for number, policy in enumerate(POLICIES):
        mask = selectors[policy["selector"]]
        p = finish(inputs, np.where(mask, routed[policy["url_weight"]], learned))
        selected_cut, _ = best_threshold(y, p)
        pred = p >= selected_cut
        frozen = p >= cut
        net = int((before != y).sum() - (pred != y).sum())
        frozen_net = int((before != y).sum() - (frozen != y).sum())
        folds = [{"fold": f, "before_errors": int((before[va] != y[va]).sum()),
                  "after_errors": int((pred[va] != y[va]).sum())} for f, (_, va) in enumerate(inputs["folds"])]
        scores = measure(y, p, selected_cut)
        passes = (net >= 5 and frozen_net >= 5 and scores["f1"] > inputs["v7"]["selected"]["metrics"]["f1"] and
                  sum(r["after_errors"] < r["before_errors"] for r in folds) >= 2 and
                  all(r["after_errors"] <= r["before_errors"]+2 for r in folds))
        rows.append({"number": number, "policy": policy, "threshold": selected_cut, "metrics": scores,
            "metrics_at_baseline_threshold": measure(y, p, cut), "frozen_net": frozen_net,
            "fixed_errors": int(((before != y) & (pred == y)).sum()),
            "new_errors": int(((before == y) & (pred != y)).sum()), "fold_comparison": folds,
            "gate_passed": bool(passes)})
        predictions.append(p)
    eligible = [r for r in rows if r["gate_passed"]]
    winner = max(eligible if eligible else rows, key=lambda r: r["metrics"]["f1"])
    report = {"baseline": inputs["v7"]["selected"]["metrics"], "gate_passed": bool(eligible), "selected": winner,
        "candidates": rows, "selector_breakdown": breakdown, "protocol": protocol,
        "external_test_labels_used": False, "train_cache_sha256": inputs["digest"]}
    if save:
        (OUT / "selection.json").write_text(json.dumps(report, indent=2)+"\n")
        np.savez_compressed(OUT / "development_oof.npz", indices=dev, labels=y,
            probabilities=predictions[winner["number"]], baseline=baseline,
            **{name:mask for name, mask in selectors.items()})
        print(json.dumps({"event": "routing_selected", "gate_passed": bool(eligible), "selected": winner,
            "selector_breakdown": breakdown,
            "all_candidates": [{"number": r["number"], "errors":r["metrics"]["fp"]+r["metrics"]["fn"],
                "frozen_net":r["frozen_net"], "f1":r["metrics"]["f1"]} for r in rows]}), flush=True)
    return report, predictions[winner["number"]]


if __name__ == "__main__":
    main()
