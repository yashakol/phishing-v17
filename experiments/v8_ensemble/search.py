"""Coarse reweighting after replacing HTML text, with original fold-local rules.

No holdout or test inputs are read. The bounded grid is written before labels
and predictions are loaded. Development OOF is reused for model selection.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.v5.select_ensemble import apply_values, memory_values
from experiments.v6.screen_extra import sha
from phishing_url.content_checkpoints import guard, load_component
from phishing_url.evaluation import best_threshold, measure
from phishing_url.soft_context import apply_count_context, fit_context, lookup_counts
from phishing_url.v5_bundle import mix_probabilities
from phishing_url.v6_bundle import mix_extra
from phishing_url.v7_hierarchy import hierarchy_keys


GRID = {"text_weight": [.7, .85], "aux_url_weight": [.1, .2],
        "cat_weight": [.15, .3], "hash_weight": [0., .25]}


def training_inputs():
    v5 = json.loads((ROOT / "reports/v5/selection.json").read_text())["selected"]
    v6 = json.loads((ROOT / "reports/v6/selection.json").read_text())["selected"]
    v7 = json.loads((ROOT / "reports/v7/selection.json").read_text())
    old = np.load(ROOT / "reports/v5/development_oof.npz")
    extra = np.load(ROOT / "reports/v6/development_oof.npz")
    baseline = np.load(ROOT / "reports/v7/development_oof.npz")
    stats = np.load(ROOT / "reports/v5/soft_development_oof.npz")
    indices, labels = old["indices"], old["labels"]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(labels, labels))
    for table in (extra, baseline, stats):
        np.testing.assert_array_equal(table["indices"], indices)
        np.testing.assert_array_equal(table["labels"], labels)
    memory = joblib.load(ROOT / "experiments/v5/development_memory.joblib")
    values = memory_values(memory["keys"], labels, folds)
    for kind in values:
        np.testing.assert_array_equal(values[kind], memory["values"][kind])
    key_cache = joblib.load(ROOT / "experiments/v7_signatures/training_keys.joblib")
    soft = joblib.load(ROOT / "experiments/v5/soft_keys.joblib")
    np.testing.assert_array_equal(soft["indices"], indices)
    digest = sha(ROOT / "data/content_train.joblib")
    assert key_cache["content_sha256"] == memory["data_sha256"] == soft["data_sha256"] == digest
    np.testing.assert_array_equal(np.asarray(key_cache["labels"])[indices], labels)
    hierarchy_config = v7["hierarchy_parameters"]
    hierarchy_models, hierarchy_keys_validation = [], []
    for fold, (tr, va) in enumerate(folds):
        training_keys = {kind: [values[int(i)] for i in tr] for kind, values in soft["keys"].items()}
        validation_keys = {kind: [values[int(i)] for i in va] for kind, values in soft["keys"].items()}
        recomputed = lookup_counts(fit_context(training_keys, labels[tr]), validation_keys)
        for kind in ("domain", "title"):
            np.testing.assert_array_equal(recomputed[kind], stats[kind+"_counts"][va])
        expected = guard(key_cache, indices[tr], {**hierarchy_config,
                         "source_sha256": sha(ROOT / "phishing_url/v7_hierarchy.py")}, digest)
        hierarchy_models.append(load_component(ROOT / f"experiments/v7_hierarchy/fold_{fold}.joblib", expected))
        hierarchy_keys_validation.append([hierarchy_keys(key_cache["urls"][int(i)]) for i in indices[va]])
        assert np.allclose(stats["prior"][va], labels[tr].mean(), rtol=0, atol=1e-15)
    return dict(v5=v5, v6=v6, v7=v7, old=old, extra=extra, baseline=baseline, stats=stats,
                indices=indices, labels=labels, folds=folds, memory=values, digest=digest,
                hierarchy_models=hierarchy_models, hierarchy_keys=hierarchy_keys_validation)


def predict_oof(inputs, text, config):
    old, labels, stats = inputs["old"], inputs["labels"], inputs["stats"]
    base = mix_probabilities(text, old["numeric"], old["url"], None, old["cat"], config, old["hashed"])
    hard = apply_values(base, inputs["memory"])
    matched = hard != base
    contextual = np.empty(len(labels))
    for tr, va in inputs["folds"]:
        contextual[va] = apply_count_context(base[va], {
            "domain": stats["domain_counts"][va], "title": stats["title_counts"][va]},
            float(labels[tr].mean()), inputs["v5"]["context_config"])
    contextual[matched] = hard[matched]
    p = mix_extra(contextual, inputs["extra"]["extra_text"], None, inputs["v6"]["config"])
    for fold, (_, va) in enumerate(inputs["folds"]):
        p[va], _ = inputs["hierarchy_models"][fold].predict_keys(inputs["hierarchy_keys"][fold], p[va])
    return p


def main(text_oof, text_parameters, text_name, save=True):
    output = ROOT / "experiments/v8_ensemble"
    output.mkdir(exist_ok=True)
    protocol = {"grid": GRID, "candidates": 16, "fixed": "v7 train-only rules, soft context and extra HTML weight 0.1; hierarchy unchanged",
                "text_name": text_name, "text_oof": str(text_oof.relative_to(ROOT)),
                "text_parameters": text_parameters,
                "selection_scope": "Original development OOF only; no holdout, added rows or test labels",
                "limitation": "Repeated OOF selection is optimistic; public scalar feedback records a baseline, not row labels."}
    if save:
        (output / f"{text_name}_preregistered.json").write_text(json.dumps(protocol, indent=2) + "\n")
    inputs = training_inputs()
    with np.load(text_oof) as text:
        np.testing.assert_array_equal(text["indices"], inputs["indices"])
        np.testing.assert_array_equal(text["labels"], inputs["labels"])
        text_p = text["probabilities"].copy()
        for fold, (_, va) in enumerate(inputs["folds"]):
            np.testing.assert_array_equal(text["folds"][va], np.full(len(va), fold))
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as original_text:
        replay = predict_oof(inputs, original_text["probabilities"], inputs["v5"]["config"])
    np.testing.assert_allclose(replay, inputs["baseline"]["probabilities"], rtol=0, atol=1e-14)
    baseline_threshold = inputs["v7"]["selected"]["threshold"]
    before = replay >= baseline_threshold
    rows, probabilities = [], {}
    for weights in itertools.product(*GRID.values()):
        config = {**inputs["v5"]["config"], **dict(zip(GRID, weights, strict=True))}
        p = predict_oof(inputs, text_p, config)
        threshold, _ = best_threshold(inputs["labels"], p)
        after = p >= threshold
        row = {"v5_config": config, "threshold": threshold, "metrics": measure(inputs["labels"], p, threshold),
               "fixed_errors": int(np.sum((before != inputs["labels"]) & (after == inputs["labels"]))),
               "new_errors": int(np.sum((before == inputs["labels"]) & (after != inputs["labels"]))),
               "fold_comparison": [{"fold": fold, "baseline_errors": int(np.sum(before[va] != inputs["labels"][va])),
                   "candidate_errors": int(np.sum(after[va] != inputs["labels"][va]))} for fold, (_, va) in enumerate(inputs["folds"])]}
        rows.append(row)
        probabilities[json.dumps(config, sort_keys=True)] = p
    chosen = max(rows, key=lambda row: (row["metrics"]["f1"], row["fixed_errors"]-row["new_errors"],
                                        -sum(abs(row["v5_config"][k]-inputs["v5"]["config"][k]) for k in GRID)))
    p = probabilities[json.dumps(chosen["v5_config"], sort_keys=True)]
    report = {"selected": chosen, "baseline_v7": inputs["v7"]["selected"]["metrics"], "candidates": rows,
              "text_name": text_name, "text_parameters": text_parameters,
              "text_oof": str(text_oof.relative_to(ROOT)), "text_oof_sha256": sha(text_oof),
              "v6_config": inputs["v6"]["config"], "train_cache_sha256": inputs["digest"], "protocol": protocol,
              "source_sha256": sha(Path(__file__)), "external_test_labels_used": False}
    if save:
        (output / f"{text_name}_selection.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(output / f"{text_name}_development_oof.npz", indices=inputs["indices"],
                            labels=inputs["labels"], probabilities=p, baseline=replay)
        print(json.dumps({"event": "v8_ensemble_selected", "text_name": text_name,
                          "selected": chosen, "baseline_v7": report["baseline_v7"]}), flush=True)
    return report, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-oof", type=Path, default=ROOT / "experiments/v7_astra/development_oof.npz")
    parser.add_argument("--text-selection", type=Path, default=ROOT / "experiments/v7_astra/oof_selection.json")
    parser.add_argument("--text-name", default="baseline_c16")
    args = parser.parse_args()
    params = json.loads(args.text_selection.read_text())["parameters"]
    main(args.text_oof.resolve(), params, args.text_name)
