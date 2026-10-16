"""Integrate new fold-local text into the unchanged supervised v6 mixture.

Only a coarse replacement grid and an inner-selected URL context policy are
considered. All original fold training rules and context are reconstructed.
No test/holdout labels or published reference data are loaded.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.v5.select_ensemble import apply_values, memory_values
from experiments.v6.screen_extra import sha
from phishing_url.content_checkpoints import guard, load_component
from phishing_url.evaluation import best_threshold, measure
from phishing_url.soft_context import apply_count_context
from phishing_url.v5_bundle import mix_probabilities
from phishing_url.v6_bundle import mix_extra
from phishing_url.v7_hierarchy import hierarchy_keys


def main(text_oof, text_selection, save=True):
    out = ROOT / "reports/v7"
    out.mkdir(exist_ok=True)
    old = np.load(ROOT / "reports/v5/development_oof.npz")
    original = np.load(ROOT / "reports/v6/development_oof.npz")
    text = np.load(text_oof)
    stats = np.load(ROOT / "reports/v5/soft_development_oof.npz")
    indices, y = old["indices"], old["labels"]
    for table in (original, text, stats):
        np.testing.assert_array_equal(table["indices"], indices)
        np.testing.assert_array_equal(table["labels"], y)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(y, y))
    for fold, (_, va) in enumerate(folds):
        np.testing.assert_array_equal(text["folds"][va], np.full(len(va), fold))
    source = json.loads(Path(text_selection).read_text())
    digest = sha(ROOT / "data/content_train.joblib")
    assert source["data_sha256"] == digest
    memory = joblib.load(ROOT / "experiments/v5/development_memory.joblib")
    np.testing.assert_array_equal(memory["indices"], indices)
    assert memory["data_sha256"] == digest
    # Recompute fold counts from labels rather than trusting stored evidence.
    values = memory_values(memory["keys"], y, folds)
    for kind in values:
        np.testing.assert_array_equal(values[kind], memory["values"][kind])
    v5 = json.loads((ROOT / "reports/v5/selection.json").read_text())["selected"]
    v6 = json.loads((ROOT / "reports/v6/selection.json").read_text())["selected"]
    key_cache = joblib.load(ROOT / "experiments/v7_signatures/training_keys.joblib")
    assert key_cache["content_sha256"] == digest
    np.testing.assert_array_equal(np.asarray(key_cache["labels"])[indices], y)
    hierarchy_rows = json.loads((ROOT / "experiments/v7_hierarchy/selection.json").read_text())
    hierarchy_config = hierarchy_rows["selected"]["config"]
    hierarchy_enabled = hierarchy_rows["useful"]
    hierarchy_models = []
    for fold, (tr, _) in enumerate(folds):
        expected = guard(key_cache, indices[tr], {**hierarchy_config,
                         "source_sha256": sha(ROOT / "phishing_url/v7_hierarchy.py")}, digest)
        hierarchy_models.append(load_component(ROOT / f"experiments/v7_hierarchy/fold_{fold}.joblib", expected))
    hierarchy_validation_keys = [[hierarchy_keys(key_cache["urls"][int(i)]) for i in indices[va]] for _, va in folds]
    candidates, probabilities = [], {}
    before = original["probabilities"] >= v6["threshold"]
    for replacement in (0., .25, .5, 1.):
        combined_text = (1-replacement) * old["text"] + replacement * text["probabilities"]
        base = mix_probabilities(combined_text, old["numeric"], old["url"], None, old["cat"],
                                 v5["config"], old["hashed"])
        hard = apply_values(base, values)
        matched = hard != base
        contextual = np.empty(len(y))
        for tr, va in folds:
            assert np.allclose(stats["prior"][va], float(y[tr].mean()), atol=1e-15, rtol=0)
            contextual[va] = apply_count_context(base[va], {"domain": stats["domain_counts"][va],
                "title": stats["title_counts"][va]}, float(y[tr].mean()), v5["context_config"])
        contextual[matched] = hard[matched]
        p = mix_extra(contextual, original["extra_text"], None, v6["config"])
        if replacement == 0:
            np.testing.assert_allclose(p, original["probabilities"], rtol=0, atol=1e-14)
        for use_hierarchy in (False, True) if hierarchy_enabled else (False,):
            final = p.copy()
            hits = 0
            if use_hierarchy:
                for fold, (_, va) in enumerate(folds):
                    final[va], matched = hierarchy_models[fold].predict_keys(hierarchy_validation_keys[fold], p[va])
                    hits += int(matched.sum())
            config = {"text_replacement": replacement, "hierarchy": use_hierarchy}
            threshold, _ = best_threshold(y, final)
            after = final >= threshold
            row = {"config": config, "threshold": threshold, "metrics": measure(y, final, threshold),
                   "metrics_at_v6_threshold": measure(y, final, v6["threshold"]), "hierarchy_matches": hits,
                   "fixed_errors": int(np.sum((before != y) & (after == y))),
                   "new_errors": int(np.sum((before == y) & (after != y))),
                   "fold_comparison": [{"fold": f, "baseline_errors": int(np.sum(before[va] != y[va])),
                       "candidate_errors": int(np.sum(after[va] != y[va])),
                       "baseline_f1": measure(y[va], original["probabilities"][va], v6["threshold"])["f1"],
                       "candidate_f1": measure(y[va], final[va], threshold)["f1"]} for f, (_, va) in enumerate(folds)]}
            candidates.append(row)
            probabilities[json.dumps(config, sort_keys=True)] = final
    selected = max(candidates, key=lambda r: (r["metrics"]["f1"], r["fixed_errors"]-r["new_errors"],
                                              -r["config"]["text_replacement"], not r["config"]["hierarchy"]))
    report = {"selected": selected, "baseline_v6": measure(y, original["probabilities"], v6["threshold"]),
              "candidates": candidates, "text_parameters": source["parameters"], "hierarchy_parameters": hierarchy_config,
              "text_oof": str(Path(text_oof).relative_to(ROOT)), "text_selection": str(Path(text_selection).relative_to(ROOT)),
              "train_cache_sha256": digest, "text_oof_sha256": sha(text_oof), "text_selection_sha256": sha(text_selection),
              "protocol": "Eight coarse supervised mixtures on original3fold OOF. Original fold rules replay-verified. All candidate fitting uses train-only labels; weights/threshold selected on OOF before holdout.",
              "limitation": "OOF is used for selection and is optimistic; existing holdout is reused diagnostic evidence.",
              "external_test_labels_used": False, "new_astra_calls": 1}
    final = probabilities[json.dumps(selected["config"], sort_keys=True)]
    if save:
        (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(out / "development_oof.npz", indices=indices, labels=y,
            probabilities=final, original=original["probabilities"], new_text=text["probabilities"], folds=text["folds"])
        print(json.dumps({"event": "v7_selected", "selected": selected, "baseline_v6": report["baseline_v6"]}), flush=True)
    return report, final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-oof", type=Path, default=ROOT / "experiments/v7_astra/development_oof.npz")
    parser.add_argument("--text-selection", type=Path, default=ROOT / "experiments/v7_astra/oof_selection.json")
    args = parser.parse_args()
    main(args.text_oof.resolve(), args.text_selection.resolve())
