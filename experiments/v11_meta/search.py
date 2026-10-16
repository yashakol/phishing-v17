"""Bounded learned-ensemble screen using original development predictions only."""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.v8_ensemble.search import training_inputs
from experiments.v9_routing.search import routed_base, finish
from experiments.v8_astra_regularization.resume import sha, scope
from phishing_url.content_checkpoints import save_component, load_component
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v11_meta import COMPONENTS, STRUCTURAL, ensemble_frame, LearnedEnsemble

CONFIGURATIONS = [{"kind": "linear", "c": .1}, {"kind": "tree", "depth": 3, "iterations": 400}]
BLENDS = [.25, .5, 1.]


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline ensemble selection")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / ".venv"):
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(("external/", "reports/v2/", "submissions/")) or "reference" in path.name.lower() or \
            "holdout" in rel.lower() or "test" in path.name.lower():
        raise PermissionError("Forbidden model-selection input: "+rel)
    mode, flags = args[1:3]
    writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
               (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
    if writing and not path.is_relative_to(OUT):
        raise PermissionError("Write outside experiment")


def write(name, values):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values, indent=2)+"\n")


def prepare():
    inputs = training_inputs()
    dev, y = inputs["indices"], inputs["labels"]
    with np.load(ROOT / "experiments/v10_margin/development_oof.npz") as svm, \
            np.load(ROOT / "experiments/v7_astra/development_oof.npz") as text, \
            np.load(ROOT / "experiments/v9_routing/development_oof.npz") as route:
        for values in (svm, text, route):
            np.testing.assert_array_equal(values["indices"], dev)
            np.testing.assert_array_equal(values["labels"], y)
        components = {"svm": svm["probabilities"].copy(), "c16": text["probabilities"].copy(),
            **{name: inputs["old"][name].copy() for name in ("numeric", "url", "cat", "hashed")},
            "extra": inputs["extra"]["extra_text"].copy()}
        routed = route["short100_or_error"].copy()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"][dev], y)
    structural = pd.concat([url["features"].iloc[dev].reset_index(drop=True),
                            cache["features"].iloc[dev].reset_index(drop=True)], axis=1)[list(STRUCTURAL)].copy()
    frame = ensemble_frame(components, structural, routed,
        {kind: inputs["stats"][kind+"_counts"] for kind in ("domain", "title")}, inputs["stats"]["prior"])
    raw = np.where(routed, routed_base(inputs, components["svm"], .5),
                   routed_base(inputs, components["svm"], inputs["v5"]["config"]["aux_url_weight"]))
    baseline = finish(inputs, raw)
    with np.load(ROOT / "experiments/v10_margin/selected_oof.npz") as old:
        np.testing.assert_allclose(old["probabilities"], baseline, rtol=0, atol=1e-14)
    cache_small = {"urls": cache["urls"], "labels": cache["labels"]}
    del cache, url
    gc.collect()
    return inputs, cache_small, frame, raw, baseline


def main():
    sys.addaudithook(audit)
    assert not (OUT / "preregistered.json").exists(), "No silent rescreening"
    protocol = {"configurations": CONFIGURATIONS, "blends": BLENDS, "classifier_fits": 6,
        "splitter": {"n_splits": 3, "shuffle": True, "random_state": 2026},
        "baseline": "Frozen v10, including original train-only memory, context, extra HTML and hierarchy",
        "gate": "At least 8 fewer errors at tuned and frozen threshold, higher F1, >=2 improved folds, no fold worsens by more than 3",
        "scope": "Original development only, 50977 rows; no holdout, added rows or test",
        "limitation": "Reused development. Meta cross-fitting uses already cross-fitted base components; base models for meta-training rows can contain meta-validation rows. This is not fully nested CV or an independent performance estimate",
        "source_sha256": sha(Path(__file__)), "model_source_sha256": sha(ROOT / "phishing_url/v11_meta.py"),
        "structure": list(STRUCTURAL), "components": list(COMPONENTS)}
    write("preregistered.json", protocol)
    started = time.perf_counter()
    inputs, cache, X, raw, baseline = prepare()
    indices, y = inputs["indices"], inputs["labels"]
    v10 = json.loads((ROOT / "reports/v10/selection.json").read_text())
    cut = v10["selected"]["threshold"]
    before = baseline >= cut
    assert measure(y, baseline, cut) == v10["selected"]["metrics"]
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    candidates, probabilities = [], []
    for number, config in enumerate(CONFIGURATIONS):
        p = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(inputs["folds"]):
            assert not set(indices[tr]) & set(indices[va])
            model = LearnedEnsemble(**config).fit(X.iloc[tr], y[tr])
            p[va], fold_ids[va] = model.predict_proba(X.iloc[va]), fold
            expected = scope(cache, indices[tr], {"parameters": model.get_params(),
                "source_sha256": protocol["model_source_sha256"], "frame_columns": list(X.columns),
                "protocol_sha256": sha(OUT / "preregistered.json")}, inputs["digest"])
            target = OUT / f"models/candidate_{number}_fold_{fold}.joblib"
            save_component(target, model, expected)
            restored = load_component(target, expected)
            np.testing.assert_array_equal(p[va[:32]], restored.predict_proba(X.iloc[va[:32]]))
            print(json.dumps({"event": "meta_fold_complete", "candidate": number, "fold": fold}), flush=True)
        assert np.isfinite(p).all()
        np.savez_compressed(OUT / f"candidate_{number}.npz", indices=indices, labels=y, probabilities=p, folds=fold_ids)
        for weight in BLENDS:
            values = finish(inputs, (1-weight)*raw+weight*p)
            threshold, _ = best_threshold(y, values)
            after, frozen = values >= threshold, values >= cut
            net, fixed_net = int((before != y).sum()-(after != y).sum()), int((before != y).sum()-(frozen != y).sum())
            folds = [{"fold": f, "before_errors": int((before[va] != y[va]).sum()),
                      "after_errors": int((after[va] != y[va]).sum())} for f, (_, va) in enumerate(inputs["folds"])]
            metrics = measure(y, values, threshold)
            passed = (net >= 8 and fixed_net >= 8 and metrics["f1"] > v10["selected"]["metrics"]["f1"] and
                sum(r["after_errors"] < r["before_errors"] for r in folds) >= 2 and
                all(r["after_errors"] <= r["before_errors"]+3 for r in folds))
            candidates.append({"candidate": number, "parameters": LearnedEnsemble(**config).get_params(),
                "weight": weight, "threshold": threshold, "metrics": metrics,
                "metrics_at_baseline_threshold": measure(y, values, cut), "net_errors": net, "frozen_net": fixed_net,
                "fixed_errors": int(((before != y) & (after == y)).sum()),
                "new_errors": int(((before == y) & (after != y)).sum()), "fold_comparison": folds,
                "gate_passed": bool(passed)})
            probabilities.append(values)
    eligible = [r for r in candidates if r["gate_passed"]]
    chosen = max(eligible if eligible else candidates, key=lambda r: (r["metrics"]["f1"], r["net_errors"], -r["weight"]))
    selected = candidates.index(chosen)
    report = {"gate_passed": bool(eligible), "selected": chosen, "baseline_v10": v10["selected"]["metrics"],
        "candidates": candidates, "train_cache_sha256": inputs["digest"], "frame_columns": list(X.columns),
        "protocol": protocol, "seconds": time.perf_counter()-started, "classifier_fits": 6,
        "external_test_labels_used": False}
    write("selection.json", report)
    np.savez_compressed(OUT / "selected_oof.npz", indices=indices, labels=y, probabilities=probabilities[selected], baseline=baseline)
    np.savez_compressed(OUT / "training_frame.npz", indices=indices, labels=y, values=X.to_numpy(), columns=np.asarray(X.columns))
    print(json.dumps({"event": "meta_selection_complete", "gate_passed": bool(eligible), "selected": chosen,
        "all_candidates": [{"kind": r["parameters"]["kind"], "weight": r["weight"],
            "errors": r["metrics"]["fp"]+r["metrics"]["fn"], "frozen_net": r["frozen_net"]} for r in candidates]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
