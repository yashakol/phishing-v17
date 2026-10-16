"""One bounded extra-static feature specialist; no diagnostic or test input."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

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

from phishing_url.content_bundle import prediction_frame
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v10_training import sha
from phishing_url.v12_expert import blend_expert, eligibility, make_expert
from phishing_url.v13_features import extend_frame

WEIGHTS = [.25, .5, .75]


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline development expert selection")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / ".venv"):
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(("external/", "reports/v2/", "submissions/")) or "reference" in path.name.lower() or \
            "holdout" in rel.lower() or "test" in path.name.lower():
        raise PermissionError("Forbidden expert-selection input: "+rel)
    mode, flags = args[1:3]
    writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
               (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
    if writing and not path.is_relative_to(OUT):
        raise PermissionError("Expert-selection write outside own experiment")


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2)+"\n")


def main():
    sys.addaudithook(audit)
    assert not (OUT / "preregistered.json").exists(), "No silent rescreening"
    protocol = {"classifier": make_expert().get_params(), "weights": WEIGHTS, "classifier_fits": 3,
        "splitter": {"n_splits": 3, "shuffle": True, "random_state": 2026},
        "eligibility": "Frozen v10 final probability between .05 and .95 inclusive; hard endpoints and other confident outputs unchanged",
        "fit_scope": "Only eligible training rows of each original development fold; no diagnostic, added rows or test",
        "numeric_schema": "Frozen v12 frame plus numeric/bool columns from the existing static JS/CSS/resource cache; fixed order, categorical strings excluded",
        "gate": "At least 8 fewer errors at tuned and frozen v10 threshold, higher F1, >=2 improved folds, no fold worse by >3",
        "limitation": "Reused development, not fully nested meta cross-fitting. Base OOF models for specialist-training rows can have seen specialist-validation rows; not independent evaluation.",
        "source_sha256": sha(Path(__file__)), "expert_source_sha256": sha(ROOT / "phishing_url/v13_features.py"),
        "base_expert_source_sha256": sha(ROOT / "phishing_url/v12_expert.py"),
        "input_sha256": {p: sha(ROOT / p) for p in ("data/content_train.joblib", "data/content_url_train.joblib", "data/extra_html_train.joblib",
            "experiments/v12_expert/training_frame.npz", "experiments/v12_expert/selection.json", "experiments/v10_margin/selected_oof.npz")}}
    write("preregistered.json", protocol)
    started = time.perf_counter()
    v10 = json.loads((ROOT / "reports/v10/selection.json").read_text())
    columns = json.loads((ROOT / "experiments/v12_expert/selection.json").read_text())["feature_columns"]
    with np.load(ROOT / "experiments/v12_expert/training_frame.npz", allow_pickle=False) as data:
        dev, y = data["indices"].copy(), data["labels"].copy()
        meta = pd.DataFrame(data["values"], columns=columns)
    with np.load(ROOT / "experiments/v10_margin/selected_oof.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["indices"], dev)
        np.testing.assert_array_equal(data["labels"], y)
        baseline = data["probabilities"].copy()
    cache = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == url["urls"]
    np.testing.assert_array_equal(aligned_splits(ROOT, cache)["development"], dev)
    np.testing.assert_array_equal(cache["labels"][dev], y)
    numeric = cache["features"].iloc[dev].reset_index(drop=True)
    X = extend_frame(meta, numeric)
    np.testing.assert_array_equal(meta.baseline_probability.to_numpy(), baseline)
    cache_small = {"urls": cache["urls"], "labels": cache["labels"]}
    del cache, url, numeric, meta
    gc.collect()
    old_cut = v10["selected"]["threshold"]
    assert measure(y, baseline, old_cut) == v10["selected"]["metrics"]
    eligible = eligibility(baseline)
    p = baseline.copy()
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    folds = list(StratifiedKFold(**protocol["splitter"]).split(np.zeros(len(y)), y))
    for fold, (tr, va) in enumerate(folds):
        fit, predict = tr[eligible[tr]], va[eligible[va]]
        assert not set(dev[fit]) & set(dev[va])
        print(json.dumps({"event": "expert_fold_started", "fold": fold, "fit_rows": len(fit), "eligible_validation_rows": len(predict)}), flush=True)
        model = make_expert().fit(X.iloc[fit], y[fit])
        p[predict] = model.predict_proba(X.iloc[predict], thread_count=2)[:, 1]
        fold_ids[va] = fold
        config = {"parameters": model.get_params(), "source_sha256": protocol["expert_source_sha256"],
                  "base_expert_source_sha256": protocol["base_expert_source_sha256"],
                  "protocol_sha256": sha(OUT / "preregistered.json"), "frame_columns": list(X.columns)}
        expected = guard(cache_small, dev[fit], config, protocol["input_sha256"]["data/content_train.joblib"])
        expected["ordered_training_indices_sha256"] = __import__("hashlib").sha256(dev[fit].tobytes()).hexdigest()
        path = OUT / f"models/fold_{fold}.joblib"
        save_component(path, model, expected)
        restored = load_component(path, expected)
        np.testing.assert_array_equal(p[predict[:32]], restored.predict_proba(X.iloc[predict[:32]], thread_count=2)[:, 1])
        print(json.dumps({"event": "expert_fold_complete", "fold": fold}), flush=True)
    before = baseline >= old_cut
    candidates, values = [], []
    for weight in WEIGHTS:
        probability = blend_expert(baseline, p, weight)
        cut, _ = best_threshold(y, probability)
        after, frozen = probability >= cut, probability >= old_cut
        metrics = measure(y, probability, cut)
        net, fixed_net = int((before != y).sum()-(after != y).sum()), int((before != y).sum()-(frozen != y).sum())
        fold_metrics = [{"fold": f, "before_errors": int((before[va] != y[va]).sum()),
            "after_errors": int((after[va] != y[va]).sum())} for f, (_, va) in enumerate(folds)]
        passed = net >= 8 and fixed_net >= 8 and metrics["f1"] > v10["selected"]["metrics"]["f1"] and \
            sum(f["after_errors"] < f["before_errors"] for f in fold_metrics) >= 2 and \
            all(f["after_errors"] <= f["before_errors"]+3 for f in fold_metrics)
        candidates.append({"weight": weight, "threshold": cut, "metrics": metrics,
            "metrics_at_baseline_threshold": measure(y, probability, old_cut), "net_errors": net,
            "frozen_net": fixed_net, "fixed_errors": int(((before != y) & (after == y)).sum()),
            "new_errors": int(((before == y) & (after != y)).sum()), "fold_comparison": fold_metrics,
            "gate_passed": bool(passed)})
        values.append(probability)
    passed = [c for c in candidates if c["gate_passed"]]
    chosen = max(passed if passed else candidates, key=lambda c: (c["metrics"]["f1"], c["net_errors"], -c["weight"]))
    report = {"gate_passed": bool(passed), "selected": chosen, "candidates": candidates,
        "baseline_v10": v10["selected"]["metrics"], "protocol": protocol, "feature_columns": list(X.columns),
        "numeric_columns": [c.removeprefix("numeric_") for c in X.columns if c.startswith("numeric_")],
        "train_cache_sha256": protocol["input_sha256"]["data/content_train.joblib"],
        "eligible_rows": int(eligible.sum()), "classifier_fits": 3,
        "seconds": time.perf_counter()-started, "external_test_labels_used": False}
    write("selection.json", report)
    np.savez_compressed(OUT / "development_expert.npz", indices=dev, labels=y, probabilities=p, folds=fold_ids, eligibility=eligible)
    np.savez_compressed(OUT / "selected_oof.npz", indices=dev, labels=y, probabilities=values[candidates.index(chosen)], baseline=baseline, folds=fold_ids)
    np.savez_compressed(OUT / "training_frame.npz", indices=dev, labels=y, values=X.to_numpy(), columns=np.asarray(X.columns, dtype=str), eligibility=eligible)
    print(json.dumps({"event": "expert_selection_complete", "gate_passed": bool(passed), "selected": chosen,
        "eligible_rows": int(eligible.sum()), "features": X.shape[1]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
