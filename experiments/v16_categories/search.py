"""Two frozen specialist families screened only on own development folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import hashlib
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

from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v10_training import sha
from phishing_url.v12_expert import eligibility
from phishing_url.v15_resources import ResourceContext, add_resource_evidence
from phishing_url.v16_expert import CATEGORY_KEYS, KINDS, category_frame, make_specialist, mix_specialist, specialist_frame

WEIGHTS = [.25, .5, .75]


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        raise PermissionError("Offline specialist development")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / ".venv"):
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(("external/", "reports/v2/", "submissions/")) or "reference" in path.name.lower() or \
            "holdout" in rel.lower() or "test" in path.name.lower():
        raise PermissionError("Forbidden development input: "+rel)
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
              (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing and not path.is_relative_to(OUT):
        raise PermissionError("Development write outside own experiment")


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2)+"\n")


def main():
    sys.addaudithook(audit)
    assert not (OUT / "preregistered.json").exists(), "No silent rescreening"
    inputs = ("data/content_train.joblib", "data/content_url_train.joblib", "data/extra_html_train.joblib",
              "experiments/v13_static/training_frame.npz", "experiments/v13_static/selection.json",
              "experiments/v15_resources/selected_oof.npz", "reports/v15/selection.json")
    protocol = {"parameters": {kind: make_specialist(kind).get_params() for kind in KINDS},
        "weights": WEIGHTS, "classifier_fits": 6,
        "splitter": {"n_splits": 3, "shuffle": True, "random_state": 2026},
        "eligibility": "Frozen v10 final probability in [.05,.95]; v15 preserved outside mask",
        "fit_scope": "Only eligible original development-fold train rows; resource context on whole fold train; training row resource counts leave its own label out",
        "gate": "At least 8 fewer errors vs v15 at tuned AND frozen v15 threshold; higher F1; >=2 improved folds; no fold worse by >3",
        "final_training_policy": "After both gates, refit same selected specialist on all uncertain own-train rows using dev OOF features plus holdout/added features from base/context fitted on development only. No test/reference labels; full own-train resource counters",
        "diagnostic_export_gate": "At least 3 fewer diagnostic errors AND higher F1 vs frozen v15; no parameter adaptation",
        "source_sha256": sha(Path(__file__)), "expert_source_sha256": sha(ROOT / "phishing_url/v16_expert.py"),
        "resource_source_sha256": sha(ROOT / "phishing_url/v15_resources.py"),
        "inputs_sha256": {name: sha(ROOT/name) for name in inputs},
        "limitation": "Reused development/diagnostic, base OOF not fully nested inside specialist CV; not independent performance estimation"}
    write("preregistered.json", protocol)
    started = time.perf_counter()
    selection = json.loads((ROOT / "reports/v15/selection.json").read_text())
    columns = json.loads((ROOT / "experiments/v13_static/selection.json").read_text())["feature_columns"]
    with np.load(ROOT / "experiments/v13_static/training_frame.npz", allow_pickle=False) as stored:
        dev, y = stored["indices"].copy(), stored["labels"].copy()
        static = pd.DataFrame(stored["values"], columns=columns)
    with np.load(ROOT / "experiments/v15_resources/selected_oof.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["indices"], dev)
        np.testing.assert_array_equal(stored["labels"], y)
        v10, v15 = stored["baseline"].copy(), stored["probabilities"].copy()
        saved_folds = stored["folds"].copy()
    np.testing.assert_array_equal(v10, static.baseline_probability.to_numpy())
    baseline_threshold = selection["selected"]["threshold"]
    assert measure(y, v15, baseline_threshold) == selection["selected"]["metrics"]
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(aligned_splits(ROOT, cache)["development"], dev)
    np.testing.assert_array_equal(cache["labels"][dev], y)
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    categories = category_frame(cache, extra, url["features"]).iloc[dev].reset_index(drop=True)
    resource_features = extra["features"].iloc[dev].reset_index(drop=True)
    small = {"urls": cache["urls"], "labels": cache["labels"]}
    joblib.dump({"features": categories, "indices": dev, "labels": y,
                 "source_sha256": protocol["expert_source_sha256"]}, OUT / "category_inputs.joblib", compress=3)
    del cache, extra, url
    gc.collect()
    mask = eligibility(v10)
    predictions = {kind: v15.copy() for kind in KINDS}
    folds = list(StratifiedKFold(**protocol["splitter"]).split(np.zeros(len(y)), y))
    for fold, (tr, va) in enumerate(folds):
        np.testing.assert_array_equal(saved_folds[va], np.full(len(va), fold))
        fit, query = tr[mask[tr]], va[mask[va]]
        context = ResourceContext().fit(resource_features.iloc[tr], y[tr])
        training = add_resource_evidence(static.iloc[fit], resource_features.iloc[fit], context, y[fit])
        validation = add_resource_evidence(static.iloc[query], resource_features.iloc[query], context)
        for kind in KINDS:
            print(json.dumps({"event": "specialist_fold_started", "fold": fold, "kind": kind,
                "training_rows": len(fit), "query_rows": len(query)}), flush=True)
            train_frame = specialist_frame(training, categories.iloc[fit], kind)
            query_frame = specialist_frame(validation, categories.iloc[query], kind)
            model = make_specialist(kind).fit(train_frame, y[fit])
            q = model.predict_proba(query_frame, thread_count=2)[:, 1]
            predictions[kind][query] = q
            config = {"parameters": model.get_params(), "source_sha256": protocol["expert_source_sha256"],
                      "protocol_sha256": sha(OUT / "preregistered.json"), "kind": kind,
                      "feature_columns": list(train_frame.columns),
                      "context_guard": guard(small, dev[tr], {"resource_source_sha256": protocol["resource_source_sha256"]},
                                             protocol["inputs_sha256"]["data/content_train.joblib"])}
            scope = guard(small, dev[fit], config, protocol["inputs_sha256"]["data/content_train.joblib"])
            scope["ordered_training_indices_sha256"] = hashlib.sha256(dev[fit].tobytes()).hexdigest()
            path = OUT / f"models/{kind}_fold_{fold}.joblib"
            save_component(path, {"classifier": model, "context": context}, scope)
            restored = load_component(path, scope)
            np.testing.assert_array_equal(q[:32], restored["classifier"].predict_proba(query_frame.iloc[:32], thread_count=2)[:, 1])
            print(json.dumps({"event": "specialist_fold_complete", "fold": fold, "kind": kind}), flush=True)
            del model, restored, train_frame, query_frame
            gc.collect()
    before = v15 >= baseline_threshold
    candidates, mixtures = [], []
    for kind in KINDS:
        for weight in WEIGHTS:
            p = mix_specialist(v10, v15, predictions[kind], weight)
            threshold, _ = best_threshold(y, p)
            after, frozen = p >= threshold, p >= baseline_threshold
            metrics = measure(y, p, threshold)
            net, frozen_net = int((before != y).sum()-(after != y).sum()), int((before != y).sum()-(frozen != y).sum())
            comparisons = [{"fold": fold, "before_errors": int((before[va] != y[va]).sum()),
                            "after_errors": int((after[va] != y[va]).sum())} for fold, (_, va) in enumerate(folds)]
            passed = net >= 8 and frozen_net >= 8 and metrics["f1"] > selection["selected"]["metrics"]["f1"] and \
                     sum(c["after_errors"] < c["before_errors"] for c in comparisons) >= 2 and \
                     all(c["after_errors"] <= c["before_errors"]+3 for c in comparisons)
            candidates.append({"kind": kind, "weight": weight, "threshold": threshold, "metrics": metrics,
                "metrics_at_baseline_threshold": measure(y, p, baseline_threshold), "net_errors": net, "frozen_net": frozen_net,
                "fixed_errors": int(((before != y) & (after == y)).sum()), "new_errors": int(((before == y) & (after != y)).sum()),
                "fold_comparison": comparisons, "gate_passed": bool(passed)})
            mixtures.append(p)
    passed = [c for c in candidates if c["gate_passed"]]
    chosen = max(passed or candidates, key=lambda c: (c["metrics"]["f1"], c["net_errors"], -c["weight"]))
    feature_columns = list(static.columns) + list(context.transform(resource_features.iloc[:1]).columns)
    if chosen["kind"] == "categorical":
        feature_columns += list(CATEGORY_KEYS)
    report = {"gate_passed": bool(passed), "selected": chosen, "candidates": candidates,
        "baseline_v15": selection["selected"]["metrics"], "feature_columns": feature_columns,
        "eligible_rows": int(mask.sum()), "classifier_fits": protocol["classifier_fits"],
        "protocol": protocol, "seconds": time.perf_counter()-started, "external_test_labels_used": False}
    write("selection.json", report)
    np.savez_compressed(OUT / "development_experts.npz", indices=dev, labels=y, folds=saved_folds,
                        numeric_deep=predictions["numeric_deep"], categorical=predictions["categorical"], eligibility=mask)
    np.savez_compressed(OUT / "selected_oof.npz", indices=dev, labels=y, folds=saved_folds, v10=v10, baseline=v15,
                        probabilities=mixtures[candidates.index(chosen)])
    print(json.dumps({"event": "specialist_selection_complete", "gate_passed": bool(passed), "selected": chosen}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
