"""Screen only on development train/tune; report once on a separate dev audit set."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import gc
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.data import load_train, sha256
from phishing_url.features import group_for_url
from phishing_url.evaluation import measure, best_threshold
from url_model import URLFeatures, SparseClassifier, URLModel


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    start = time.perf_counter()
    urls, labels, _ = load_train(ROOT / "data/train.csv")
    dev = np.load(ROOT / "reports/split_indices.npz")["development"]
    urls = np.asarray(urls, dtype=object)[dev]
    labels = labels[dev]
    outer_train, audit = train_test_split(np.arange(len(dev)), test_size=0.20, stratify=labels, random_state=20261006)
    fit, tune = train_test_split(outer_train, test_size=0.15, stratify=labels[outer_train], random_state=18428)
    np.savez_compressed(OUT / "internal_split.npz", train_indices=dev[fit], tune_indices=dev[tune],
                        audit_indices=dev[audit], outer_train_indices=dev[outer_train], development_indices=dev)
    print(json.dumps({"event": "split", "train": len(fit), "tune": len(tune), "audit": len(audit)}), flush=True)
    feature_start = time.perf_counter()
    features = URLFeatures()
    train_x = features.matrices(urls[fit].tolist(), fit=True)
    tune_x = features.matrices(urls[tune].tolist())
    feature_seconds = time.perf_counter() - feature_start
    print(json.dumps({"event": "features_ready", "seconds": feature_seconds,
                      "dimensions": {k: v.shape[1] for k, v in train_x.items()},
                      "nnz": {k: v.nnz for k, v in train_x.items()}}), flush=True)
    configs = [
        {"name": "baseline", "representation": "base", "c": 4, "nb": False},
        {"name": "base_c16", "representation": "base", "c": 16, "nb": False},
        {"name": "base_nb4", "representation": "base", "c": 4, "nb": True},
        {"name": "rich_c4", "representation": "rich", "c": 4, "nb": False},
        {"name": "rich_c16", "representation": "rich", "c": 16, "nb": False},
        {"name": "rich_nb4", "representation": "rich", "c": 4, "nb": True},
    ]
    rows, probabilities = [], {}
    best_model, best_row, baseline_model = None, None, None
    for cfg in configs:
        model = SparseClassifier(c=cfg["c"], nb=cfg["nb"]).fit(train_x[cfg["representation"]], labels[fit])
        p = model.predict_proba(tune_x[cfg["representation"]])
        threshold, _ = best_threshold(labels[tune], p)
        row = {"config": cfg, "fit_seconds": model.fit_seconds,
               "iterations": model.model.n_iter_.tolist(), "tune_at_05": measure(labels[tune], p),
               "tune_selected_threshold": measure(labels[tune], p, threshold)}
        rows.append(row)
        probabilities[cfg["name"]] = p
        if cfg["name"] == "baseline":
            baseline_model = model
        if best_row is None or row["tune_selected_threshold"]["f1"] > best_row["tune_selected_threshold"]["f1"]:
            best_model, best_row = model, row
        print(json.dumps({"event": "candidate", **row}), flush=True)
        write_json("screening_progress.json", rows)
    np.savez_compressed(OUT / "tune_predictions.npz", indices=dev[tune], y=labels[tune], **probabilities)
    selected = dict(best_row["config"])
    selected["threshold"] = best_row["tune_selected_threshold"]["threshold"]
    base_threshold = rows[0]["tune_selected_threshold"]["threshold"]
    blend_candidates = []
    for weight in (0, 0.25, 0.5, 0.75, 1):
        p = weight * probabilities[selected["name"]] + (1 - weight) * probabilities["baseline"]
        threshold, f1 = best_threshold(labels[tune], p)
        blend_candidates.append({"weight_selected": weight, "threshold": threshold, "tune_f1": f1})
    blend = max(blend_candidates, key=lambda x: x["tune_f1"])
    # Freeze selection before audit predictions/metrics are computed.
    write_json("selection.json", {"selected": selected, "baseline_threshold": base_threshold,
                                  "blend": blend, "blend_candidates": blend_candidates})
    del train_x, tune_x
    gc.collect()
    refit_start = time.perf_counter()
    features = URLFeatures()
    outer_x = features.matrices(urls[outer_train].tolist(), fit=True)
    baseline_model = SparseClassifier(c=4, nb=False).fit(outer_x["base"], labels[outer_train])
    best_model = SparseClassifier(c=selected["c"], nb=selected["nb"]).fit(
        outer_x[selected["representation"]], labels[outer_train])
    refit_seconds = time.perf_counter() - refit_start
    print(json.dumps({"event": "outer_refit", "seconds": refit_seconds, "rows": len(outer_train)}), flush=True)
    del outer_x
    gc.collect()
    prediction_start = time.perf_counter()
    audit_x = features.matrices(urls[audit].tolist())
    base_p = baseline_model.predict_proba(audit_x["base"])
    p = best_model.predict_proba(audit_x[selected["representation"]])
    blend_p = blend["weight_selected"] * p + (1 - blend["weight_selected"]) * base_p
    predict_seconds = time.perf_counter() - prediction_start
    seen = set(group_for_url(u) for u in urls[outer_train])
    unseen = np.asarray([group_for_url(u) not in seen for u in urls[audit]])
    evaluation = {}
    for name, prob, threshold in [("baseline", base_p, base_threshold),
                                  ("selected", p, selected["threshold"]),
                                  ("blend", blend_p, blend["threshold"])]:
        evaluation[name] = {"all": measure(labels[audit], prob, threshold),
                            "all_at_05": measure(labels[audit], prob),
                            "unseen_domain": measure(labels[audit][unseen], prob[unseen], threshold)}
    base_errors = (base_p >= base_threshold) != labels[audit]
    selected_errors = (p >= selected["threshold"]) != labels[audit]
    overlap = {"baseline_wrong_selected_correct": int((base_errors & ~selected_errors).sum()),
               "baseline_correct_selected_wrong": int((~base_errors & selected_errors).sum()),
               "both_wrong": int((base_errors & selected_errors).sum())}
    np.savez_compressed(OUT / "audit_predictions.npz", indices=dev[audit], y=labels[audit],
                        baseline=base_p, selected=p, blend=blend_p, unseen_domain=unseen,
                        parameters=json.dumps(selected), train_indices=dev[outer_train], tune_indices=dev[tune])
    bundle = URLModel(selected["representation"], selected["c"], selected["nb"])
    bundle.features, bundle.classifier = features, best_model
    joblib.dump(bundle, OUT / "internal_selected.joblib", compress=3)
    result = {"data_sha256": sha256(ROOT / "data/train.csv"),
              "split_sha256": sha256(ROOT / "reports/split_indices.npz"),
              "scope": "Only development labels used; original holdout never evaluated.",
              "seeds": [20261006, 18428], "screen_train_rows": len(fit), "tune_rows": len(tune),
              "train_rows": len(outer_train), "outer_refit_seconds": refit_seconds,
              "audit_rows": len(audit), "feature_seconds": feature_seconds,
              "audit_predict_seconds": predict_seconds, "candidates": rows,
              "selected": selected, "blend": blend, "audit": evaluation, "error_overlap": overlap,
              "total_seconds": time.perf_counter() - start, "threadpools": threadpool_info()}
    write_json("results.json", result)
    print(json.dumps({"event": "finished", **result}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
