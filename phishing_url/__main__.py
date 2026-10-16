"""Reproducible training, validation, inference and submission verification."""

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from .data import load_test, load_train, sha256, write_submission
from .evaluation import best_threshold, choose_ensemble, measure
from .features import group_for_url, make_features
from .models import TextModel, boost_probability, fit_boost

LOG_PATH = None


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    if LOG_PATH:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def subset(urls, indices):
    return [urls[int(i)] for i in indices]


def environment():
    names = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "tldextract", "joblib"]
    return {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": {name: importlib.metadata.version(name) for name in names}}


def predict_bundle(bundle, urls, frame=None, exact_override=True):
    frame = make_features(urls) if frame is None else frame
    probabilities = bundle["weight"] * bundle["text"].predict_proba(urls)
    if bundle.get("boost") is not None:
        probabilities += (1 - bundle["weight"]) * boost_probability(bundle["boost"], frame)
    overridden = 0
    if exact_override:
        for i, url in enumerate(urls):
            if url in bundle["known_urls"]:
                probabilities[i] = bundle["known_urls"][url]
                overridden += 1
        if bundle.get("external_reference") is not None:
            from .reference import apply_reference
            probabilities, _ = apply_reference(urls, probabilities, bundle["external_reference"])
    return probabilities, overridden


def run_train(args):
    global LOG_PATH
    root = Path(args.root).resolve()
    reports, artifacts, submissions = [root / name for name in ["reports", "artifacts", "submissions"]]
    for folder in [reports, artifacts, submissions]:
        folder.mkdir(parents=True, exist_ok=True)
    LOG_PATH = reports / "training.log"
    LOG_PATH.write_text("", encoding="utf-8")
    started = time.perf_counter()
    urls, labels, audit = load_train(root / "data/train.csv", reports)
    test_urls = load_test(root / "data/test.csv")
    groups = np.array([group_for_url(url) for url in urls])
    frame = make_features(urls)
    test_frame = make_features(test_urls)
    known_urls = dict(zip(urls, labels.tolist(), strict=True))
    hosts = set(frame.cat_host)
    domains = set(groups)
    audit["test"] = {"rows": len(test_urls), "sha256": sha256(root / "data/test.csv"),
                     "sample_sha256": sha256(root / "data/sample.csv"),
                     "duplicate_url_rows": len(test_urls) - len(set(test_urls)),
                     "known_exact_url_rows": sum(url in known_urls for url in test_urls),
                     "known_host_rows": sum(host in hosts for host in test_frame.cat_host),
                     "known_registered_domain_rows": sum(group_for_url(url) in domains for url in test_urls)}
    audit["features"] = {"structural_columns": len(frame.columns), "categorical_columns": int(sum(c.startswith("cat_") for c in frame))}
    dump(reports / "data_audit.json", audit)
    dump(reports / "environment.json", environment())
    log(f"Clean data: {len(urls)} unique URLs; {audit['rejected_records']} rejected; test: {len(test_urls)}")

    indices = np.arange(len(urls))
    dev, holdout = train_test_split(indices, test_size=0.2, stratify=labels, random_state=704)
    np.savez_compressed(reports / "split_indices.npz", development=dev, holdout=holdout)
    # Nothing in holdout influences vectorizers, model fitting, thresholds or blend selection.
    oof = {name: np.zeros(len(dev), dtype=np.float64) for name in ["tfidf", "nb", "boost"]}
    fold_rows = []
    for fold, (training, validation) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=2026).split(dev, labels[dev]), start=1):
        tr, va = dev[training], dev[validation]
        assert not set(subset(urls, tr)).intersection(subset(urls, va))
        train_urls, validation_urls = subset(urls, tr), subset(urls, va)
        for name in ["tfidf", "nb"]:
            log(f"Development fold {fold}/3: fitting {name}")
            model = TextModel(name).fit(train_urls, labels[tr])
            probability = model.predict_proba(validation_urls)
            oof[name][validation] = probability
            score = measure(labels[va], probability)
            fold_rows.append({"fold": fold, "model": name, "fit_seconds": model.fit_seconds,
                              "text_features": model.n_features, **score})
            log(f"{name}: F1 at 0.5 = {score['f1']:.6f}; {model.fit_seconds:.1f}s")
            del model
            gc.collect()
        log(f"Development fold {fold}/3: fitting CatBoost")
        model = fit_boost(frame.iloc[tr], labels[tr], threads=args.threads, iterations=args.iterations)
        probability = boost_probability(model, frame.iloc[va])
        oof["boost"][validation] = probability
        score = measure(labels[va], probability)
        fold_rows.append({"fold": fold, "model": "boost", "fit_seconds": model.fit_seconds, **score})
        log(f"CatBoost: F1 at 0.5 = {score['f1']:.6f}; {model.fit_seconds:.1f}s")
        del model
        pd.DataFrame(fold_rows).to_csv(reports / "development_folds.csv", index=False)
    np.savez_compressed(reports / "development_oof.npz", labels=labels[dev], **oof)
    selected, candidates = choose_ensemble(labels[dev], oof)
    pd.DataFrame(candidates).to_csv(reports / "blend_candidates.csv", index=False)
    individual = {}
    for name, proba in oof.items():
        threshold, _ = best_threshold(labels[dev], proba)
        individual[name] = measure(labels[dev], proba, threshold)
    alternate = max(["tfidf", "nb"], key=lambda name: individual[name]["f1"])
    alternate_threshold = individual[alternate]["threshold"]
    config = {**selected, "alternate_text_model": alternate, "alternate_threshold": alternate_threshold,
              "seed": 2026, "holdout_seed": 704, "folds": 3, "iterations": args.iterations,
              "threads": args.threads, "text_C": 4.0}
    dump(artifacts / "config.json", config)
    log(f"Configuration frozen using development data: {json.dumps(config)}")

    needed_text = sorted({selected["text_model"], alternate})
    validation_text = {}
    for name in needed_text:
        log(f"Fitting {name} on development pool for untouched holdout")
        validation_text[name] = TextModel(name).fit(subset(urls, dev), labels[dev])
    validation_boost = fit_boost(frame.iloc[dev], labels[dev], threads=args.threads, iterations=args.iterations)
    holdout_pred = {name: model.predict_proba(subset(urls, holdout)) for name, model in validation_text.items()}
    holdout_pred["boost"] = boost_probability(validation_boost, frame.iloc[holdout])
    primary_probability = selected["text_weight"] * holdout_pred[selected["text_model"]] + (1 - selected["text_weight"]) * holdout_pred["boost"]
    validation_scores = {name: measure(labels[holdout], proba, individual[name]["threshold"]) for name, proba in holdout_pred.items()}
    validation_scores["primary_ensemble"] = measure(labels[holdout], primary_probability, selected["threshold"])
    log(f"Untouched holdout primary F1 = {validation_scores['primary_ensemble']['f1']:.6f}")
    pd.DataFrame({"clean_row": holdout, "url": subset(urls, holdout), "truth": labels[holdout],
                  "probability": primary_probability, "predicted": (primary_probability >= selected["threshold"]).astype(int)}).to_csv(reports / "holdout_predictions.csv", index=False)
    importance = pd.DataFrame({"feature": frame.columns, "importance": validation_boost.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)
    development_domains = set(groups[dev])
    metrics = {"development_oof_tuned": individual, "selected_configuration": config,
               "untouched_holdout": validation_scores, "split": {"development_rows": len(dev), "holdout_rows": len(holdout),
               "exact_url_overlap": 0, "holdout_known_domain_rows": int(sum(group in development_domains for group in groups[holdout]))},
               "protocol": "Models and decision thresholds selected on development OOF; holdout measured once with frozen settings."}
    dump(reports / "metrics.json", metrics)
    del validation_text, validation_boost
    gc.collect()

    log("Checking generalization to unseen registered domains (development pool only)")
    grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1401)
    training, validation = next(grouped.split(dev, labels[dev], groups[dev]))
    tr, va = dev[training], dev[validation]
    overlap = set(groups[tr]).intersection(groups[va])
    assert not overlap
    np.savez_compressed(reports / "domain_split_indices.npz", training=tr, validation=va)
    group_text = TextModel(selected["text_model"]).fit(subset(urls, tr), labels[tr])
    group_boost = fit_boost(frame.iloc[tr], labels[tr], threads=args.threads, iterations=args.iterations)
    group_t = group_text.predict_proba(subset(urls, va))
    group_b = boost_probability(group_boost, frame.iloc[va])
    group_p = selected["text_weight"] * group_t + (1 - selected["text_weight"]) * group_b
    metrics["unseen_domain_check"] = {"primary_ensemble": measure(labels[va], group_p, selected["threshold"]),
                                     "text": measure(labels[va], group_t, individual[selected["text_model"]]["threshold"]),
                                     "boost": measure(labels[va], group_b, individual["boost"]["threshold"]),
                                     "training_rows": len(tr), "validation_rows": len(va), "domain_overlap": len(overlap),
                                     "private_suffixes_included": True,
                                     "threshold_source": "random development OOF, not the domain validation labels"}
    dump(reports / "metrics.json", metrics)
    pd.DataFrame({"clean_row": va, "url": subset(urls, va), "domain": groups[va], "truth": labels[va],
                  "probability": group_p, "predicted": (group_p >= selected["threshold"]).astype(int)}).to_csv(reports / "domain_predictions.csv", index=False)
    log(f"Unseen-domain primary F1 = {metrics['unseen_domain_check']['primary_ensemble']['f1']:.6f}")
    del group_text, group_boost
    gc.collect()

    full_text = {}
    for name in needed_text:
        log(f"Final fit on all clean training URLs: {name}")
        full_text[name] = TextModel(name).fit(urls, labels)
    log("Final fit on all clean training URLs: CatBoost")
    full_boost = fit_boost(frame, labels, threads=args.threads, iterations=args.iterations)
    bundle = {"text": full_text[selected["text_model"]], "boost": full_boost,
              "weight": selected["text_weight"], "threshold": selected["threshold"], "known_urls": known_urls,
              "input_sha256": audit["input_sha256"], "environment": environment(), "config": config}
    joblib.dump(bundle, artifacts / "primary.joblib", compress=3)
    alternate_bundle = {"text": full_text[alternate], "boost": None, "weight": 1.0,
                        "threshold": alternate_threshold, "known_urls": known_urls,
                        "input_sha256": audit["input_sha256"], "environment": environment(), "config": config}
    joblib.dump(alternate_bundle, artifacts / "text.joblib", compress=3)
    submissions_info = {}
    for name, fitted in [("primary", bundle), ("text", alternate_bundle)]:
        probability, overridden = predict_bundle(fitted, test_urls, test_frame)
        predicted = (probability >= fitted["threshold"]).astype(np.int8)
        submissions_info[name] = write_submission(root / "data/sample.csv", submissions / f"submission_{name}.csv", predicted)
        submissions_info[name]["exact_known_url_overrides"] = overridden
        pd.DataFrame({"Id": np.arange(len(test_urls)), "probability": probability}).to_csv(submissions / f"probabilities_{name}.csv", index=False)
    primary_labels = pd.read_csv(submissions / "submission_primary.csv").Predicted.to_numpy()
    text_labels = pd.read_csv(submissions / "submission_text.csv").Predicted.to_numpy()
    submissions_info["different_predictions"] = int(np.sum(primary_labels != text_labels))
    dump(reports / "submission_checks.json", submissions_info)
    metrics["total_training_seconds"] = time.perf_counter() - started
    dump(reports / "metrics.json", metrics)
    log(f"DONE. Submissions saved; differing labels: {submissions_info['different_predictions']}; total: {metrics['total_training_seconds']:.1f}s")


def run_predict(args):
    bundle = joblib.load(args.model)
    urls = load_test(args.test)
    proba, overridden = predict_bundle(bundle, urls)
    result = write_submission(args.sample, args.output, (proba >= bundle["threshold"]).astype(np.int8))
    external = bundle.get("external_reference", {})
    log(json.dumps({**result, "known_url_overrides": overridden,
                    "external_reference_matches": sum(url in external for url in urls)}))


def run_verify(args):
    urls = load_test(args.test)
    with Path(args.submission).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError("Incorrect submission header")
        rows = list(reader)
    assert len(rows) == len(urls), "Row count differs from test"
    assert [row["Id"] for row in rows] == [str(i) for i in range(len(urls))], "IDs/order incorrect"
    assert all(row["Predicted"] in {"0", "1"} for row in rows), "Non-binary labels"
    log(f"VALID: {len(rows)} rows, header Id,Predicted, binary labels, consecutive IDs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--root", default=".")
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--iterations", type=int, default=1000)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", default="artifacts/primary.joblib")
    predict.add_argument("--test", default="data/test.csv")
    predict.add_argument("--sample", default="data/sample.csv")
    predict.add_argument("--output", default="submissions/reproduced.csv")
    verify = commands.add_parser("verify")
    verify.add_argument("--test", default="data/test.csv")
    verify.add_argument("--submission", default="submissions/submission_primary.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"train": run_train, "predict": run_predict, "verify": run_verify}[args.command](args)


if __name__ == "__main__":
    main()
