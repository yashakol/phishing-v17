"""Evaluate a frozen extra-HTML mixture, then fit added models on the full train."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit
from experiments.v6.oof_extra import parameters
from experiments.v6.screen_extra import scope, sha
from phishing_url.content_bundle import prediction_frame
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_memory import fit_memory, text_fingerprint
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.data import write_submission
from phishing_url.evaluation import measure
from phishing_url.extra_text import ExtraTextModel
from phishing_url.neighbors import NeighborModel
from phishing_url.soft_context import context_keys, fit_context
from phishing_url.v6_bundle import predict_v6
import train_v5


def subset(cache, indices):
    result = {"urls": [cache["urls"][int(i)] for i in indices],
              "records": records_at(cache, indices),
              "features": cache["features"].iloc[indices].reset_index(drop=True)}
    if "hashes" in cache:
        result["hashes"] = [cache["hashes"][int(i)] for i in indices]
    return result


def baseline_development(cache, frame, splits, version, digest):
    tr = splits["development"]
    base = train_v5.development_base(cache, frame, splits, digest)
    memory = fit_memory(records_at(cache, tr), [cache["hashes"][int(i)] for i in tr], cache["labels"][tr], 3, 5)
    if version == 3:
        base.update(memory=memory, memory_minimum_support=3, domain_minimum_support=5)
        return base
    chosen = json.loads((ROOT / "reports/v5/selection.json").read_text())["selected"]
    config = chosen["config"]
    bundle = {"base": base, "config": config, "threshold": chosen["threshold"], "memory": memory,
              "training_rows": len(tr), "leaf": None,
              "cat": train_v5.complementary(cache, frame, tr, "cat", "holdout", digest) if config["cat_weight"] else None,
              "hashed": train_v5.complementary(cache, frame, tr, "hash", "holdout", digest) if config["hash_weight"] else None}
    if "context_config" in chosen:
        bundle.update(context_config=chosen["context_config"],
                      context=fit_context(context_keys(records_at(cache, tr)), cache["labels"][tr]))
    return bundle


def added_model(kind, stage, cache, extra, frame, indices, digest, refit):
    if kind == "neighbors":
        config = json.loads((ROOT / "experiments/v6/url_neighbor_selection.json").read_text())["model_parameters"]
        scoped = {**config, "source_sha256": sha(ROOT / "phishing_url/neighbors.py")}
    else:
        config = parameters(kind)
        scoped = scope(config)
    expected = guard(cache, indices, scoped, digest)
    path = ROOT / f"artifacts/v6/component_cache/{stage}_{kind}.joblib"
    if path.exists() and not refit:
        return load_component(path, expected)
    started = time.perf_counter()
    print(json.dumps({"event": "v6_fit_started", "kind": kind, "stage": stage, "rows": len(indices)}), flush=True)
    if kind == "text":
        model = ExtraTextModel(**config).fit(records_at(extra, indices), cache["labels"][indices])
    elif kind == "neighbors":
        model = NeighborModel(**config).fit(records_at(cache, indices), cache["labels"][indices])
    else:
        combined = pd.concat([frame, extra["features"].reset_index(drop=True)], axis=1)
        model = fit(combined.iloc[indices], cache["labels"][indices], config, threads=4)
    save_component(path, model, expected)
    print(json.dumps({"event": "v6_fit_complete", "kind": kind, "stage": stage,
                      "seconds": time.perf_counter() - started}), flush=True)
    return model


def bundle_for(base, chosen, models, frame, extra, rows):
    return {"base": base, "config": chosen["config"], "threshold": chosen["threshold"],
            "extra_text": models.get("text"), "extra_boost": models.get("boost"),
            "neighbors": models.get("neighbors"), "training_rows": rows,
            "feature_columns": list(frame.columns) + list(extra["features"].columns)}


def main(refit):
    started = time.perf_counter()
    out = ROOT / "reports/v6"
    artifacts = ROOT / "artifacts/v6"
    artifacts.mkdir(exist_ok=True)
    selection = json.loads((out / "selection.json").read_text())
    chosen, config = selection["selected"], selection["selected"]["config"]
    assert chosen["metrics"]["f1"] > selection["baseline_v5"]["f1"]
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    digest = sha(ROOT / "data/content_train.joblib")
    assert selection["train_cache_sha256"] == digest
    assert selection["extra_cache_sha256"] == sha(ROOT / "data/extra_html_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["development"], splits["holdout"]
    assert not set(tr) & set(va) and len(splits["added"]) == 178
    frame = prediction_frame(cache, url["features"])
    kinds = (["text"] if config["extra_text_weight"] else []) + (["boost"] if config["extra_boost_weight"] else [])
    if config.get("neighbor_adjustment"):
        kinds.append("neighbors")
    base = baseline_development(cache, frame, splits, config["base_version"], digest)
    models = {kind: added_model(kind, "holdout", cache, extra, frame, tr, digest, refit) for kind in kinds}
    bundle = bundle_for(base, chosen, models, frame, extra, len(tr))
    val, valextra = subset(cache, va), subset(extra, va)
    p, hits = predict_v6(bundle, val, valextra, url["features"].iloc[va])
    pred, y = p >= chosen["threshold"], cache["labels"][va]
    old = pd.read_csv(ROOT / "reports/v5/holdout_predictions.csv")
    np.testing.assert_array_equal(old.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(old.label.to_numpy(), y)
    assert old.url.tolist() == val["urls"]
    before = old.prediction.to_numpy()
    raw = set(cache["hashes"][int(i)] for i in tr)
    compact = {text_fingerprint(record) for record in records_at(cache, tr)}
    novel = np.array([h not in raw and text_fingerprint(r) not in compact
                      for r, h in zip(val["records"], val["hashes"], strict=True)])
    metrics = {"selection": chosen, "holdout": measure(y, p, chosen["threshold"]),
               "previous_v5_holdout": json.loads((ROOT / "reports/v5/metrics.json").read_text())["holdout"],
               "holdout_fixed_errors": int(np.sum((before != y) & (pred == y))),
               "holdout_new_errors": int(np.sum((before == y) & (pred != y))),
               "unseen_content_rows": int(novel.sum()),
               "unseen_content_metrics": measure(y[novel], p[novel], chosen["threshold"]),
               "holdout_note": "Previously used diagnostic holdout. Configuration frozen on original OOF before evaluation.",
               "training_rows": len(cache["urls"]), "external_test_labels_used": False, "new_astra_calls": 0}
    pd.DataFrame({"cache_index": va, "url": val["urls"], "label": y, "probability": p,
                  "prediction": pred, "previous_prediction": before,
                  "training_rule_match": hits["training_match"], "unseen_content": novel}).to_csv(out / "holdout_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"event": "v6_holdout_complete", **{k: metrics[k] for k in (
        "holdout", "holdout_fixed_errors", "holdout_new_errors", "unseen_content_metrics")}}), flush=True)
    joblib.dump(bundle, artifacts / "holdout.joblib", compress=3)
    del base, bundle, models, val, valextra, raw, compact
    gc.collect()
    indices = np.arange(len(cache["urls"]))
    base = joblib.load(ROOT / ("artifacts/v5/ensemble.joblib" if config["base_version"] == 5 else "artifacts/v3/content.joblib"))
    learned = base["base"] if config["base_version"] == 5 else base
    assert learned["known_urls"] == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    assert base["training_rows"] == len(indices) == 63900
    models = {kind: added_model(kind, "full", cache, extra, frame, indices, digest, refit) for kind in kinds}
    bundle = bundle_for(base, chosen, models, frame, extra, len(indices))
    bundle.update(train_cache_sha256=digest, extra_cache_sha256=selection["extra_cache_sha256"],
                  extra_extractor_sha256=sha(ROOT / "phishing_url/extra_html.py"), validation=metrics)
    joblib.dump(bundle, artifacts / "ensemble.joblib", compress=3)
    (artifacts / "config.json").write_text(json.dumps(chosen, indent=2) + "\n")
    del cache, extra, frame, url, base, models
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    test_extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and test_extra["labels"] is None
    assert test["urls"] == test_extra["urls"] == url["urls"]
    p, hits = predict_v6(bundle, test, test_extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v6.csv", (p >= chosen["threshold"]).astype(np.int8))
    old = pd.read_csv(ROOT / "submissions/submission_content_v5.csv").Predicted.to_numpy()
    info.update(prediction_changes_vs_v5=int(np.sum(old != (p >= chosen["threshold"]))),
                known_training_urls=int(hits["known_url"].sum()), training_rule_matches=int(hits["training_match"].sum()))
    metrics.update(submission=info, seconds=time.perf_counter() - started)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame({"Id": np.arange(len(p)), "url": test["urls"], "probability": p,
                  "training_rule_match": hits["training_match"], "known_training_url": hits["known_url"]}).to_csv(out / "test_prediction_provenance.csv", index=False)
    print(json.dumps({"event": "v6_complete", "holdout": metrics["holdout"], "submission": info,
                      "seconds": metrics["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-new-models", action="store_true")
    with threadpool_limits(limits=2):
        main(parser.parse_args().refit_new_models)
