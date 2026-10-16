"""Fit a frozen supervised candidate, evaluate diagnostic holdout, then export."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import argparse
import gc
import hashlib
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

from experiments.v6.screen_extra import sha
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.data import write_submission
from phishing_url.evaluation import measure
from phishing_url.v7_bundle import predict_v7
from phishing_url.v7_hierarchy import HierarchicalEvidence, hierarchy_keys
from phishing_url.v7_text import FullVocabularyTextModel
from train_v6 import subset


def fit_added(cache, indices, stage, selection, refit, reuse_only=False):
    output = ROOT / "artifacts/v7/component_cache"
    digest = selection["train_cache_sha256"]
    models = {"replacement_text": None, "hierarchy": None}
    if selection["selected"]["config"]["text_replacement"]:
        parameters = selection["text_parameters"]
        scoped = {**parameters, "source_sha256": sha(ROOT / "phishing_url/v7_text.py"),
                  "html_text_source_sha256": sha(ROOT / "phishing_url/html_text.py")}
        expected = guard(cache, indices, scoped, digest)
        tag = hashlib.sha256(json.dumps(scoped, sort_keys=True).encode()).hexdigest()[:12]
        path = output / f"{stage}_text_{tag}.joblib"
        if reuse_only and not path.exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {path}")
        if path.exists() and not refit:
            models["replacement_text"] = load_component(path, expected)
        else:
            print(json.dumps({"event": "v7_text_fit_started", "stage": stage, "rows": len(indices)}), flush=True)
            started = time.perf_counter()
            model = FullVocabularyTextModel(**parameters).fit(records_at(cache, indices), cache["labels"][indices])
            assert model.vocabulary_fit_rows_ == len(indices)
            save_component(path, model, expected)
            models["replacement_text"] = model
            print(json.dumps({"event": "v7_text_fit_complete", "stage": stage,
                              "seconds": time.perf_counter()-started}), flush=True)
    if selection["selected"]["config"]["hierarchy"]:
        config = selection["hierarchy_parameters"]
        scoped = {**config, "source_sha256": sha(ROOT / "phishing_url/v7_hierarchy.py")}
        expected = guard(cache, indices, scoped, digest)
        tag = hashlib.sha256(json.dumps(scoped, sort_keys=True).encode()).hexdigest()[:12]
        path = output / f"{stage}_hierarchy_{tag}.joblib"
        if reuse_only and not path.exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {path}")
        if path.exists() and not refit:
            models["hierarchy"] = load_component(path, expected)
        else:
            model = HierarchicalEvidence(**config).fit_keys([hierarchy_keys(cache["urls"][int(i)]) for i in indices],
                                                           cache["labels"][indices])
            save_component(path, model, expected)
            models["hierarchy"] = model
    return models


def main(refit):
    started = time.perf_counter()
    output = ROOT / "reports/v7"
    artifacts = ROOT / "artifacts/v7"
    artifacts.mkdir(exist_ok=True)
    selection = json.loads((output / "selection.json").read_text())
    selected = selection["selected"]
    assert selected["metrics"]["f1"] > selection["baseline_v6"]["f1"]
    digest = sha(ROOT / "data/content_train.joblib")
    assert digest == selection["train_cache_sha256"]
    assert sha(ROOT / selection["text_oof"]) == selection["text_oof_sha256"]
    assert sha(ROOT / selection["text_selection"]) == selection["text_selection_sha256"]
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["development"], splits["holdout"]
    assert len(tr) == 50977 and len(va) == 12745 and len(splits["added"]) == 178
    assert not set(tr) & set(va)
    base = joblib.load(ROOT / "artifacts/v6/holdout.joblib")
    assert base["training_rows"] == len(tr)
    models = fit_added(cache, tr, "holdout", selection, refit)
    bundle = {"base": base, "config": selected["config"], "threshold": selected["threshold"], **models,
              "training_rows": len(tr), "selection_sha256": sha(output / "selection.json")}
    validation, validation_extra = subset(cache, va), subset(extra, va)
    p, hits = predict_v7(bundle, validation, validation_extra, url["features"].iloc[va])
    y = cache["labels"][va]
    previous = pd.read_csv(ROOT / "reports/v6/holdout_predictions.csv")
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(previous.label.to_numpy(), y)
    assert previous.url.tolist() == validation["urls"]
    before, after = previous.prediction.to_numpy(), p >= selected["threshold"]
    report = {"selection": selected, "holdout": measure(y, p, selected["threshold"]),
              "previous_v6_holdout": json.loads((ROOT / "reports/v6/metrics.json").read_text())["holdout"],
              "holdout_fixed_errors": int(np.sum((before != y) & (after == y))),
              "holdout_new_errors": int(np.sum((before == y) & (after != y))),
              "holdout_note": "Reused diagnostic holdout; candidate configuration/threshold frozen before its evaluation.",
              "external_test_labels_used": False, "new_astra_calls": 1, "training_rows": len(cache["urls"])}
    pd.DataFrame({"cache_index": va, "url": validation["urls"], "label": y, "probability": p, "prediction": after,
                  "previous_prediction": before}).to_csv(output / "holdout_predictions.csv", index=False)
    joblib.dump(bundle, artifacts / "holdout.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "v7_holdout_complete", "holdout": report["holdout"],
                      "fixed_errors": report["holdout_fixed_errors"], "new_errors": report["holdout_new_errors"]}), flush=True)
    del base, models, bundle, validation, validation_extra
    gc.collect()
    tr = np.arange(len(cache["urls"]))
    base = joblib.load(ROOT / "artifacts/v6/ensemble.joblib")
    assert base["training_rows"] == len(tr) == 63900
    learned = base["base"]["base"]
    assert learned["known_urls"] == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    assert "external_reference" not in learned and "external_reference" not in base
    models = fit_added(cache, tr, "full", selection, refit)
    bundle = {"base": base, "config": selected["config"], "threshold": selected["threshold"], **models,
              "training_rows": len(tr), "selection_sha256": sha(output / "selection.json"),
              "train_cache_sha256": digest, "validation": report}
    joblib.dump(bundle, artifacts / "ensemble.joblib", compress=3)
    (artifacts / "config.json").write_text(json.dumps(selected, indent=2) + "\n")
    del cache, extra, url, base, models
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    test_extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and test_extra["labels"] is None
    assert test["urls"] == test_extra["urls"] == url["urls"]
    p, hits = predict_v7(bundle, test, test_extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v7.csv", (p >= selected["threshold"]).astype(np.int8))
    previous = pd.read_csv(ROOT / "submissions/submission_content_v6.csv").Predicted.to_numpy()
    info.update(prediction_changes_vs_v6=int(np.sum(previous != (p >= selected["threshold"]))),
                known_training_urls=int(hits["known_url"].sum()), training_rule_matches=int(hits["training_match"].sum()))
    if "hierarchy_match" in hits:
        info["hierarchy_matches"] = int(hits["hierarchy_match"].sum())
    pd.DataFrame({"Id": np.arange(len(p)), "url": test["urls"], "probability": p}).to_csv(output / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "v7_complete", "submission": info, "holdout": report["holdout"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-new-models", action="store_true")
    with threadpool_limits(limits=2):
        main(parser.parse_args().refit_new_models)
