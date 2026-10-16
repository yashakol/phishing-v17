"""Build a frozen C32/refitted-mixture candidate, then diagnose and export it."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

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

from experiments.v6.screen_extra import sha
from phishing_url.content_validation import aligned_splits
from phishing_url.data import write_submission
from phishing_url.evaluation import measure
from phishing_url.v8_bundle import predict_v8
from refit_v8_regularization import refit_coefficients, semantic_snapshot
from train_v6 import subset
from train_v7 import fit_added


def fitted_text(cache, indices, stage, selection, refit=False, reuse_only=False):
    old_selection = json.loads((ROOT / "reports/v7/selection.json").read_text())
    checked = fit_added(cache, indices, stage, old_selection, False, reuse_only=True)["replacement_text"]
    params = selection["text_parameters"]
    path = ROOT / f"artifacts/v8/component_cache/{stage}_text.joblib"
    return refit_coefficients(checked, cache, indices, params, path,
                             selection["train_cache_sha256"], sha(ROOT / "phishing_url/v7_text.py"), refit, reuse_only)


def bundle_for(base, text, selection, rows):
    return {"base": base, "replacement_text": text, "v5_config": selection["selected"]["v5_config"],
            "v6_config": selection["v6_config"], "threshold": selection["selected"]["threshold"],
            "resource_weight": 0., "resource_text": None, "training_rows": rows,
            "train_cache_sha256": selection["train_cache_sha256"],
            "selection_sha256": sha(ROOT / "reports/v8/selection.json")}


def main(refit=False):
    started = time.perf_counter()
    output, artifacts = ROOT / "reports/v8", ROOT / "artifacts/v8"
    artifacts.mkdir(exist_ok=True)
    selection = json.loads((output / "selection.json").read_text())
    assert selection["selected"]["metrics"]["f1"] > selection["baseline_v7"]["f1"]
    assert selection["external_test_labels_used"] is False
    assert selection["train_cache_sha256"] == sha(ROOT / "data/content_train.joblib")
    assert selection["text_oof_sha256"] == sha(ROOT / selection["text_oof"])
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["development"], splits["holdout"]
    assert len(tr) == 50977 and len(va) == 12745 and len(splits["added"]) == 178
    assert not set(tr) & set(va)
    base = joblib.load(ROOT / "artifacts/v7/holdout.joblib")
    assert base["training_rows"] == len(tr)
    original = base["base"]["base"]["base"]
    assert original["known_urls"] == dict(zip([cache["urls"][int(i)] for i in tr], cache["labels"][tr].tolist(), strict=True))
    print(json.dumps({"event": "v8_coefficient_fit", "stage": "holdout", "rows": len(tr)}), flush=True)
    text = fitted_text(cache, tr, "holdout", selection, refit)
    model = bundle_for(base, text, selection, len(tr))
    p, _ = predict_v8(model, subset(cache, va), subset(extra, va), url["features"].iloc[va])
    y = cache["labels"][va]
    previous = pd.read_csv(ROOT / "reports/v7/holdout_predictions.csv")
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(previous.label.to_numpy(), y)
    assert previous.url.tolist() == [cache["urls"][int(i)] for i in va]
    before, after = previous.prediction.to_numpy(), p >= model["threshold"]
    report = {"holdout": measure(y, p, model["threshold"]),
              "previous_v7_holdout": json.loads((ROOT / "reports/v7/metrics.json").read_text())["holdout"],
              "holdout_fixed_errors": int(np.sum((before != y) & (after == y))),
              "holdout_new_errors": int(np.sum((before == y) & (after != y))),
              "holdout_note": "Reused diagnostic holdout; configuration and threshold frozen before its replay.",
              "selection": selection["selected"], "text_parameters": selection["text_parameters"],
              "training_rows": len(cache["urls"]), "external_test_labels_used": False, "new_astra_calls": 1}
    pd.DataFrame({"cache_index": va, "url": previous.url, "label": y, "probability": p,
                  "prediction": after, "previous_prediction": before}).to_csv(output / "holdout_predictions.csv", index=False)
    joblib.dump(model, artifacts / "holdout.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "v8_holdout_complete", "holdout": report["holdout"],
                      "fixed_errors": report["holdout_fixed_errors"], "new_errors": report["holdout_new_errors"]}), flush=True)
    del model, base, original, text
    gc.collect()
    base = joblib.load(ROOT / "artifacts/v7/ensemble.joblib")
    assert base["training_rows"] == len(cache["urls"]) == 63900
    original = base["base"]["base"]["base"]
    assert original["known_urls"] == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    assert all("external_reference" not in item for item in (base, base["base"], base["base"]["base"], original))
    print(json.dumps({"event": "v8_coefficient_fit", "stage": "full", "rows": len(cache["urls"])}), flush=True)
    text = fitted_text(cache, np.arange(len(cache["urls"])), "full", selection, refit)
    model = bundle_for(base, text, selection, len(cache["urls"]))
    model["validation"] = report
    joblib.dump(model, artifacts / "ensemble.joblib", compress=3)
    (artifacts / "config.json").write_text(json.dumps(selection, indent=2) + "\n")
    del cache, extra, url, text, base, original
    gc.collect()
    test = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"]
    p, hits = predict_v8(model, test, extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v8.csv",
                            (p >= model["threshold"]).astype(np.int8))
    old = pd.read_csv(ROOT / "submissions/submission_content_v7.csv").Predicted.to_numpy()
    info.update(prediction_changes_vs_v7=int(np.sum(old != (p >= model["threshold"]))),
                known_training_urls=int(hits["known_url"].sum()), training_rule_matches=int(hits["training_match"].sum()))
    pd.DataFrame({"Id": np.arange(len(p)), "url": test["urls"], "probability": p}).to_csv(output / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "v8_complete", "submission": info, "holdout": report["holdout"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-new-models", action="store_true")
    with threadpool_limits(limits=2):
        main(parser.parse_args().refit_new_models)
