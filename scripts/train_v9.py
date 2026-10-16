"""Bundle existing learned classifiers under frozen routing and export CSV."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.v6.screen_extra import sha
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v9_routing import predict_v9, subset_cache
from phishing_url import rich_text

# Historical development pickle used this module name; its source is the
# same verified URL estimator now provided through the package namespace.
assert (ROOT / "experiments/astra/url_model.py").read_bytes() == (ROOT / "phishing_url/rich_text.py").read_bytes()
sys.modules["url_model"] = rich_text


def bundle(base, selection, rows):
    return {"base": base, "routing": selection["selected"]["policy"],
        "threshold": selection["selected"]["threshold"], "training_rows": rows,
        "selection_sha256": sha(ROOT / "reports/v9/selection.json"),
        "train_cache_sha256": selection["train_cache_sha256"]}


def main():
    started = time.perf_counter()
    out, artifacts = ROOT / "reports/v9", ROOT / "artifacts/v9"
    artifacts.mkdir(exist_ok=True)
    assert not (out / "metrics.json").exists(), "Frozen candidate only; no adaptive reruns"
    selection = json.loads((out / "selection.json").read_text())
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_holdout.json").read_bytes()
    assert selection["routing_source_sha256"] == sha(ROOT / "phishing_url/v9_routing.py")
    assert selection["base_v7_model_sha256"] == sha(ROOT / "artifacts/v7/ensemble.joblib")
    assert selection["train_cache_sha256"] == sha(ROOT / "data/content_train.joblib")
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["development"], splits["holdout"]
    base = joblib.load(ROOT / "artifacts/v7/holdout.joblib")
    assert base["training_rows"] == len(tr) == 50977
    original = base["base"]["base"]["base"]
    assert original["known_urls"] == dict(zip([cache["urls"][int(i)] for i in tr], cache["labels"][tr].tolist(), strict=True))
    model = bundle(base, selection, len(tr))
    p, hits = predict_v9(model, subset_cache(cache, va), subset_cache(extra, va), url["features"].iloc[va])
    y = cache["labels"][va]
    old = pd.read_csv(ROOT / "reports/v7/holdout_predictions.csv")
    np.testing.assert_array_equal(old.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(old.label.to_numpy(), y)
    before, after = old.prediction.to_numpy(), p >= model["threshold"]
    report = {"holdout": measure(y, p, model["threshold"]),
        "previous_v7_holdout": json.loads((ROOT / "reports/v7/metrics.json").read_text())["holdout"],
        "holdout_fixed_errors": int(((before != y) & (after == y)).sum()),
        "holdout_new_errors": int(((before == y) & (after != y)).sum()),
        "holdout_url_routed_rows": int(hits["url_routed"].sum()),
        "holdout_note": "Reused diagnostic holdout; routing configuration and original threshold frozen beforehand",
        "selection": selection["selected"], "new_classifier_fits_in_active_solution": 0,
        "external_test_labels_used": False}
    pd.DataFrame({"cache_index": va, "url": old.url, "label": y, "probability": p,
        "prediction": after, "previous_prediction": before, "url_routed": hits["url_routed"]}).to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump(model, artifacts / "holdout.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"event": "v9_holdout_complete", **report}), flush=True)
    del model, base, original
    gc.collect()
    base = joblib.load(ROOT / "artifacts/v7/ensemble.joblib")
    assert base["training_rows"] == len(cache["urls"]) == 63900
    original = base["base"]["base"]["base"]
    assert original["known_urls"] == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    model = bundle(base, selection, len(cache["urls"]))
    joblib.dump(model, artifacts / "ensemble.joblib", compress=3)
    (artifacts / "config.json").write_text(json.dumps(selection, indent=2)+"\n")
    del cache, extra, url, base, original
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    p, hits = predict_v9(model, cache, extra, url["features"])
    pred = (p >= model["threshold"]).astype(np.int8)
    info = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v9.csv", pred)
    info.update(prediction_changes_vs_v7=int((pred != pd.read_csv(ROOT / "submissions/submission_content_v7.csv").Predicted.to_numpy()).sum()),
        prediction_changes_vs_v8=int((pred != pd.read_csv(ROOT / "submissions/submission_content_v8.csv").Predicted.to_numpy()).sum()),
        known_training_urls=int(hits["known_url"].sum()), training_rule_matches=int(hits["training_match"].sum()),
        url_routed_rows=int(hits["url_routed"].sum()))
    pd.DataFrame({"Id": np.arange(len(p)), "url": cache["urls"], "probability": p,
        "url_routed": hits["url_routed"]}).to_csv(out / "test_prediction_provenance.csv", index=False)
    report.update(submission=info, seconds=time.perf_counter()-started)
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"event": "v9_complete", "submission": info, "holdout": report["holdout"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
