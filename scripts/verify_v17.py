"""Verify full own-train refit scopes and byte-identical complete inference."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import guard, load_component
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.v9_routing import subset_cache
from phishing_url.v10_training import sha
from phishing_url.v12_expert import eligibility, make_expert
from phishing_url.v15_bundle import predict_v15, resource_inputs
from phishing_url.v15_resources import ResourceContext


def known_labels(model):
    while "base" in model:
        assert "external_reference" not in model
        model = model["base"]
    return model["known_urls"]


def main():
    out = ROOT / "reports/v17"
    protocol, metrics, fit = [json.loads((out/name).read_text()) for name in ("preregistered.json", "metrics.json", "expert_fit.json")]
    assert metrics["submission"] is not None and metrics["new_independent_quality_evaluation"] is False
    for name, expected in {**protocol["sources_sha256"], **protocol["inputs_sha256"], "scripts/train_v17.py": protocol["source_sha256"]}.items():
        assert sha(ROOT/name) == expected, name
    assert make_expert().get_params() == protocol["parameters"] == fit["parameters"]
    with np.load(out/"full_training_frame.npz", allow_pickle=False) as saved:
        indices, labels, values, columns = saved["indices"].copy(), saved["labels"].copy(), saved["values"].copy(), saved["columns"].tolist()
        dev_mask, query_indices, query_mask = saved["development_mask"].copy(), saved["query_indices"].copy(), saved["query_eligibility"].copy()
    assert sha(out/"full_training_frame.npz") == fit["guard"]["config"]["frame_sha256"]
    assert len(indices) == len(set(indices.tolist())) == metrics["expert_training_rows"] == fit["training_rows"] == 7119
    cache, extra, url = [joblib.load(ROOT/name) for name in ("data/content_train.joblib", "data/extra_html_train.joblib", "data/content_url_train.joblib")]
    assert cache["urls"] == extra["urls"] == url["urls"]
    np.testing.assert_array_equal(extra["labels"], cache["labels"])
    np.testing.assert_array_equal(labels, cache["labels"][indices])
    splits = aligned_splits(ROOT, cache)
    dev, diagnostic, added = splits["development"], splits["holdout"], splits["added"]
    with np.load(ROOT/"experiments/v15_resources/training_frame.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], dev)
        np.testing.assert_array_equal(saved["labels"], cache["labels"][dev])
        np.testing.assert_array_equal(saved["eligibility"], dev_mask)
        assert saved["columns"].tolist() == columns
        np.testing.assert_array_equal(values[:int(dev_mask.sum())], saved["values"][dev_mask])
    previous = pd.read_csv(ROOT/"reports/v10/holdout_predictions.csv")
    np.testing.assert_array_equal(previous.cache_index.to_numpy(), diagnostic)
    np.testing.assert_array_equal(previous.label.to_numpy(), cache["labels"][diagnostic])
    assert previous.url.tolist() == [cache["urls"][int(i)] for i in diagnostic]
    hmask = eligibility(previous.probability.to_numpy())
    np.testing.assert_array_equal(query_indices, np.concatenate([diagnostic[hmask], added]))
    np.testing.assert_array_equal(indices, np.concatenate([dev[dev_mask], query_indices[query_mask]]))
    assert not set(query_indices) & set(dev)
    scope = guard(cache, indices, fit["guard"]["config"], protocol["inputs_sha256"]["data/content_train.joblib"])
    scope["ordered_training_indices_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
    assert scope == fit["guard"]
    expert = load_component(ROOT/"artifacts/v17/component_cache/full_expert.joblib", scope)
    model = joblib.load(ROOT/"artifacts/v17/ensemble.joblib")
    assert model["threshold"] == protocol["threshold"] and model["expert_weight"] == protocol["weight"]
    assert model["selection_sha256"] == sha(out/"preregistered.json") and model["refit_version"] == "v17"
    assert model["expert_training_rows"] == len(indices) and model["training_rows"] == model["base"]["training_rows"] == len(cache["urls"]) == 63900
    assert known_labels(model) == dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    reconstructed = ResourceContext().fit(extra["features"], cache["labels"])
    assert model["resource_context"].counts_ == reconstructed.counts_ and model["resource_context"].prior_ == reconstructed.prior_
    assert model["resource_context"].training_rows_ == model["resource_context_training_rows"] == 63900
    assert model["expert"].get_params() == expert.get_params() == make_expert().get_params()
    X = pd.DataFrame(values, columns=columns)
    np.testing.assert_array_equal(model["expert"].predict_proba(X, thread_count=2), expert.predict_proba(X, thread_count=2))
    old_model = joblib.load(ROOT/"artifacts/v15/holdout.joblib")
    assert known_labels(old_model) == {cache["urls"][int(i)]: int(cache["labels"][i]) for i in dev}
    dev_context = ResourceContext().fit(extra["features"].iloc[dev], cache["labels"][dev])
    assert old_model["resource_context"].counts_ == dev_context.counts_ and old_model["resource_context"].prior_ == dev_context.prior_
    with np.load(out/"new_training_features.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], query_indices)
        np.testing.assert_array_equal(saved["labels"], cache["labels"][query_indices])
        assert saved["columns"].tolist() == columns
        expected_query, expected_p = saved["values"].copy(), saved["baseline"].copy()
    resource_values = dev_context.transform(extra["features"].iloc[query_indices])
    np.testing.assert_array_equal(expected_query[:, -resource_values.shape[1]:], resource_values.to_numpy())
    np.testing.assert_array_equal(values[int(dev_mask.sum()):], expected_query[query_mask])
    query, query_extra = subset_cache(cache, query_indices), subset_cache(extra, query_indices)
    query["labels"] = query_extra["labels"] = None
    query_url = url["features"].iloc[query_indices].reset_index(drop=True)
    del cache, extra, url, reconstructed, dev_context, model, expert, X, values
    gc.collect()
    print(json.dumps({"event": "v17_verify_new_training_features_started", "rows": len(query_indices)}), flush=True)
    frame, baseline, hits = resource_inputs(old_model["base"], old_model["resource_context"], query, query_extra, query_url)
    assert not hits["known_url"].any()
    np.testing.assert_allclose(frame.to_numpy(), expected_query, rtol=0, atol=2e-12)
    np.testing.assert_allclose(baseline, expected_p, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(query_mask, eligibility(baseline))
    del old_model, query, query_extra, query_url, frame, expected_query, expected_p
    gc.collect()
    model = joblib.load(ROOT/"artifacts/v17/ensemble.joblib")
    test, extra, url = [joblib.load(ROOT/name) for name in ("data/content_test.joblib", "data/extra_html_test.joblib", "data/content_url_test.joblib")]
    assert test["labels"] is None and extra["labels"] is None
    assert test["urls"] == extra["urls"] == url["urls"] == load_test(ROOT/"data/test.csv")
    print(json.dumps({"event": "v17_verify_full_test_started", "rows": len(test["urls"])}), flush=True)
    probability, hits = predict_v15(model, test, extra, url["features"])
    info = write_submission(ROOT/"data/sample.csv", out/"reproduced_submission.csv", (probability >= model["threshold"]).astype(np.int8))
    assert (out/"reproduced_submission.csv").read_bytes() == (ROOT/"submissions/submission_content_v17.csv").read_bytes()
    result = {"status": "passed", "frozen_inputs_parameters_and_threshold": True, "own_train_labels_and_indices_verified": True,
        "expert_training_rows": len(indices), "training_rows_by_source": fit["training_rows_by_source"],
        "old_development_features_exactly_reused": True, "new_train_features_replayed": len(query_indices),
        "resource_counters_rebuilt_from_raw_own_train": True, "component_serialization_predictions_exact": True,
        "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(probability),
        "submission": info, "model_sha256": sha(ROOT/"artifacts/v17/ensemble.joblib"), "external_test_labels_used": False,
        "new_independent_quality_evaluation": False}
    (out/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with patch("socket.create_connection", side_effect=AssertionError("Network disabled")), \
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")), threadpool_limits(limits=2):
        main()
