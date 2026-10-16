"""Check scopes, selected development mixture and complete offline inference."""

import gc
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.v8_ensemble.search import training_inputs
from experiments.v9_routing.search import routed_base, finish
from phishing_url.content_checkpoints import guard, load_component
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.v10_bundle import predict_v10
from phishing_url.v10_training import sha
from phishing_url.v9_routing import routing_mask, subset_cache


def assert_bundle(model, selection, rows):
    assert model["selection_sha256"] == sha(ROOT / "reports/v10/selection.json")
    assert model["threshold"] == selection["selected"]["threshold"]
    assert model["text_replacement"] == selection["selected"]["text_replacement"]
    assert model["train_cache_sha256"] == selection["train_cache_sha256"]
    assert model["training_rows"] == model["margin"].training_rows_ == rows
    assert model["margin"].transformer_.vocabulary_fit_rows_ == rows
    assert model["margin"].get_params() == selection["margin_parameters"]
    assert model["margin"].classifier_fits_ == 4
    node = model
    while isinstance(node, dict):
        assert "external_reference" not in node
        if "base" not in node:
            break
        node = node["base"]
    return node


def check_margin_scope(cache, indices, stage, digest):
    path = ROOT / f"artifacts/v10/component_cache/{stage}_margin.joblib"
    stored = joblib.load(path)
    expected = guard(cache, indices, stored["guard"]["config"], digest)
    assert stored["guard"] == expected
    config = expected["config"]
    assert config["model_source_sha256"] == sha(ROOT / "phishing_url/v9_margin.py")
    assert config["training_source_sha256"] == sha(ROOT / "phishing_url/v10_training.py")
    assert config["selection_sha256"] == sha(ROOT / "reports/v10/selection.json")
    assert config["base_component_sha256"] == sha(ROOT / config["base_component"])
    original = load_component(ROOT / config["base_component"], config["base_guard"])
    assert stored["model"].training_rows_ == len(indices)
    for name, vectorizer in original.vectorizers_.items():
        copied = stored["model"].transformer_.vectorizers_[name]
        assert copied.vocabulary_ == vectorizer.vocabulary_
        np.testing.assert_array_equal(copied.idf_, vectorizer.idf_)
    return sha(path)


def replay_oof(selection):
    inputs = training_inputs()
    assert inputs["digest"] == selection["train_cache_sha256"]
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz") as old, \
            np.load(ROOT / "experiments/v10_margin/development_oof.npz") as margin, \
            np.load(ROOT / "experiments/v9_routing/development_oof.npz") as routing:
        for table in (old, margin, routing):
            np.testing.assert_array_equal(table["indices"], inputs["indices"])
            np.testing.assert_array_equal(table["labels"], inputs["labels"])
        weight = selection["selected"]["text_replacement"]
        text = (1-weight)*old["probabilities"]+weight*margin["probabilities"]
        standard = routed_base(inputs, text, inputs["v5"]["config"]["aux_url_weight"])
        varied = routed_base(inputs, text, selection["routing"]["url_weight"])
        p = finish(inputs, np.where(routing[selection["routing"]["selector"]], varied, standard))
        for fold, (tr, va) in enumerate(inputs["folds"]):
            with np.load(ROOT / f"experiments/v10_margin/fold_{fold}.npz") as part:
                np.testing.assert_array_equal(part["train_indices"], inputs["indices"][tr])
                np.testing.assert_array_equal(part["indices"], inputs["indices"][va])
                np.testing.assert_array_equal(part["labels"], inputs["labels"][va])
                np.testing.assert_array_equal(part["probabilities"], margin["probabilities"][va])
                assert not set(part["train_indices"]) & set(part["indices"])
            report = json.loads((ROOT / f"experiments/v10_margin/fold_{fold}_execution.json").read_text())
            assert report["classifier_fits"] == 4 and report["base_model_unchanged"]
            assert report["model_sha256"] == sha(ROOT / f"experiments/v10_margin/models/fold_{fold}.joblib")
    with np.load(ROOT / "experiments/v10_margin/selected_oof.npz") as saved:
        np.testing.assert_allclose(saved["probabilities"], p, rtol=0, atol=1e-14)
        assert measure(saved["labels"], p, selection["selected"]["threshold"]) == selection["selected"]["metrics"]
    return len(p)


def main():
    out = ROOT / "reports/v10"
    selection = json.loads((out / "selection.json").read_text())
    protocol = json.loads((out / "integration_protocol.json").read_text())
    assert selection["gate_passed"]
    assert (out / "selection.json").read_bytes() == (out / "frozen_before_diagnostic.json").read_bytes()
    for path, digest in {**protocol["sources_sha256"], **protocol["base_files_sha256"]}.items():
        assert sha(ROOT / path) == digest, path
    assert sha(ROOT / "data/content_train.joblib") == selection["train_cache_sha256"]
    assert sha(ROOT / "experiments/v10_margin/development_oof.npz") == selection["text_oof_sha256"]
    oof_rows = replay_oof(selection)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["diagnostic_gate_passed"] and metrics["submission"] is not None
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    component_hashes = {stage: check_margin_scope(cache, indices, stage, selection["train_cache_sha256"])
                        for stage, indices in (("holdout", splits["development"]),
                                              ("full", np.arange(len(cache["urls"]), dtype=np.int64)))}
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    assert cache["urls"] == extra["urls"] == url["urls"]
    va, tr = splits["holdout"], splits["development"]
    model = joblib.load(ROOT / "artifacts/v10/holdout.joblib")
    original = assert_bundle(model, selection, len(tr))
    assert original["known_urls"] == dict(zip([cache["urls"][int(i)] for i in tr], cache["labels"][tr].tolist(), strict=True))
    p, hits = predict_v10(model, subset_cache(cache, va), subset_cache(extra, va), url["features"].iloc[va])
    saved = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(saved.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(saved.label.to_numpy(), cache["labels"][va])
    np.testing.assert_allclose(saved.probability.to_numpy(), p, atol=2e-15, rtol=0)
    assert measure(cache["labels"][va], p, model["threshold"]) == metrics["holdout"]
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    del cache, extra, url, model, original
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    model = joblib.load(ROOT / "artifacts/v10/ensemble.joblib")
    original = assert_bundle(model, selection, 63900)
    assert original["known_urls"] == labels
    p, hits = predict_v10(model, cache, extra, url["features"])
    np.testing.assert_array_equal(hits["url_routed"], routing_mask(cache, selection["routing"]["selector"]))
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv",
                            (p >= model["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v10.csv").read_bytes()
    report = {"oof_mixture_direct_replay": "passed", "oof_rows": oof_rows,
              "frozen_configuration": "passed", "holdout_model_replay": "passed", "holdout_rows": len(va),
              "margin_training_scopes": "passed", "component_hashes": component_hashes,
              "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
              "submission": info, "known_training_urls": int(hits["known_url"].sum()),
              "url_routed_rows": int(hits["url_routed"].sum()),
              "model_sha256": sha(ROOT / "artifacts/v10/ensemble.joblib"), "external_test_labels_used": False}
    (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    with patch("socket.create_connection", side_effect=AssertionError("Network disabled")), \
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")), threadpool_limits(limits=2):
        main()
