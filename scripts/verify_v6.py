"""Verify OOF scope and replay saved development/full models with sockets blocked."""

import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.v6.oof_extra import parameters
from experiments.v6.screen_extra import scope, sha
from phishing_url.content_checkpoints import guard, load_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.neighbors import adjust_neighbors
from phishing_url.v6_bundle import mix_extra, gate_extra, predict_v6
from train_v6 import subset


def assert_metrics(actual, expected):
    for key in ("f1", "fp", "fn", "tp", "tn", "rows"):
        assert actual[key] == expected[key], key


def no_network():
    return (patch("socket.create_connection", side_effect=AssertionError("Network disabled")),
            patch("socket.getaddrinfo", side_effect=AssertionError("Network disabled")))


def main():
    out = ROOT / "reports/v6"
    selection = json.loads((out / "selection.json").read_text())
    chosen, config = selection["selected"], selection["selected"]["config"]
    metrics = json.loads((out / "metrics.json").read_text())
    cache = joblib.load(ROOT / "data/content_train.joblib")
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    assert cache["urls"] == extra["urls"]
    np.testing.assert_array_equal(cache["labels"], extra["labels"])
    digest = sha(ROOT / "data/content_train.joblib")
    assert selection["train_cache_sha256"] == digest
    assert selection["extra_cache_sha256"] == sha(ROOT / "data/extra_html_train.joblib")
    splits = aligned_splits(ROOT, cache)
    dev = splits["development"]
    table = np.load(out / "development_oof.npz")
    np.testing.assert_array_equal(table["indices"], dev)
    np.testing.assert_array_equal(table["labels"], cache["labels"][dev])
    y = table["labels"]
    base = table[f"base_v{config['base_version']}"]
    replay = mix_extra(base, table["extra_text"], table["extra_boost"], config)
    replay = gate_extra(base, replay, table["extra_text"], cache["features"].html_text_length.to_numpy()[dev], config)
    if config.get("neighbor_adjustment"):
        neighbors = np.load(ROOT / "experiments/v6/url_neighbors_development_oof.npz")
        np.testing.assert_array_equal(neighbors["indices"], dev)
        replay = adjust_neighbors(replay, neighbors["similarities"], neighbors["neighbor_labels"],
                                  {**config["neighbor_adjustment"], "prior": neighbors["prior"]})
    np.testing.assert_allclose(replay, table["probabilities"], rtol=0, atol=1e-14)
    assert_metrics(measure(y, replay, chosen["threshold"]), chosen["metrics"])
    checks = []
    for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(dev, y)):
        if config["extra_text_weight"]:
            expected = guard(cache, dev[tr], scope(parameters("text")), digest)
            saved = np.load(ROOT / f"experiments/v6/extra_text_oof/fold_{fold}.npz")
            np.testing.assert_array_equal(saved["indices"], dev[va])
            np.testing.assert_array_equal(saved["labels"], y[va])
            assert str(saved["scope_json"]) == json.dumps(expected, sort_keys=True)
            model = load_component(ROOT / f"experiments/v6/extra_text_oof/fold_{fold}.joblib", expected)
            positions = va[:32]
            p = model.predict_proba(records_at(extra, dev[positions]))
            np.testing.assert_allclose(p, table["extra_text"][positions], rtol=0, atol=1e-14)
            checks.append({"kind": "extra_text", "fold": fold, "training_rows": len(tr),
                           "training_scope_guard": "passed", "replayed_rows": len(positions)})
            del model
    hold = joblib.load(ROOT / "artifacts/v6/holdout.joblib")
    assert hold["training_rows"] == len(dev) and hold["config"] == config
    learned = hold["base"]["base"] if config["base_version"] == 5 else hold["base"]
    assert learned["known_urls"] == dict(zip([cache["urls"][int(i)] for i in dev], cache["labels"][dev].tolist(), strict=True))
    va = splits["holdout"]
    url = joblib.load(ROOT / "data/content_url_train.joblib")
    a, b = no_network()
    with a, b:
        p, _ = predict_v6(hold, subset(cache, va), subset(extra, va), url["features"].iloc[va])
    hold_table = pd.read_csv(out / "holdout_predictions.csv")
    np.testing.assert_array_equal(hold_table.cache_index.to_numpy(), va)
    np.testing.assert_array_equal(hold_table.label.to_numpy(), cache["labels"][va])
    np.testing.assert_allclose(p, hold_table.probability.to_numpy(), rtol=0, atol=2e-15)
    assert_metrics(measure(cache["labels"][va], p, chosen["threshold"]), metrics["holdout"])
    labels = dict(zip(cache["urls"], cache["labels"].tolist(), strict=True))
    del cache, extra, table, hold, url
    gc.collect()
    cache = joblib.load(ROOT / "data/content_test.joblib")
    extra = joblib.load(ROOT / "data/extra_html_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(ROOT / "data/test.csv")
    path = ROOT / "artifacts/v6/ensemble.joblib"
    model = joblib.load(path)
    assert model["config"] == config and model["threshold"] == chosen["threshold"]
    assert model["train_cache_sha256"] == digest and model["extra_cache_sha256"] == selection["extra_cache_sha256"]
    assert model["extra_extractor_sha256"] == sha(ROOT / "phishing_url/extra_html.py")
    learned = model["base"]["base"] if config["base_version"] == 5 else model["base"]
    assert learned["known_urls"] == labels and model["training_rows"] == 63900
    assert "external_reference" not in model and "external_reference" not in learned
    a, b = no_network()
    with a, b:
        p, hits = predict_v6(model, cache, extra, url["features"])
    info = write_submission(ROOT / "data/sample.csv", out / "reproduced_submission.csv", (p >= chosen["threshold"]).astype(np.int8))
    assert (out / "reproduced_submission.csv").read_bytes() == (ROOT / "submissions/submission_content_v6.csv").read_bytes()
    result = {"oof_mixture_direct_replay": "passed", "oof_rows": len(y), "oof_model_checks": checks,
              "holdout_model_replay": "passed", "holdout_rows": len(va),
              "offline_full_inference_byte_identical": True, "test_rows_in_original_order": len(p),
              "submission": info, "model_sha256": sha(path), "external_test_labels_used": False,
              "known_training_urls": int(hits["known_url"].sum()), "new_astra_calls": 0}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
