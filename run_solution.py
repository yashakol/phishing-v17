"""Offline inference and replay of the selected V17 final training stage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def offline(*args, **kwargs):
    raise RuntimeError("Network is disabled during prediction and training")


socket.create_connection = offline
socket.getaddrinfo = offline
socket.socket.connect = offline

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import phishing_url
from phishing_url.data import load_test, write_submission
from phishing_url.v12_expert import make_expert
from phishing_url.v15_bundle import predict_v15
from delivery_assets import check_csv, digest, restore, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["predict", "train"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert Path(phishing_url.__file__).resolve().is_relative_to(ROOT)
    tick = time.perf_counter()
    asset_check = restore(ROOT)
    source_count = verify_sources(ROOT)
    runtime = ROOT / ".runtime"
    config = json.loads((ROOT / "delivery_config.json").read_text())
    model = joblib.load(runtime / "artifacts/v17/ensemble.joblib")
    trained = args.action == "train"
    if trained:
        target = ROOT / "outputs/retrained_v17.joblib"
        if target.exists():
            raise FileExistsError("Retrained model already exists; preserve the previous run")
        fit = json.loads((runtime / "reports/v17/expert_fit.json").read_text())
        scope = fit["guard"]["config"]
        assert digest(runtime / "reports/v17/preregistered.json") == scope["protocol_sha256"]
        assert digest(runtime / "reports/v17/full_training_frame.npz") == scope["frame_sha256"]
        assert model["expert"].get_params() == make_expert().get_params() == fit["parameters"]
        with np.load(runtime / "reports/v17/full_training_frame.npz", allow_pickle=False) as data:
            frame = pd.DataFrame(data["values"], columns=data["columns"].tolist())
            labels, indices = data["labels"].copy(), data["indices"].copy()
        own_rows = pd.read_csv(runtime / "data/v17_training_rows.csv")
        np.testing.assert_array_equal(own_rows.cache_index.to_numpy(), indices)
        np.testing.assert_array_equal(own_rows.result.to_numpy(), labels)
        assert len(frame) == len(own_rows) == model["expert_training_rows"] == 7119
        assert frame.columns.tolist() == model["feature_columns"] == fit["feature_columns"]
        ordered = hashlib.sha256()
        for url, label in zip(own_rows.url, labels, strict=True):
            ordered.update(url.encode("utf-8")); ordered.update(b"\x00"); ordered.update(bytes([int(label)]))
        assert ordered.hexdigest() == fit["guard"]["ordered_training_sha256"]
        print(json.dumps({"event": "training_started", "rows": len(frame), "frozen_base_reused": True}), flush=True)
        with threadpool_limits(limits=2):
            expert = make_expert().fit(frame, labels)
            before = model["expert"].predict_proba(frame, thread_count=2)
            after = expert.predict_proba(frame, thread_count=2)
        np.testing.assert_array_equal(before, after)
        model = {**model, "expert": expert}
        target.parent.mkdir(exist_ok=True)
        joblib.dump(model, target, compress=3)
        model = joblib.load(target)
        print(json.dumps({"event": "training_finished", "training_probabilities_exact": True}), flush=True)
    cache = joblib.load(runtime / "data/content_test.joblib")
    extra = joblib.load(runtime / "data/extra_html_test.joblib")
    url = joblib.load(runtime / "data/content_url_test.joblib")
    assert cache["labels"] is None and extra["labels"] is None
    assert cache["urls"] == extra["urls"] == url["urls"] == load_test(runtime / "data/test.csv")
    print(json.dumps({"event": "full_inference_started", "rows": len(cache["urls"])}), flush=True)
    with threadpool_limits(limits=2):
        p, hits = predict_v15(model, cache, extra, url["features"])
    output = args.output or ROOT / "outputs" / ("retrained_submission_v17.csv" if trained else "submission_content_v17.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(runtime / "data/sample.csv", output, (p >= model["threshold"]).astype(np.int8))
    result = check_csv(output, 16000, config["expected_csv_sha256"])
    assert output.read_bytes() == (ROOT / config["expected_file"]).read_bytes()
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip():
            name, version = line.split("==")
            actual = importlib.metadata.version(name)
            assert actual == version, (name, actual, version)
            pins[name] = actual
    report = {"status": "passed", "action": args.action, "variant": "v17", "submission": result,
        "all_test_rows_replayed": True, "network_disabled": True, "historical_sources_checked": source_count,
        "payload": asset_check, "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": pins, "fresh_prefix": str(Path(sys.prefix).relative_to(ROOT)),
        "external_or_test_labels_used_for_training": False, "frozen_base_reused": trained,
        "new_classifier_fits": int(trained), "training_rows": 7119 if trained else None,
        "training_probabilities_exact": True if trained else None, "seconds": time.perf_counter() - tick}
    (ROOT / "verification").mkdir(exist_ok=True)
    (ROOT / "verification" / (args.action + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
