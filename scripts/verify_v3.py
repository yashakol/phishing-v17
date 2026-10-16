"""Offline replay, ordered submission checks and validation metric recovery."""

import csv
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from phishing_url.content import extract_html
from phishing_url.content_bundle import predict_content
from phishing_url.content_validation import aligned_splits
from phishing_url.data import load_test, write_submission
from phishing_url.evaluation import measure


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    reports = ROOT / "reports/v3"
    metrics = json.loads((reports / "metrics.json").read_text())
    train = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, train)
    dev, hold, added = [set(splits[name].tolist()) for name in ("development", "holdout", "added")]
    assert not dev & hold and not dev & added and not hold & added
    assert dev | hold | added == set(range(len(train["urls"])))
    del train
    recovered = {}
    for name, key in (("holdout", "holdout"), ("domain", "domain_disjoint")):
        table = pd.read_csv(reports / f"{name}_predictions.csv")
        score = measure(table.label.to_numpy(), table.probability.to_numpy(), metrics["config"]["selected"]["threshold"])
        for field in ("f1", "precision", "recall", "tn", "fp", "fn", "tp", "rows"):
            assert np.isclose(score[field], metrics[key]["metrics"][field], rtol=0, atol=1e-12)
        expected = table.probability.to_numpy() >= score["threshold"]
        np.testing.assert_array_equal(table.prediction.to_numpy(), expected)
        recovered[name] = score
    cache = joblib.load(ROOT / "data/content_test.joblib")
    url = joblib.load(ROOT / "data/content_url_test.joblib")
    original_order = load_test(ROOT / "data/test.csv")
    assert cache["urls"] == original_order == url["urls"]
    assert cache["labels"] is None
    model_path = ROOT / "artifacts/v3/content.joblib"
    bundle = joblib.load(model_path)
    assert "external_reference" not in bundle
    assert bundle["training_rows"] == 63900
    with patch("socket.create_connection", side_effect=AssertionError("No network allowed")), patch("socket.getaddrinfo", side_effect=AssertionError("No DNS allowed")):
        p, hits = predict_content(bundle, cache, url["features"])
        # Recheck the parser's inert treatment of a script and external resources.
        record, _ = extract_html("https://example.invalid/", '<html><p>hello</p><script src="https://other.invalid/x.js">window.location="https://other.invalid";</script></html>')
        assert "window.location" not in record["text"]
    output = reports / "reproduced_submission.csv"
    report = write_submission(ROOT / "data/sample.csv", output, (p >= bundle["threshold"]).astype(np.int8))
    submitted = ROOT / "submissions/submission_content_v3.csv"
    assert submitted.read_bytes() == output.read_bytes()
    with submitted.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Id", "Predicted"]
        rows = list(reader)
    assert len(rows) == len(original_order) == 16000
    assert [r["Id"] for r in rows] == [str(i) for i in range(16000)]
    assert all(r["Predicted"] in {"0", "1"} for r in rows)
    for i, u in enumerate(original_order):
        if u in bundle["known_urls"]:
            assert int(rows[i]["Predicted"]) == bundle["known_urls"][u]
    result = {"ordered_rows": 16000, "byte_identical_replay": True, "offline_inference": True,
              "external_reference_used": False, "splits_disjoint": True, "extra_train_rows_excluded_from_validation": len(added),
              "known_training_urls": int(hits["known_url"].sum()), "training_rule_matches": int(hits["training_match"].sum()),
              "submission": report, "recovered_validation": recovered,
              "model_sha256": digest(model_path),
              "train_cache_sha256": digest(ROOT / "data/content_train.joblib"),
              "test_cache_sha256": digest(ROOT / "data/content_test.joblib")}
    (reports / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
