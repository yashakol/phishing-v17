"""Tests protect data integrity, split boundaries, URL parsing and inference."""

import csv
import socket
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.metrics import f1_score

from phishing_url.data import load_test, load_train, write_submission
from phishing_url.evaluation import best_threshold
from phishing_url.features import feature_row, group_for_url, make_features
from phishing_url.models import TextModel


def test_damaged_rows_duplicates_and_conflicts(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text('url,result\nhttps://a.example/,0\nhttps://a.example/,0\n<html>\nhttps://b.example/,1\nhttps://b.example/,0\n.com/Login.php,1\n', encoding="utf-8")
    urls, labels, audit = load_train(path, tmp_path / "report")
    assert urls == ["https://a.example/"]
    assert labels.tolist() == [0]
    assert audit["duplicate_rows_removed"] == 1
    assert audit["conflicting_urls_removed"] == 1
    assert audit["rejected_records"] == 2


def test_invalid_test_rows_cannot_be_silently_dropped(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("url\nhttps://a.example/\n<html>\n")
    with pytest.raises(ValueError, match="cannot discard"):
        load_test(path)


def test_features_do_not_access_network():
    with patch.object(socket, "create_connection", side_effect=AssertionError("network access")), patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS access")):
        assert group_for_url("https://x.sub.example.co.uk/a") == "example.co.uk"
        assert group_for_url("https://alice.github.io/") != group_for_url("https://bob.github.io/")
        row = feature_row("http://user:pass@192.0.2.1:8080/login?a=%40#fragment")
    assert row["is_ip"] == 1 and row["has_credentials"] == 1
    assert row["port"] == 8080 and row["has_query"] == 1
    assert row["keyword_login"] == 1


@pytest.mark.parametrize("url", ["https://[invalid", "https://пример.рф/a", "https://[2001:db8::1]/", "https://example.com:wrong/"])
def test_unusual_urls_produce_finite_numeric_features(url):
    row = feature_row(url)
    assert all(np.isfinite(value) for key, value in row.items() if not key.startswith("cat_"))


def test_threshold_matches_exhaustive_f1():
    labels = np.array([0, 1, 1, 0, 1, 0])
    probabilities = np.array([0.1, 0.4, 0.6, 0.4, 0.9, 0.8])
    threshold, score = best_threshold(labels, probabilities)
    maximum = max(f1_score(labels, probabilities >= t) for t in np.unique(probabilities))
    assert score == pytest.approx(maximum)
    assert f1_score(labels, probabilities >= threshold) == pytest.approx(score)


def test_submission_preserves_sample_order_and_rejects_bad_labels(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("Id,Predicted\n0,1\n1,1\n2,1\n")
    target = tmp_path / "answer.csv"
    info = write_submission(sample, target, np.array([1, 0, 1]))
    assert info["rows"] == 3
    assert target.read_text() == "Id,Predicted\n0,1\n1,0\n2,1\n"
    with pytest.raises(ValueError, match="counts differ"):
        write_submission(sample, target, [1, 0])
    with pytest.raises(ValueError, match="binary"):
        write_submission(sample, target, [1, 0, 2])


def test_text_model_save_load_and_unseen_urls(tmp_path):
    safe = [f"https://library.example.edu/article/history-{i}" for i in range(12)]
    bad = [f"http://verify-account-{i}.example/login/password" for i in range(12)]
    urls, labels = safe + bad, np.array([0] * 12 + [1] * 12)
    for kind in ["tfidf", "nb"]:
        model = TextModel(kind).fit(urls, labels)
        assert model.predict_proba(safe).mean() < model.predict_proba(bad).mean()
        saved = tmp_path / f"{kind}.joblib"
        joblib.dump(model, saved)
        loaded = joblib.load(saved)
        assert np.array_equal(model.predict_proba(urls), loaded.predict_proba(urls))
        probabilities = loaded.predict_proba(["https://completely-unseen.example/a"])
        assert len(probabilities) == 1 and 0 <= probabilities[0] <= 1
