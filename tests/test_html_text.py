"""Contract, inert input, field bounds, and serialization for HTML text model."""

import socket
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from phishing_url.html_text import ContentTextModel


def examples():
    return [dict(url=f"https://example.test/{i}",
                 title="Library research" if i < 8 else "Account verification",
                 text="Journal articles science research library" if i < 8 else
                      "Confirm password account access verify login",
                 markup="html body article citation" if i < 8 else
                        "html body form input password submit") for i in range(16)]


@pytest.mark.parametrize("nb", [False, True])
def test_fit_serialization_and_offline_prediction(tmp_path, nb):
    records, y = examples(), np.array([0] * 8 + [1] * 8)
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
         patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS")):
        model = ContentTextModel(nb=nb, include_url=False).fit(records, y)
        p = model.predict_proba(records)
        assert p.shape == (16,) and np.isfinite(p).all()
        assert p[:8].mean() < 0.5 < p[8:].mean()
        path = tmp_path / "content.joblib"
        joblib.dump(model, path)
        np.testing.assert_array_equal(p, joblib.load(path).predict_proba(records))
        assert model.predict_proba([]).shape == (0,)
        weird = model.predict_proba([dict(url="http://127.0.0.1/", title="未知", text="",
                                          markup='<script>fetch("https://example.test/")</script>')])
        assert weird.shape == (1,) and 0 <= weird[0] <= 1
    assert clone(model).get_params() == model.get_params()


def test_field_bounds_and_missing_content():
    model = ContentTextModel(char_text_chars=100)
    doc = model._documents([dict(title="t" * 601, text="x" * 12001,
                                 markup="m" * 16001, url="u" * 4097)])
    assert len(doc["word"][0]) == 600 * 2 + 2 + 12000
    assert len(doc["char"][0]) == 600 + 1 + 100
    assert len(doc["markup"][0]) == 16000
    assert len(doc["url"][0]) == 4096
    model.fit([{}, {}, {}, {}], [0, 0, 1, 1])
    np.testing.assert_allclose(model.predict_proba([{}, {"title": None}]), [0.5, 0.5])


def test_training_vocabulary_does_not_learn_from_inference():
    model = ContentTextModel(include_char=False, include_url=False,
                              include_markup=False).fit(examples(), [0] * 8 + [1] * 8)
    vocabulary = dict(model.vectorizers_["word"].vocabulary_)
    model.predict_proba([dict(text="uniquewordunseenonthetrain")])
    assert model.vectorizers_["word"].vocabulary_ == vocabulary


def test_bounded_vocabulary_sample_and_batched_transform_are_reproducible():
    records, y = examples(), [0] * 8 + [1] * 8
    model = ContentTextModel(vocabulary_fit_rows=8, transform_batch_size=3).fit(records, y)
    assert model.vocabulary_fit_rows_ == 8
    p = model.predict_proba(records)
    model.transform_batch_size = 100
    np.testing.assert_array_equal(p, model.predict_proba(records))
    repeated = ContentTextModel(vocabulary_fit_rows=8, transform_batch_size=3).fit(records, y)
    np.testing.assert_array_equal(p, repeated.predict_proba(records))


def test_deterministic_refit_and_invalid_inputs():
    records, y = examples(), [0] * 8 + [1] * 8
    a = ContentTextModel().fit(records, y).predict_proba(records)
    b = ContentTextModel().fit(records, y).predict_proba(records)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(NotFittedError):
        ContentTextModel().predict_proba(records)
    with pytest.raises(ValueError, match="align"):
        ContentTextModel().fit(records, [0, 1])
    with pytest.raises(ValueError, match="binary"):
        ContentTextModel().fit(records, [1] * 16)
    with pytest.raises(ValueError, match="threads"):
        ContentTextModel(threads=3).fit(records, y)


def test_development_alignment_excludes_archive_and_holdout(tmp_path, monkeypatch):
    import csv
    from experiments.html_astra import screen

    reports = tmp_path / "reports"
    reports.mkdir()
    with open(reports / "clean_train.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["url", "result"])
        writer.writerows([("a", 0), ("original_holdout", 1), ("b", 1), ("c", 0)])
    old_urls = [f"u{i:03d}" for i in range(120)]
    with open(reports / "clean_train.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["url", "result"])
        writer.writerows((url, i % 2) for i, url in enumerate(old_urls))
    development = np.arange(100)[::-1]
    np.savez(reports / "split_indices.npz", development=development, holdout=np.arange(100, 120))
    urls = ["new_archive"] + old_urls[::-1]
    cache = dict(urls=urls, records=[dict(url=u) for u in urls],
                 labels=np.array([1] + [i % 2 for i in range(119, -1, -1)]))
    path = tmp_path / "content.joblib"
    joblib.dump(cache, path)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    records, y, aligned_urls, original, cache_indices, _, total, splits = screen.aligned_development(path)
    assert aligned_urls.tolist() == [old_urls[i] for i in development]
    assert y.tolist() == [i % 2 for i in development]
    assert cache_indices.tolist() == [120 - i for i in development]
    assert original.tolist() == development.tolist()
    assert total == 121 and len(records) == 100
    assert len(splits["screening_validation"]) == 20
    assert len(splits["inner_validation"]) == 12
    cache["labels"][120] = 1
    joblib.dump(cache, path)
    with pytest.raises(ValueError, match="labels disagree"):
        screen.aligned_development(path)
    cache["urls"][120] = "unmatched"
    joblib.dump(cache, path)
    with pytest.raises(ValueError, match="missing"):
        screen.aligned_development(path)


def test_shared_screen_matrix_matches_public_model():
    from experiments.html_astra.screen import fit_matrix

    records, y = examples(), np.asarray([0] * 8 + [1] * 8)
    prototype = ContentTextModel()
    x = prototype._matrix(records, fit=True)
    for nb, include_url in [(False, True), (True, True), (False, False)]:
        size = x.shape[1] - (0 if include_url else len(prototype.vectorizers_["url"].vocabulary_))
        config = dict(c=4., nb=nb, include_url=include_url)
        reused, warnings = fit_matrix(prototype, x[:, :size], y, config)
        independent = ContentTextModel(**config).fit(records, y)
        np.testing.assert_array_equal(reused.predict_proba(records), independent.predict_proba(records))
        assert not warnings


def test_screen_protocol_freezes_selection_before_validation(tmp_path, monkeypatch):
    import csv
    import json
    import sys
    from experiments.html_astra import screen
    from phishing_url.content_validation import aligned_splits

    reports = tmp_path / "reports"
    reports.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    urls = [f"https://example.test/{i}" for i in range(121)]
    labels = np.arange(121) % 2
    with open(reports / "clean_train.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["url", "result"])
        writer.writerows(zip(urls[:120], labels[:120]))
    np.savez(reports / "split_indices.npz", development=np.arange(100), holdout=np.arange(100, 120))
    records = [dict(url=url, title="library" if labels[i] == 0 else "verify password",
                    text="journal science book" if labels[i] == 0 else "login confirm password",
                    markup="tag_article" if labels[i] == 0 else "tag_form type_password")
               for i, url in enumerate(urls)]
    cache = dict(records=records, labels=labels, urls=urls)
    cache_path = tmp_path / "cache.joblib"
    joblib.dump(cache, cache_path)
    splits = aligned_splits(tmp_path, cache)
    validation_urls = {urls[i] for i in splits["screening_validation"]}
    excluded_urls = set(urls[100:])
    original_documents = ContentTextModel._documents

    def guarded_documents(self, rows):
        observed = {row["url"] for row in rows}
        assert not observed & excluded_urls
        if observed & validation_urls:
            assert (output / "selection.json").exists()
        return original_documents(self, rows)

    monkeypatch.setattr(ContentTextModel, "_documents", guarded_documents)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "OUT", output)
    monkeypatch.setattr(screen.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["screen", "--cache", str(cache_path)])
    screen.main()
    result = json.loads((output / "results.json").read_text())
    assert result["validation"]["rows"] == 20
    assert result["serialization_verified_rows"] == 20
    with np.load(output / "tune_predictions.npz") as tune, \
         np.load(output / "validation_predictions.npz") as validation:
        np.testing.assert_array_equal(tune["cache_indices"], splits["inner_validation"])
        np.testing.assert_array_equal(validation["cache_indices"], splits["screening_validation"])
        assert not set(tune["cache_indices"]) & set(validation["cache_indices"])


def test_oof_checkpoint_requires_matching_config_source_and_indices(tmp_path):
    from experiments.html_astra.run_oof import atomic_json, atomic_npz, validated_checkpoint, digest

    train, valid, labels = np.array([1, 3, 8]), np.array([0, 7]), np.array([0, 1])
    path = tmp_path / "fold_0.npz"
    atomic_npz(path, train_indices=train, indices=valid, labels=labels, probabilities=np.array([.1, .9]))
    metadata = dict(fold=0, guard_digest="guard", predictions_sha256=digest(path))
    atomic_json(tmp_path / "fold_0.json", metadata)
    p, _ = validated_checkpoint(tmp_path, 0, "guard", train, valid, labels)
    np.testing.assert_array_equal(p, [.1, .9])
    with pytest.raises(ValueError, match="guard mismatch"):
        validated_checkpoint(tmp_path, 0, "different", train, valid, labels)
    with pytest.raises(ValueError, match="alignment mismatch"):
        validated_checkpoint(tmp_path, 0, "guard", train[::-1], valid, labels)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validated_checkpoint(tmp_path, 0, "guard", train, valid, labels)


def test_shared_component_serialization_and_scope_guards(tmp_path):
    from phishing_url.content_checkpoints import guard, data_digest, save_component, load_component

    records, labels = examples(), np.array([0] * 8 + [1] * 8)
    cache = dict(records=records, labels=labels, urls=[r["url"] for r in records])
    (tmp_path / "data").mkdir()
    cache_path = tmp_path / "data/content_train.joblib"
    joblib.dump(cache, cache_path)
    checksum = data_digest(tmp_path)
    indices = np.arange(len(records))
    model = ContentTextModel().fit(records, labels)
    parameters = model.get_params()
    scope = guard(cache, indices, parameters, checksum)
    path = tmp_path / "component.joblib"
    save_component(path, model, scope)
    np.testing.assert_array_equal(load_component(path, scope).predict_proba(records), model.predict_proba(records))
    wrong_scopes = [guard(cache, indices[::-1], parameters, checksum),
                    guard(cache, indices, {**parameters, "c": 16.}, checksum),
                    guard(cache, indices, parameters, "different-data-checksum")]
    cache["labels"][0] = 1
    wrong_scopes.append(guard(cache, indices, parameters, checksum))
    for wrong in wrong_scopes:
        with pytest.raises(ValueError, match="scope differs"):
            load_component(path, wrong)
    joblib.dump(cache, cache_path)
    assert data_digest(tmp_path) != checksum
