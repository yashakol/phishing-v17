"""Refit only logistic coefficients on a guarded full-vocabulary training model."""

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import records_at


def semantic_snapshot(model):
    result = hashlib.sha256()
    for name, vectorizer in model.vectorizers_.items():
        result.update(name.encode())
        vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
        result.update(json.dumps(vocabulary, sort_keys=True).encode())
        result.update(vectorizer.idf_.tobytes())
    result.update(model.classifier_.coef_.tobytes())
    result.update(model.classifier_.intercept_.tobytes())
    return result.hexdigest()


def refit_coefficients(base, cache, indices, parameters, path, data_sha256, source_sha256, refit=False, reuse_only=False):
    if base.vocabulary_fit_rows_ != len(indices):
        raise ValueError("Vocabulary training rows differ from coefficient training rows")
    before = semantic_snapshot(base)
    scope = {"parameters": parameters, "representation": "existing guarded c16 vocabulary/IDF, float32 transform",
             "base_model_semantics_sha256": before, "source_sha256": source_sha256,
             "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    expected = guard(cache, indices, scope, data_sha256)
    path = Path(path)
    if reuse_only and not path.exists():
        raise FileNotFoundError(f"Missing trained coefficient checkpoint: {path}")
    if path.exists() and not refit:
        return load_component(path, expected)
    model = copy.deepcopy(base)
    matrix = model._matrix(records_at(cache, indices))
    classifier = LogisticRegression(C=parameters["c"], solver="liblinear", max_iter=parameters["max_iter"],
                                    random_state=parameters["random_state"])
    started = time.perf_counter()
    classifier.fit(matrix, cache["labels"][indices])
    model.classifier_ = classifier
    model.classes_ = classifier.classes_
    model.c = parameters["c"]
    model.training_rows_ = len(indices)
    model.regularization_fit_seconds_ = time.perf_counter()-started
    assert model.get_params() == parameters
    assert semantic_snapshot(base) == before
    save_component(path, model, expected)
    return model
