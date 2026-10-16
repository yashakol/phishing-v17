"""Fit one SVM coefficient model on an exactly guarded C16 representation."""

import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np

from .content_checkpoints import guard, load_component, save_component
from .content_validation import records_at
from .v9_margin import CalibratedMarginText


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def representation_snapshot(model):
    digest = hashlib.sha256()
    for name, vectorizer in model.vectorizers_.items():
        digest.update(name.encode())
        digest.update(json.dumps({key: int(value) for key, value in vectorizer.vocabulary_.items()},
                                 sort_keys=True).encode())
        digest.update(vectorizer.idf_.tobytes())
    digest.update(model.classifier_.coef_.tobytes())
    digest.update(model.classifier_.intercept_.tobytes())
    return digest.hexdigest()


def fit_margin(root, cache, indices, stage, selection):
    root = Path(root)
    indices = np.asarray(indices, dtype=np.int64)
    target = root / f"artifacts/v10/component_cache/{stage}_margin.joblib"
    if target.exists():
        raise ValueError("Refusing to overwrite an existing margin checkpoint")
    c16 = json.loads((root / "reports/v7/selection.json").read_text())
    parameters = c16["text_parameters"]
    scoped = {**parameters, "source_sha256": sha(root / "phishing_url/v7_text.py"),
              "html_text_source_sha256": sha(root / "phishing_url/html_text.py")}
    expected = guard(cache, indices, scoped, selection["train_cache_sha256"])
    tag = hashlib.sha256(json.dumps(scoped, sort_keys=True).encode()).hexdigest()[:12]
    source = root / f"artifacts/v7/component_cache/{stage}_text_{tag}.joblib"
    base = load_component(source, expected)
    if base.training_rows_ != len(indices) or base.vocabulary_fit_rows_ != len(indices) or \
            base.get_params() != parameters or base.nb or base.ratio_ is not None:
        raise ValueError("C16 representation and training scope differ")
    before, source_sha = representation_snapshot(base), sha(source)
    print(json.dumps({"event": "v10_margin_transform_started", "stage": stage,
                      "rows": len(indices)}), flush=True)
    started = time.perf_counter()
    X = base._matrix(records_at(cache, indices))
    print(json.dumps({"event": "v10_margin_fit_started", "stage": stage,
                      "shape": list(X.shape), "nnz": int(X.nnz)}), flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CalibratedMarginText(**selection["margin_parameters"]).fit_representation(
            base, X, cache["labels"][indices])
    del X
    if representation_snapshot(base) != before or sha(source) != source_sha:
        raise ValueError("Original C16 representation changed")
    config = {"margin_parameters": selection["margin_parameters"],
              "base_component": str(source.relative_to(root)), "base_component_sha256": source_sha,
              "base_representation_sha256": before, "base_guard": expected,
              "model_source_sha256": sha(root / "phishing_url/v9_margin.py"),
              "training_source_sha256": sha(Path(__file__)),
              "selection_sha256": sha(root / "reports/v10/selection.json")}
    expected_margin = guard(cache, indices, config, selection["train_cache_sha256"])
    save_component(target, model, expected_margin)
    restored = load_component(target, expected_margin)
    sample = records_at(cache, indices[:32])
    np.testing.assert_array_equal(model.predict_proba(sample), restored.predict_proba(sample))
    try:
        load_component(target, dict(expected_margin, training_rows=-1))
    except ValueError:
        pass
    else:
        raise ValueError("Invalid scope accepted")
    report = {"stage": stage, "training_rows": len(indices), "new_classifier_fits": model.classifier_fits_,
              "vocabulary_fits": 0, "features": model.n_features_, "source_component_unchanged": True,
              "wrong_guard_rejected": True, "serialization_replay_rows": 32,
              "model_sha256": sha(target), "parameters": model.get_params(),
              "warnings": [str(item.message) for item in caught],
              "seconds": time.perf_counter()-started}
    (root / f"reports/v10/{stage}_fit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"event": "v10_margin_fit_complete", **report}), flush=True)
    return model
