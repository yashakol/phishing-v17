"""Finish the saved screen after an estimator-interface fix; zero new fits."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.v8_astra_regularization.resume import checked_base, prediction, scope, sha
from experiments.v9_margin.screen import metrics
from phishing_url.content_checkpoints import data_digest, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import measure
from phishing_url.v9_margin import CalibratedMarginText


def main():
    assert not (OUT / "selection.json").exists(), "Complete failed interface check only"
    protocol = json.loads((OUT / "preregistered.json").read_text())
    assert protocol["parameters"] == CalibratedMarginText().get_params()
    assert sha(OUT / "source_before_adapter.py") == protocol["model_source_sha256"]
    assert sha(OUT / "screen.py") == protocol["source_sha256"]
    recovery = {"reason": "check_is_fitted requires a public fit method; added guarded fit adapter only",
        "mathematical_training_change": False, "additional_classifier_fits": 0,
        "original_fitted_checkpoint_sha256": sha(OUT / "inner.joblib"),
        "original_model_source_sha256": protocol["model_source_sha256"],
        "repaired_model_source_sha256": sha(ROOT / "phishing_url/v9_margin.py"),
        "source_before_adapter": "experiments/v9_margin/source_before_adapter.py",
        "recovery_source_sha256": sha(Path(__file__))}
    (OUT / "recovery.json").write_text(json.dumps(recovery, indent=2)+"\n")
    digest = data_digest(ROOT)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    assert len(tr) == 34663 and len(va) == 6118 and not set(tr) & set(va)
    base, origin = checked_base(cache, tr, "inner_full_vocabulary", digest)
    expected = scope(cache, tr, {"parameters": protocol["parameters"], "representation_origin": origin,
        "model_source_sha256": protocol["model_source_sha256"], "screen_source_sha256": protocol["source_sha256"],
        "protocol_sha256": sha(OUT / "preregistered.json")}, digest)
    model = load_component(OUT / "inner.joblib", expected)
    assert model.get_params() == protocol["parameters"] and model.classifier_fits_ == 4
    assert model.training_rows_ == len(tr)
    for name, vectorizer in base.vectorizers_.items():
        other = model.transformer_.vectorizers_[name]
        assert vectorizer.vocabulary_ == other.vocabulary_
        np.testing.assert_array_equal(vectorizer.idf_, other.idf_)
    svm = model.classifier_.calibrated_classifiers_[0].estimator
    assert svm.C == .5 and svm.random_state == 20261008
    assert np.isfinite(svm.coef_).all() and np.isfinite(svm.intercept_).all()
    old = prediction("experiments/v7_astra/inner_full_vocabulary.npz", va, cache["labels"][va], tr)
    cat = prediction("experiments/v5/cat_inner_1.npz", va, cache["labels"][va])
    y = cache["labels"][va]
    baseline = metrics(y, .5*old+.5*cat)
    assert baseline["errors"] == 51
    before = (.5*old+.5*cat) >= baseline["threshold"]
    p = model.predict_proba(records_at(cache, va))
    migrated = scope(cache, tr, {"parameters": protocol["parameters"], "representation_origin": origin,
        "model_source_sha256": recovery["repaired_model_source_sha256"],
        "fit_model_source_sha256": protocol["model_source_sha256"],
        "recovery_source_sha256": recovery["recovery_source_sha256"],
        "protocol_sha256": sha(OUT / "preregistered.json")}, digest)
    save_component(OUT / "inner_recovered.joblib", model, migrated)
    restored = load_component(OUT / "inner_recovered.joblib", migrated)
    np.testing.assert_array_equal(p[:32], restored.predict_proba(records_at(cache, va[:32])))
    rows = []
    for weight in protocol["blends"]:
        mixed = .5*((1-weight)*old+weight*p)+.5*cat
        candidate = metrics(y, mixed)
        pred = mixed >= candidate["threshold"]
        frozen = measure(y, mixed, baseline["threshold"])
        net_frozen = baseline["errors"]-frozen["fp"]-frozen["fn"]
        rows.append({"text_replacement": weight, "metrics": candidate,
            "metrics_at_baseline_threshold": frozen, "frozen_net": net_frozen,
            "fixed_errors": int(((before != y) & (pred == y)).sum()),
            "new_errors": int(((before == y) & (pred != y)).sum()),
            "gate_passed": bool(weight > 0 and candidate["errors"] <= 49 and net_frozen >= 2 and candidate["f1"] > baseline["f1"])})
    eligible = [r for r in rows if r["gate_passed"]]
    selected = max(eligible if eligible else rows, key=lambda r: r["metrics"]["f1"])
    report = {"gate_passed": bool(eligible), "baseline": baseline, "standalone": metrics(y, p),
        "selected": selected, "candidates": rows, "parameters": protocol["parameters"], "protocol": protocol,
        "classifier_fits": 4, "additional_classifier_fits_in_recovery": 0,
        "convergence_warning_log": "screen_run.log", "serialization_rows": 32, "data_sha256": digest,
        "origin": origin, "model": "experiments/v9_margin/inner_recovered.joblib", "recovery": recovery,
        "external_test_labels_used": False}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2)+"\n")
    (OUT / "inner_scope.json").write_text(json.dumps(migrated, indent=2)+"\n")
    np.savez_compressed(OUT / "inner_predictions.npz", train_indices=tr, indices=va, labels=y, probabilities=p)
    print(json.dumps({"event": "margin_screen_recovered", **report}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
