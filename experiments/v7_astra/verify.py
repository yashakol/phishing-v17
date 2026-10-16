"""Verify the saved experiment without changing selection or fitting a model."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import time
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.content_checkpoints import data_digest, guard, load_component
from phishing_url.evaluation import measure
from experiments.v7_astra.run import (OUT, CONFIG, audit, provenance, sha,
                                     write_json, aligned_prediction, metrics)


def main():
    sys.addaudithook(audit)
    started = time.perf_counter()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["sources"] == provenance()
    assert selection["data_sha256"] == data_digest(ROOT)
    expected = json.loads((OUT / "guards/inner_full_vocabulary.json").read_text())
    fresh = guard(cache, splits["inner_train"], expected["config"], selection["data_sha256"])
    for key in fresh:
        assert fresh[key] == expected[key]
    model = load_component(OUT / "models/inner_full_vocabulary.joblib", expected)
    assert clone(model).get_params() == CONFIG
    assert model.training_rows_ == model.vocabulary_fit_rows_ == len(splits["inner_train"])
    # Independently recount 32 word document frequencies across ALL fit rows.
    # This verifies IDF population as well as the recorded fit-row counter.
    vectorizer = model.vectorizers_["word"]
    ordered = sorted(vectorizer.vocabulary_)
    terms = [ordered[i] for i in np.linspace(0, len(ordered)-1, 32, dtype=int)]
    terms_set = set(terms)
    df = {term: 0 for term in terms}
    analyzer = vectorizer.build_analyzer()
    for offset in range(0, len(splits["inner_train"]), 1024):
        records = records_at(cache, splits["inner_train"][offset:offset+1024])
        for document in model._documents(records)["word"]:
            for term in terms_set.intersection(analyzer(document)):
                df[term] += 1
    expected_idf = np.log((1+model.training_rows_)/(1+np.asarray([df[t] for t in terms])))+1
    actual_idf = np.asarray([vectorizer.idf_[vectorizer.vocabulary_[t]] for t in terms])
    np.testing.assert_allclose(actual_idf, expected_idf, rtol=1e-6, atol=1e-6)
    result = dict(idf_terms_checked=32, idf_documents_recounted=model.training_rows_,
                  idf_max_abs_error=float(np.max(abs(actual_idf-expected_idf))),
                  clone_parameters_match=True, source_hashes_match=True,
                  input_data_hash_matches=True, guard_matches_training_rows=True)
    old_model = joblib.load(ROOT / "experiments/html_astra/inner_selected.joblib")
    assert old_model.vocabulary_fit_rows_ == 10000
    result["old_vocabulary_fit_rows"] = old_model.vocabulary_fit_rows_
    result["vocabulary_comparison"] = {}
    for name, new_vectorizer in model.vectorizers_.items():
        old_terms = set(old_model.vectorizers_[name].vocabulary_)
        new_terms = set(new_vectorizer.vocabulary_)
        result["vocabulary_comparison"][name] = dict(old=len(old_terms),new=len(new_terms),
                shared=len(old_terms & new_terms),new_only=len(new_terms-old_terms),
                old_only=len(old_terms-new_terms))
    del old_model
    va = splits["inner_validation"]
    labels = cache["labels"][va]
    old = aligned_prediction("experiments/html_astra/inner_predictions.npz",va,labels)
    cat = aligned_prediction("experiments/v5/cat_inner_1.npz",va,labels)
    new = aligned_prediction("experiments/v7_astra/inner_full_vocabulary.npz",va,labels)
    inner = json.loads((OUT / "inner_results.json").read_text())
    assert metrics(labels,.5*old+.5*cat) == inner["baseline"]
    assert metrics(labels,new) == inner["new_text"]
    assert metrics(labels,.5*new+.5*cat) == inner["replace_text_half_cat"]
    result["inner_metrics_recomputed"] = True
    if (OUT / "development_oof.npz").exists():
        frozen = json.loads((OUT / "oof_selection.json").read_text())
        assert frozen["sources"] == {**provenance(),
                "experiments/v7_astra/run_oof.py":sha(OUT / "run_oof.py")}
        assert frozen["inner_results_sha256"] == sha(OUT / "inner_results.json")
        indices = splits["development"]
        y = cache["labels"][indices]
        fold_ids = np.full(len(indices), -1, dtype=np.int8)
        counts = np.zeros(len(indices), dtype=np.int8)
        for f, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(indices,y)):
            fold_ids[va] = f
            counts[va] += 1
            with np.load(OUT / f"fold_{f}.npz") as a:
                np.testing.assert_array_equal(a["train_indices"], indices[tr])
                np.testing.assert_array_equal(a["indices"], indices[va])
                np.testing.assert_array_equal(a["labels"], y[va])
            metadata = json.loads((OUT / f"fold_{f}_fit.json").read_text())
            assert metadata["model_sha256"] == sha(OUT / f"models/fold_{f}.joblib")
            assert metadata["serialization_verified_real_validation_rows"] >= 32
            scope = json.loads((OUT / f"guards/fold_{f}.json").read_text())
            fresh = guard(cache,indices[tr],dict(parameters=CONFIG,sources=frozen["sources"]),
                          selection["data_sha256"])
            for key in fresh:
                assert fresh[key] == scope[key]
        assert np.all(counts == 1)
        for path in [OUT / "development_oof.npz",
                     ROOT / "experiments/html_astra/development_oof.npz",
                     ROOT / "experiments/v5/cat_development_oof.npz"]:
            with np.load(path) as a:
                np.testing.assert_array_equal(a["indices"], indices)
                np.testing.assert_array_equal(a["labels"], y)
                np.testing.assert_array_equal(a["folds"], fold_ids)
        with np.load(OUT / "development_oof.npz") as a:
            report = json.loads((OUT / "oof_results.json").read_text())
            assert measure(y,a["probabilities"],frozen["standalone_threshold"]) == report["standalone"]
            assert measure(y,a["replacement"],frozen["threshold"]) == report["replacement"]
            assert measure(y,a["baseline"],frozen["baseline_threshold"]) == report["baseline"]
            cat = aligned_prediction("experiments/v5/cat_development_oof.npz",indices,y)
            np.testing.assert_array_equal(a["replacement"],.5*a["probabilities"]+.5*cat)
        result.update(oof_rows=len(indices), exact_three_fold_coverage=True,
                      baseline_fold_assignments_match=True, oof_metrics_recomputed=True,
                      all_training_guards_match=True, frozen_inner_weights_reproduced=True)
    result["seconds"] = time.perf_counter()-started
    write_json("verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
