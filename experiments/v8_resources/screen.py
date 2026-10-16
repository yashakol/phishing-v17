"""One fixed offline resource-character model, screened on inner development."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.v6.screen_extra import sha
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v8_resource_text import ResourceCharacterTextModel

CONFIG = dict(c=4., min_df=2, threads=2, random_state=20261008,
              transform_batch_size=2048, verbose=True)


def scope():
    return {"parameters": CONFIG, "extra_cache_sha256": sha(ROOT / "data/extra_html_train.joblib"),
            "source_sha256": sha(ROOT / "phishing_url/v8_resource_text.py"),
            "html_text_source_sha256": sha(ROOT / "phishing_url/html_text.py")}


def main(refit=False):
    out = ROOT / "experiments/v8_resources"
    out.mkdir(exist_ok=True)
    protocol = {"parameters": CONFIG, "blending_weights": [0., .1, .25],
                "gate": "At least one fewer inner blend classification error, not F1 denominator only",
                "training_scope": "inner train only, with all its rows for vocabulary/IDF",
                "forbidden": "test/holdout labels, public reference and live URLs",
                "source_sha256": scope()}
    (out / "preregistered.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cache = joblib.load(ROOT / "data/extra_html_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    y = cache["labels"][va]
    with np.load(ROOT / "experiments/v7_astra/inner_full_vocabulary.npz") as text, \
         np.load(ROOT / "experiments/v5/cat_inner_1.npz") as cat:
        for table in (text, cat):
            np.testing.assert_array_equal(table["indices"], va)
            np.testing.assert_array_equal(table["labels"], y)
        base = .5 * text["probabilities"] + .5 * cat["probabilities"]
    cut, _ = best_threshold(y, base)
    baseline = measure(y, base, cut)
    path = out / "inner.joblib"
    expected = guard(cache, tr, scope(), sha(ROOT / "data/content_train.joblib"))
    started = time.perf_counter()
    if path.exists() and not refit:
        model = load_component(path, expected)
    else:
        print(json.dumps({"event": "v8_resource_fit_started", "rows": len(tr)}), flush=True)
        model = ResourceCharacterTextModel(**CONFIG).fit(records_at(cache, tr), cache["labels"][tr])
        assert model.vocabulary_fit_rows_ == len(tr)
        save_component(path, model, expected)
    records = records_at(cache, va)
    p = model.predict_proba(records)
    restored = load_component(path, expected)
    np.testing.assert_array_equal(p[:32], restored.predict_proba(records[:32]))
    results = []
    for weight in (0., .1, .25):
        blended = (1-weight) * base + weight * p
        threshold, _ = best_threshold(y, blended)
        results.append({"weight": weight, "threshold": threshold, "metrics": measure(y, blended, threshold)})
    chosen = max(results, key=lambda row: row["metrics"]["f1"])
    threshold, _ = best_threshold(y, p)
    report = {"selected": chosen, "baseline": baseline, "candidates": results,
              "standalone": measure(y, p, threshold), "model_parameters": CONFIG,
              "useful": chosen["metrics"]["fp"]+chosen["metrics"]["fn"] < baseline["fp"]+baseline["fn"],
              "seconds": time.perf_counter()-started, "guard": expected,
              "serialization_verified_real_rows": 32, "external_test_labels_used": False}
    np.savez_compressed(out / "inner_predictions.npz", indices=va, labels=y, probabilities=p, baseline=base)
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "v8_resource_screened", **report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true")
    with threadpool_limits(limits=2):
        main(parser.parse_args().refit)
