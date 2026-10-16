"""Screen bounded signature policies on original inner development only."""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from phishing_url.content_checkpoints import guard, save_component
from phishing_url.content_memory import apply_memory, fit_memory
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v7_signatures import SignatureEvidence, signature_keys
from experiments.v6.screen_extra import sha


POLICIES = [
    dict(families=["scripts", "styles"], minimum_support=3, minimum_domains=2, weight=.85, confidence_limit=.98),
    dict(families=["title_scripts", "title_styles", "title_resources"], minimum_support=3,
         minimum_domains=1, weight=.85, confidence_limit=.98),
    dict(families=["title_markup"], minimum_support=3, minimum_domains=1, weight=.85, confidence_limit=.98),
    dict(families=["scripts", "styles", "title_scripts", "title_styles", "title_resources", "title_markup"],
         minimum_support=5, minimum_domains=2, weight=.85, confidence_limit=.98),
    dict(families=["scripts", "styles", "title_scripts", "title_styles", "title_resources", "title_markup"],
         minimum_support=3, minimum_domains=1, weight=.5, confidence_limit=.98),
]


def main():
    started = time.perf_counter()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    print(json.dumps({"event": "signature_screen_loaded", "training_rows": len(tr), "validation_rows": len(va)}), flush=True)
    extra = joblib.load(ROOT / "data/extra_html_train.joblib")
    assert extra["urls"] == cache["urls"]
    np.testing.assert_array_equal(extra["labels"], cache["labels"])
    keys = [signature_keys(r, e) for r, e in zip(cache["records"], extra["records"], strict=True)]
    key_cache = {"urls": cache["urls"], "labels": cache["labels"], "keys": keys,
                 "content_sha256": sha(ROOT / "data/content_train.joblib"),
                 "extra_sha256": sha(ROOT / "data/extra_html_train.joblib"),
                 "source_sha256": sha(ROOT / "phishing_url/v7_signatures.py")}
    joblib.dump(key_cache, OUT / "training_keys.joblib", compress=3)
    del extra
    gc.collect()
    text = np.load(ROOT / "experiments/html_astra/inner_predictions.npz")
    tree = np.load(ROOT / "experiments/v5/cat_inner_1.npz")
    for table in (text, tree):
        np.testing.assert_array_equal(table["indices"], va)
        np.testing.assert_array_equal(table["labels"], cache["labels"][va])
    base = .5 * text["probabilities"] + .5 * tree["probabilities"]
    memory = fit_memory(records_at(cache, tr), [cache["hashes"][int(i)] for i in tr], cache["labels"][tr], 3, 5)
    base, _ = apply_memory(memory, records_at(cache, va), [cache["hashes"][int(i)] for i in va], base)
    y = cache["labels"][va]
    baseline_threshold, _ = best_threshold(y, base)
    baseline = measure(y, base, baseline_threshold)
    rows = []
    for number, config in enumerate(POLICIES):
        model = SignatureEvidence(**config).fit_keys([keys[int(i)] for i in tr], [cache["urls"][int(i)] for i in tr], cache["labels"][tr])
        p, hits = model.predict_keys([keys[int(i)] for i in va], base)
        cut, _ = best_threshold(y, p)
        prediction = p >= cut
        previous = base >= baseline_threshold
        row = {"number": number, "config": config, "metrics": measure(y, p, cut),
               "metrics_at_frozen_base_threshold": measure(y, p, baseline_threshold),
               "matches": int(hits["signature_match"].sum()), "conflicts": int(hits["signature_conflict"].sum()),
               "evidence_groups": len(model.evidence_), "training_conflicts": model.conflicting_groups_,
               "fixed_errors": int(np.sum((previous != y) & (prediction == y))),
               "new_errors": int(np.sum((previous == y) & (prediction != y)))}
        rows.append(row)
        scoped = {**config, "signature_source_sha256": key_cache["source_sha256"], "extra_sha256": key_cache["extra_sha256"]}
        save_component(OUT / f"inner_{number}.joblib", model, guard(cache, tr, scoped, key_cache["content_sha256"]))
        np.savez_compressed(OUT / f"inner_{number}.npz", indices=va, labels=y, probabilities=p, base=base)
        print(json.dumps({"event": "signature_inner_candidate", **row}), flush=True)
    selected = max(rows, key=lambda r: (r["metrics"]["f1"], -r["new_errors"]))
    useful = selected["metrics"]["f1"] > baseline["f1"] + 1e-12
    report = {"baseline": baseline, "selected": selected, "useful": useful, "candidates": rows,
              "scope": "Original inner development only. No holdout, additional train or external/test labels.",
              "seconds": time.perf_counter() - started}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "signature_screen_complete", "baseline": baseline, "selected": selected, "useful": useful,
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

