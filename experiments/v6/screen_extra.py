"""Choose extra HTML candidates on inner development only, before any holdout."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit, load_frame
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.extra_text import ExtraTextModel


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scope(config):
    return {**config, "extra_cache_sha256": sha(ROOT / "data/extra_html_train.joblib"),
            "source_sha256": sha(ROOT / "phishing_url/extra_text.py"),
            "base_text_source_sha256": sha(ROOT / "phishing_url/html_text.py"),
            "extractor_sha256": sha(ROOT / "phishing_url/extra_html.py")}


def main(kind):
    out = ROOT / "experiments/v6"
    cache = joblib.load(ROOT / "data/extra_html_train.joblib")
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    text = np.load(ROOT / "experiments/html_astra/inner_predictions.npz")
    tree = np.load(ROOT / "experiments/v5/cat_inner_1.npz")
    for table in (text, tree):
        np.testing.assert_array_equal(table["indices"], va)
        np.testing.assert_array_equal(table["labels"], cache["labels"][va])
    base = .5 * text["probabilities"] + .5 * tree["probabilities"]
    y = cache["labels"][va]
    cut, _ = best_threshold(y, base)
    baseline = measure(y, base, cut)
    frame = None
    digest = sha(ROOT / "data/content_train.joblib")
    if kind == "boost":
        del cache["records"]
        gc.collect()
        original = joblib.load(ROOT / "data/content_train.joblib")
        assert original["urls"] == cache["urls"]
        np.testing.assert_array_equal(original["labels"], cache["labels"])
        frame = pd.concat([load_frame(original), cache["features"].reset_index(drop=True)], axis=1)
        del original
        gc.collect()
    candidates = [dict(c=4., nb=False, threads=2), dict(c=16., nb=False, threads=2)] if kind == "text" else [
        dict(iterations=3000, depth=8, learning_rate=.045, l2_leaf_reg=5)]
    results = []
    for n, config in enumerate(candidates):
        started = time.perf_counter()
        print(json.dumps({"event": "extra_inner_started", "kind": kind, "config": config}), flush=True)
        checkpoint = out / f"extra_{kind}_inner_{n}.joblib"
        expected = guard(cache, tr, scope(config), digest)
        if checkpoint.exists():
            model = load_component(checkpoint, expected)
        elif kind == "text":
            model = ExtraTextModel(**config).fit(records_at(cache, tr), cache["labels"][tr])
            save_component(checkpoint, model, expected)
        else:
            model = fit(frame.iloc[tr], cache["labels"][tr], config, (frame.iloc[va], y), threads=4)
            save_component(checkpoint, model, expected)
        p = model.predict_proba(records_at(cache, va) if kind == "text" else frame.iloc[va])
        if kind == "boost":
            p = p[:, 1]
        cut, _ = best_threshold(y, p)
        mixtures = []
        for weight in (0., .1, .25, .4, .6):
            combined = (1 - weight) * base + weight * p
            cut_mix, _ = best_threshold(y, combined)
            mixtures.append({"extra_weight": weight, "metrics": measure(y, combined, cut_mix), "threshold": cut_mix})
        if kind == "boost":
            config = {**config, "iterations": int(model.tree_count_)}
        result = {"kind": kind, "candidate": n, "model_parameters": config,
                  "standalone": measure(y, p, cut), "best_blend": max(mixtures, key=lambda row: row["metrics"]["f1"]),
                  "blends": mixtures, "seconds": time.perf_counter() - started}
        np.savez_compressed(out / f"extra_{kind}_inner_{n}.npz", indices=va, labels=y, probabilities=p)
        (out / f"extra_{kind}_inner_{n}.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(json.dumps({"event": "extra_inner_complete", **{key: value for key, value in result.items() if key != "blends"}}), flush=True)
        del model
        gc.collect()
    selected = max(results, key=lambda row: row["best_blend"]["metrics"]["f1"])
    report = {"selected": selected, "baseline": baseline, "candidates": results,
              "scope": "Inner development only; no holdout, added-row or external test labels.",
              "extra_cache_sha256": sha(ROOT / "data/extra_html_train.joblib")}
    (out / f"extra_{kind}_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "extra_screen_complete", "kind": kind, "selected": selected}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("text", "boost"), required=True)
    with threadpool_limits(limits=2):
        main(parser.parse_args().kind)
