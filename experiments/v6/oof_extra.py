"""Fit each extra-HTML OOF model only on the original fold training rows."""

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
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit, load_frame
from experiments.v6.screen_extra import scope, sha
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.extra_text import ExtraTextModel


def parameters(kind):
    return json.loads((ROOT / f"experiments/v6/extra_{kind}_selection.json").read_text())["selected"]["model_parameters"]


def main(kind):
    cache = joblib.load(ROOT / "data/extra_html_train.joblib")
    dev = aligned_splits(ROOT, cache)["development"]
    y = cache["labels"][dev]
    old = np.load(ROOT / "reports/v5/development_oof.npz")
    np.testing.assert_array_equal(old["indices"], dev)
    np.testing.assert_array_equal(old["labels"], y)
    frame = None
    if kind == "boost":
        del cache["records"]
        gc.collect()
        original = joblib.load(ROOT / "data/content_train.joblib")
        assert original["urls"] == cache["urls"]
        np.testing.assert_array_equal(original["labels"], cache["labels"])
        frame = pd.concat([load_frame(original), cache["features"].reset_index(drop=True)], axis=1)
        del original
        gc.collect()
    config = parameters(kind)
    digest = sha(ROOT / "data/content_train.joblib")
    scoped = scope(config)
    output = ROOT / f"experiments/v6/extra_{kind}_oof"
    output.mkdir(exist_ok=True)
    predictions = np.zeros(len(dev))
    fold_ids = np.zeros(len(dev), dtype=np.int8)
    for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(dev, y)):
        started = time.perf_counter()
        path = output / f"fold_{fold}.joblib"
        expected = guard(cache, dev[tr], scoped, digest)
        print(json.dumps({"event": "extra_oof_started", "kind": kind, "fold": fold,
                          "training_rows": len(tr), "validation_rows": len(va)}), flush=True)
        if path.exists():
            model = load_component(path, expected)
        else:
            model = ExtraTextModel(**config).fit(records_at(cache, dev[tr]), y[tr]) if kind == "text" else \
                fit(frame.iloc[dev[tr]], y[tr], config, threads=4)
            save_component(path, model, expected)
        p = model.predict_proba(records_at(cache, dev[va]) if kind == "text" else frame.iloc[dev[va]])
        if kind == "boost":
            p = p[:, 1]
        predictions[va], fold_ids[va] = p, fold
        np.savez_compressed(output / f"fold_{fold}.npz", indices=dev[va], labels=y[va], probabilities=p,
                            scope_json=json.dumps(expected, sort_keys=True))
        print(json.dumps({"event": "extra_oof_fold_complete", "kind": kind, "fold": fold,
                          "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()
    cut, _ = best_threshold(y, predictions)
    result = {"kind": kind, "model_parameters": config, "metrics": measure(y, predictions, cut),
              "threshold": cut, "scope": "Three original development folds; each model fitted on that fold training only.",
              "extra_cache_sha256": scoped["extra_cache_sha256"], "external_test_labels_used": False}
    np.savez_compressed(ROOT / f"experiments/v6/extra_{kind}_development_oof.npz",
                        indices=dev, labels=y, probabilities=predictions, folds=fold_ids)
    (ROOT / f"experiments/v6/extra_{kind}_oof_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "extra_oof_complete", **result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("text", "boost"), required=True)
    with threadpool_limits(limits=2):
        main(parser.parse_args().kind)
