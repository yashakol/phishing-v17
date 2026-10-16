"""Select stacker using only honest base predictions on inner development.

The inner text/tree models never saw any meta-training labels. A second
development screening split is evaluated with the already saved outer models.
No holdout or test data/labels are loaded.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"

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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.content.boost import load_frame, fit
from experiments.v6.screen_extra import sha
from phishing_url.content_validation import aligned_splits
from phishing_url.content_checkpoints import guard, load_component, save_component
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v7_stacking import PageStacker, stacking_frame


CONFIGURATIONS = [dict(kind="linear", c=.5), dict(kind="linear", c=2.), dict(kind="tree", depth=3, iterations=400)]


def main():
    started = time.perf_counter()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    frame = load_frame(cache)
    splits = aligned_splits(ROOT, cache)
    del cache["records"], cache["hashes"]
    gc.collect()
    va = splits["inner_validation"]
    text = np.load(ROOT / "experiments/html_astra/inner_predictions.npz")
    tree = np.load(ROOT / "experiments/v5/cat_inner_1.npz")
    for p in (text, tree):
        np.testing.assert_array_equal(p["indices"], va)
        np.testing.assert_array_equal(p["labels"], cache["labels"][va])
    y = cache["labels"][va]
    x = stacking_frame(text["probabilities"], tree["probabilities"], frame.iloc[va])
    np.savez_compressed(OUT / "training_input.npz", indices=va, labels=y,
                        text=text["probabilities"], numeric=tree["probabilities"], frame=x.to_numpy(), columns=np.asarray(x.columns))
    baseline_rows = []
    for weight in (.3, .5, .7, .85):
        p = weight * text["probabilities"] + (1-weight) * tree["probabilities"]
        cut, _ = best_threshold(y, p)
        baseline_rows.append({"weight": weight, "metrics": measure(y, p, cut)})
    baseline = max(baseline_rows, key=lambda r: r["metrics"]["f1"])
    baseline_prob = baseline["weight"] * text["probabilities"] + (1-baseline["weight"]) * tree["probabilities"]
    configurations = []
    for number, config in enumerate(CONFIGURATIONS):
        p = np.empty(len(y))
        for fold, (tr, val) in enumerate(StratifiedKFold(3, shuffle=True, random_state=20261008).split(x, y)):
            model = PageStacker(**config).fit(x.iloc[tr], y[tr])
            p[val] = model.predict_proba(x.iloc[val])
        candidates = []
        for weight in (0., .25, .5, 1.):
            mixed = (1-weight) * baseline_prob + weight * p
            cut, _ = best_threshold(y, mixed)
            candidates.append({"stack_weight": weight, "metrics": measure(y, mixed, cut)})
        best = max(candidates, key=lambda r: (r["metrics"]["f1"], -r["stack_weight"]))
        row = {"number": number, "config": config, "selected_blend": best, "blends": candidates}
        configurations.append(row)
        np.savez_compressed(OUT / f"inner_oof_{number}.npz", indices=va, labels=y, probabilities=p)
        print(json.dumps({"event": "stack_inner_complete", **row}), flush=True)
    selected = max(configurations, key=lambda r: (r["selected_blend"]["metrics"]["f1"], -r["selected_blend"]["stack_weight"]))
    useful = selected["selected_blend"]["metrics"]["f1"] > baseline["metrics"]["f1"] + 1e-12
    result = {"baseline": baseline, "selected": selected, "candidates": configurations, "useful": useful,
              "scope": "Meta OOF on inner validation whose base models fit only inner train. No holdout/test labels.",
              "source_sha256": sha(ROOT / "phishing_url/v7_stacking.py")}
    (OUT / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    model = PageStacker(**selected["config"]).fit(x, y)
    joblib.dump(model, OUT / "inner_stacker.joblib", compress=3)
    restored = joblib.load(OUT / "inner_stacker.joblib")
    np.testing.assert_array_equal(model.predict_proba(x.iloc[:32]), restored.predict_proba(x.iloc[:32]))
    if useful:
        screening = splits["screening_validation"]
        text = np.load(ROOT / "experiments/html_astra/screen_predictions.npz")
        np.testing.assert_array_equal(text["indices"], screening)
        np.testing.assert_array_equal(text["labels"], cache["labels"][screening])
        config = json.loads((ROOT / "experiments/v5/cat_selection.json").read_text())["selected"]["config"]
        training = splits["screening_train"]
        expected = guard(cache, training, config, sha(ROOT / "data/content_train.joblib"))
        path = OUT / "screen_numeric.joblib"
        print(json.dumps({"event": "stack_screen_numeric_started", "rows": len(training), "config": config}), flush=True)
        if path.exists():
            numeric_model = load_component(path, expected)
        else:
            numeric_model = fit(frame.iloc[training], cache["labels"][training], config, threads=2)
            save_component(path, numeric_model, expected)
        numeric = numeric_model.predict_proba(frame.iloc[screening])[:, 1]
        np.savez_compressed(OUT / "screen_numeric.npz", indices=screening, labels=cache["labels"][screening], probabilities=numeric)
        sx = stacking_frame(text["probabilities"], numeric, frame.iloc[screening])
        base = baseline["weight"] * text["probabilities"] + (1-baseline["weight"]) * numeric
        sw = selected["selected_blend"]["stack_weight"]
        p = (1-sw) * base + sw * model.predict_proba(sx)
        cut = selected["selected_blend"]["metrics"]["threshold"]
        result["screen"] = measure(cache["labels"][screening], p, cut)
        result["screen_baseline"] = measure(cache["labels"][screening], base, baseline["metrics"]["threshold"])
        np.savez_compressed(OUT / "screen_predictions.npz", indices=screening, labels=cache["labels"][screening], probabilities=p)
    result["seconds"] = time.perf_counter() - started
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "stack_screen_complete", "baseline": baseline, "selected": selected,
                      "useful": useful, "screen": result.get("screen"), "screen_baseline": result.get("screen_baseline"),
                      "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
