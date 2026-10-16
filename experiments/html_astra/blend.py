"""Tune a numeric/content mixture on inner tune, then evaluate it once.

Numeric CSVs must have clean_index, url, label, probability columns, and use
the exact saved screen_split.npz training convention. Numeric tune predictions
must come from inner train only; validation predictions from outer train only.
No model is fit here. Supply only development predictions.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import argparse
import csv
import json
import numpy as np
from phishing_url.evaluation import best_threshold, measure


def numeric_aligned(path, content):
    with open(path, newline="") as source:
        rows = list(csv.DictReader(source))
    mapping = {int(row["clean_index"]): row for row in rows}
    indices = content["clean_indices"]
    if len(rows) != len(mapping) or set(mapping) != set(indices.tolist()):
        raise ValueError("numeric predictions must match content split exactly, without duplicate indices")
    probabilities = []
    for index, url, label in zip(indices, content["urls"], content["labels"]):
        row = mapping[int(index)]
        if row["url"] != url or int(row["label"]) != label:
            raise ValueError("URL or label alignment mismatch")
        probabilities.append(float(row["probability"]))
    probabilities = np.asarray(probabilities)
    if not (np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("invalid numeric probabilities")
    return probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-tune", required=True, type=Path)
    parser.add_argument("--numeric-validation", required=True, type=Path)
    parser.add_argument("--confirm-training-splits", required=True, action="store_true",
                        help="Acknowledge numeric fit uses the exact inner/outer train splits")
    args = parser.parse_args()
    with np.load(OUT / "tune_predictions.npz") as tune:
        numeric = numeric_aligned(args.numeric_tune, tune)
        candidates = []
        for weight in (0., .15, .3, .5, .7, .85, 1.):
            p = weight * tune["probabilities"] + (1 - weight) * numeric
            threshold, f1 = best_threshold(tune["labels"], p)
            candidates.append(dict(content_weight=weight, threshold=threshold, tune_f1=f1))
        selected = max(candidates, key=lambda row: (row["tune_f1"], -abs(row["content_weight"] - .5)))
    # Record frozen choice before opening the outer development validation file.
    (OUT / "blend_selection.json").write_text(json.dumps(dict(selected=selected, candidates=candidates), indent=2))
    with np.load(OUT / "validation_predictions.npz") as validation:
        numeric = numeric_aligned(args.numeric_validation, validation)
        p = selected["content_weight"] * validation["probabilities"] + (1 - selected["content_weight"]) * numeric
        result = dict(selected=selected, validation=measure(validation["labels"], p, selected["threshold"]))
        np.savez_compressed(OUT / "blend_validation_predictions.npz", clean_indices=validation["clean_indices"],
                            urls=validation["urls"], labels=validation["labels"], probabilities=p,
                            threshold=selected["threshold"])
    (OUT / "blend_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
