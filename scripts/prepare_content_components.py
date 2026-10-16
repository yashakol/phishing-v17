"""Fit numeric components in advance; never score holdout before selection."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.content.boost import load_frame, fit
from phishing_url.content_checkpoints import guard, data_digest, save_component, load_component
from phishing_url.content_validation import aligned_splits


def main():
    cache = joblib.load(ROOT / "data/content_train.joblib")
    frame = load_frame(cache)
    splits = aligned_splits(ROOT, cache)
    # The numeric components need no text during fitting.
    del cache["records"], cache["hashes"]
    numeric = json.loads((ROOT / "experiments/content/selection.json").read_text())["selected"]["config"]
    digest = data_digest(ROOT)
    domain = np.load(ROOT / "reports/domain_split_indices.npz")
    training_sets = {"holdout": splits["development"], "domain": splits["old_to_cache"][domain["training"]],
                     "full": np.arange(len(cache["urls"]))}
    for name, indices in training_sets.items():
        path = ROOT / f"artifacts/v3/component_cache/{name}_boost.joblib"
        scope = guard(cache, indices, numeric, digest)
        if path.exists():
            load_component(path, scope)
            print("Validated existing numeric component:", name, flush=True)
            continue
        start = time.perf_counter()
        print("Fitting numeric component:", name, len(indices), flush=True)
        model = fit(frame.iloc[indices], cache["labels"][indices], numeric)
        save_component(path, model, scope)
        del model
        print("Saved numeric component:", name, "seconds", round(time.perf_counter()-start,1), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
