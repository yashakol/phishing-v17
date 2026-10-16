"""Prepare guarded text components after OOF and final ensemble selection.

Only fits models. Never scores holdout/domain/test or changes selection.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse
import gc
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits, threadpool_info

from phishing_url.html_text import ContentTextModel
from phishing_url.content_checkpoints import guard, data_digest, save_component, load_component
from phishing_url.content_validation import aligned_splits, records_at
from experiments.html_astra.run_oof import atomic_json
from experiments.html_astra.screen import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=10800)
    args = parser.parse_args()
    ensemble_path = ROOT / "reports/v3/ensemble_selection.json"
    deadline = time.monotonic() + args.wait_seconds
    while not ((OUT / "oof_results.json").exists() and ensemble_path.exists()):
        if time.monotonic() >= deadline:
            raise TimeoutError("OOF and final ensemble selection must finish before component fitting")
        print(json.dumps({"event": "waiting_for_oof_and_ensemble_selection"}), flush=True)
        time.sleep(30)
    selection = json.loads(ensemble_path.read_text())
    text_selection = json.loads((OUT / "selection.json").read_text())
    if float(selection["selected"]["text_weight"]) == 0:
        result = dict(status="skipped", reason="final ensemble text_weight is zero")
        atomic_json(OUT / "component_results.json", result)
        print(json.dumps(result), flush=True)
        return
    parameters = text_selection["model_parameters"]
    model_source = digest(ROOT / "phishing_url/html_text.py")
    oof_guard = json.loads((OUT / "oof/run_guard.json").read_text())
    if parameters != oof_guard["model_parameters"] or model_source != oof_guard["model_source_sha256"]:
        raise ValueError("text configuration or source changed since OOF")
    started = time.perf_counter()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, cache)
    # Retain text/URLs/labels only. No numeric frame or content hashes are needed.
    cache.pop("features", None)
    cache.pop("hashes", None)
    gc.collect()
    cache_digest = data_digest(ROOT)
    if cache_digest != oof_guard["cache_sha256"]:
        raise ValueError("content training data changed since OOF")
    with np.load(ROOT / "reports/domain_split_indices.npz") as domain:
        domain_training = splits["old_to_cache"][domain["training"]]
    training_sets = dict(holdout=splits["development"], domain=domain_training,
                         full=np.arange(len(cache["urls"])))
    results = []
    directory = OUT / "component_cache"
    directory.mkdir(exist_ok=True)
    for stage, indices in training_sets.items():
        path = directory / f"{stage}_text.joblib"
        scope = guard(cache, indices, parameters, cache_digest)
        if path.exists():
            model = load_component(path, scope)
            del model
            gc.collect()
            row = dict(stage=stage, status="validated_existing", rows=len(indices), bytes=path.stat().st_size)
        else:
            print(json.dumps({"event": "component_started", "stage": stage, "rows": len(indices)}), flush=True)
            stage_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = ContentTextModel(**parameters).fit(records_at(cache, indices), cache["labels"][indices])
            save_component(path, model, scope)
            row = dict(stage=stage, status="saved", rows=len(indices), bytes=path.stat().st_size,
                       fit_seconds=model.fit_seconds_, view_fit_seconds=model.view_fit_seconds_,
                       total_seconds=time.perf_counter() - stage_started,
                       iterations=model.classifier_.n_iter_.tolist(), warnings=[str(w.message) for w in caught])
            del model
            gc.collect()
            # Reload through the same scope validator the main training pipeline uses.
            restored = load_component(path, scope)
            if restored.get_params() != parameters:
                raise AssertionError("reloaded text component parameters disagree")
            del restored
            gc.collect()
        results.append(row)
        atomic_json(directory / f"{stage}_text.json", dict(**row, guard=scope, model_source_sha256=model_source))
        atomic_json(OUT / "component_progress.json", results)
        print(json.dumps({"event": "component_finished", **row}), flush=True)
    result = dict(status="complete", components=results, model_parameters=parameters,
                  model_source_sha256=model_source, cache_sha256=cache_digest,
                  ensemble_selection_sha256=digest(ensemble_path),
                  total_seconds=time.perf_counter() - started, threadpools=threadpool_info(),
                  scope="fit only; no holdout/domain/test scoring; added archive rows only in full stage")
    atomic_json(OUT / "component_results.json", result)
    print(json.dumps({"event": "components_complete", **result}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
