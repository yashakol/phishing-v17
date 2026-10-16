"""One preregistered full-vocabulary candidate, inner selection then optional OOF."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse
import gc
import hashlib
import importlib.metadata
import json
import resource
import time
import warnings
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_info, threadpool_limits
from phishing_url.content_checkpoints import data_digest, guard, save_component, load_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v7_text import FullVocabularyTextModel

CONFIG = FullVocabularyTextModel().get_params()
WEIGHTS = (0.1, 0.25, 0.5)
READS = set()


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo")):
        raise PermissionError("This experiment is offline")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return
    if (relative.startswith(("external/", "reports/v2/")) or
            relative == "submissions/submission_reference.csv" or
            "holdout" in relative.lower() or "test" in path.name.lower()):
        raise PermissionError(f"Forbidden input: {relative}")
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing and not relative.startswith("experiments/v7_astra/"):
        raise PermissionError(f"Write outside experiment: {relative}")
    if not writing:
        READS.add(relative)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def aligned_prediction(path, indices, labels):
    with np.load(ROOT / path, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["indices"], indices)
        np.testing.assert_array_equal(a["labels"], labels)
        p = a["probabilities"].copy()
    assert p.shape == labels.shape and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def metrics(y, p):
    threshold, _ = best_threshold(y, p)
    return dict(tuned=measure(y, p, threshold), at_05=measure(y, p))


def provenance():
    paths = ["phishing_url/v7_text.py", "phishing_url/html_text.py",
             "phishing_url/content_checkpoints.py", "phishing_url/content_validation.py",
             "phishing_url/evaluation.py", "experiments/v7_astra/run.py",
             "reports/clean_train.csv", "reports/split_indices.npz"]
    return {p: sha(ROOT / p) for p in paths}


def fit_one(cache, tr, va, name, digest, sources):
    started = time.perf_counter()
    assert len(set(tr) & set(va)) == 0
    config_scope = dict(parameters=CONFIG, sources=sources)
    expected = guard(cache, tr, config_scope, digest)
    expected["ordered_training_indices_sha256"] = hashlib.sha256(tr.tobytes()).hexdigest()
    model_path = OUT / "models" / f"{name}.joblib"
    model_path.parent.mkdir(exist_ok=True)
    emit("fit_started", name=name, train_rows=len(tr), validation_rows=len(va), threads=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = FullVocabularyTextModel(**CONFIG).fit(records_at(cache, tr), cache["labels"][tr])
    assert model.training_rows_ == len(tr) == model.vocabulary_fit_rows_
    predict_started = time.perf_counter()
    p = model.predict_proba(records_at(cache, va))
    predict_seconds = time.perf_counter() - predict_started
    assert p.shape == (len(va),) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    save_component(model_path, model, expected)
    restored = load_component(model_path, expected)
    np.testing.assert_array_equal(p[:32], restored.predict_proba(records_at(cache, va[:32])))
    # A deliberately wrong training scope must never be accepted.
    rejected = False
    try:
        load_component(model_path, dict(expected, training_rows=-1))
    except ValueError:
        rejected = True
    assert rejected
    result = dict(name=name, fit_seconds=model.fit_seconds_,
                  view_fit_seconds=model.view_fit_seconds_, predict_seconds=predict_seconds,
                  total_seconds=time.perf_counter()-started,
                  vocabulary_fit_rows=model.vocabulary_fit_rows_,
                  dimensions={k: len(v.vocabulary_) for k, v in model.vectorizers_.items()},
                  iterations=model.classifier_.n_iter_.tolist(),
                  warnings=[str(w.message) for w in caught],
                  serialization_verified_real_validation_rows=32,
                  serialization_max_abs_error=0., wrong_guard_rejected=rejected,
                  model_sha256=sha(model_path), model_bytes=model_path.stat().st_size,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  threadpools=threadpool_info())
    assert all(pool["num_threads"] <= 2 for pool in result["threadpools"])
    np.savez_compressed(OUT / f"{name}.npz", train_indices=tr, indices=va,
                        labels=cache["labels"][va], probabilities=p)
    write_json(f"guards/{name}.json", expected)
    write_json(f"{name}_fit.json", result)
    emit("fit_finished", **result)
    del model, restored
    gc.collect()
    return p, result


def inner(cache, splits, digest, sources):
    tr, va = splits["inner_train"], splits["inner_validation"]
    y = cache["labels"][va]
    text = aligned_prediction("experiments/html_astra/inner_predictions.npz", va, y)
    cat = aligned_prediction("experiments/v5/cat_inner_1.npz", va, y)
    base = .5 * text + .5 * cat
    protocol = dict(candidate_count=1, parameters=CONFIG, weights=list(WEIGHTS),
                    baseline="0.5 old text + 0.5 cat_inner_1",
                    selection="maximum inner F1; ties prefer smaller new-model weight",
                    oof_gate="strictly positive best inner blend F1 gain over tuned baseline",
                    secondary="0.5 new text + 0.5 cat (reported, not selected)",
                    splitter=dict(n_splits=3, shuffle=True, random_state=2026),
                    sources=sources, data_sha256=digest,
                    forbidden="holdout, added rows, test labels, external data, network")
    write_json("preregistered.json", protocol)
    np.savez_compressed(OUT / "inner_split.npz", inner_train=tr, inner_validation=va)
    baseline = metrics(y, base)
    emit("inner_baseline", **baseline)
    p, timing = fit_one(cache, tr, va, "inner_full_vocabulary", digest, sources)
    blends = [dict(weight=w, **metrics(y, (1-w)*base+w*p)) for w in WEIGHTS]
    selected = max(blends, key=lambda r: (r["tuned"]["f1"], -r["weight"]))
    delta = selected["tuned"]["f1"] - baseline["tuned"]["f1"]
    selection = dict(parameters=CONFIG, new_text_weight=selected["weight"],
                     threshold=selected["tuned"]["threshold"],
                     standalone_threshold=metrics(y,p)["tuned"]["threshold"],
                     baseline_threshold=baseline["tuned"]["threshold"],
                     old_text_threshold=metrics(y,text)["tuned"]["threshold"],
                     inner_f1_gain=delta, run_oof=bool(delta > 0),
                     selection_scope="inner_train/inner_validation only", sources=sources,
                     data_sha256=digest)
    write_json("selection.json", selection)
    result = dict(baseline=baseline, old_text=metrics(y,text), cat=metrics(y,cat),
                  new_text=metrics(y,p), blends=blends,
                  replace_text_half_cat=metrics(y,.5*p+.5*cat),
                  selection=selection, timing=timing)
    write_json("inner_results.json", result)
    emit("inner_complete", **{k:v for k,v in result.items() if k != "timing"})


def oof(cache, splits, digest, sources):
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["sources"] == sources and selection["data_sha256"] == digest
    assert selection["parameters"] == CONFIG
    if not selection["run_oof"]:
        emit("oof_skipped", reason="No positive inner blend gain")
        return
    indices = splits["development"]
    y = cache["labels"][indices]
    assert len(indices) == 50977
    p = np.full(len(indices), np.nan)
    folds = np.full(len(indices), -1, dtype=np.int8)
    covered = np.zeros(len(indices), dtype=np.int8)
    timings = []
    for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(indices,y)):
        fp, timing = fit_one(cache, indices[tr], indices[va], f"fold_{fold}", digest, sources)
        p[va], folds[va] = fp, fold
        covered[va] += 1
        timings.append(timing)
    assert np.isfinite(p).all() and np.all(covered == 1)
    # Read comparison predictions only after all three new models are finished.
    old_text = aligned_prediction("experiments/html_astra/development_oof.npz", indices, y)
    old_cat = aligned_prediction("experiments/v5/cat_development_oof.npz", indices, y)
    baseline = .5 * old_text + .5 * old_cat
    v6 = aligned_prediction("reports/v6/development_oof.npz", indices, y)
    w = selection["new_text_weight"]
    blended = (1-w)*baseline+w*p
    cutoff = selection["threshold"]
    report = dict(rows=len(indices), selection=selection,
                  protocol="original 3fold OOF; all parameters/weights/thresholds frozen on inner; no OOF retuning",
                  limitation="Inner selection rows are inside development; this is not nested unbiased model-selection evaluation. v6 was selected on development OOF and is contextual only.",
                  standalone=measure(y,p,selection["standalone_threshold"]),
                  standalone_at_05=measure(y,p),
                  old_text=measure(y,old_text,selection["old_text_threshold"]),
                  old_text_at_05=measure(y,old_text),
                  baseline=measure(y,baseline,selection["baseline_threshold"]),
                  baseline_at_05=measure(y,baseline), blend=measure(y,blended,cutoff),
                  blend_at_05=measure(y,blended), v6_at_05=measure(y,v6),
                  folds=[dict(fold=f, standalone=measure(y[folds==f],p[folds==f],selection["standalone_threshold"]),
                              baseline=measure(y[folds==f],baseline[folds==f],selection["baseline_threshold"]),
                              blend=measure(y[folds==f],blended[folds==f],cutoff)) for f in range(3)],
                  measured_fit_seconds=sum(t["fit_seconds"] for t in timings),
                  measured_fold_total_seconds=sum(t["total_seconds"] for t in timings))
    np.savez_compressed(OUT / "development_oof.npz", indices=indices, labels=y,
                        probabilities=p, blended=blended, baseline=baseline, folds=folds)
    write_json("oof_results.json", report)
    emit("oof_complete", **report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inner", "oof"))
    args = parser.parse_args()
    sys.addaudithook(audit)
    started = time.perf_counter()
    cache = joblib.load(ROOT / "data/content_train.joblib")
    assert len(cache["records"]) == len(cache["labels"]) == 63900
    splits = aligned_splits(ROOT, cache)
    digest, sources = data_digest(ROOT), provenance()
    assert len(splits["development"]) == 50977
    assert set(splits["inner_train"]).isdisjoint(splits["inner_validation"])
    assert set(splits["inner_train"]) | set(splits["inner_validation"]) <= set(splits["development"])
    emit("loaded", stage=args.stage, seconds=time.perf_counter()-started,
         inner_train=len(splits["inner_train"]), inner_validation=len(splits["inner_validation"]))
    (inner if args.stage == "inner" else oof)(cache, splits, digest, sources)
    write_json(f"{args.stage}_execution.json", dict(seconds=time.perf_counter()-started,
               project_files_read=sorted(READS), threadpools=threadpool_info(),
               environment={n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","joblib")},
               scope="Only original development fitted/evaluated; holdout and added records never selected for fit/predict; no network"))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
