"""Explore the positive preregistered SECONDARY replacement on original OOF.

The primary additive-blend gate failed. The secondary half-new-text/half-cat
comparison improved inner F1 by one net error, so the requested OOF is run for
that fixed configuration. This follow-up decision was made after seeing inner
results and is explicitly not represented as the original primary gate.
"""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.content_checkpoints import data_digest
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import measure
from experiments.v7_astra.run import (OUT, CONFIG, READS, audit, sha, provenance,
                                     fit_one, aligned_prediction, write_json, emit)


def main():
    sys.addaudithook(audit)
    started = time.perf_counter()
    inner = json.loads((OUT / "inner_results.json").read_text())
    previous = inner["selection"]
    assert previous["sources"] == provenance()
    assert previous["parameters"] == CONFIG
    # Write this before reading any OOF predictions or evaluating OOF labels.
    frozen = dict(parameters=CONFIG, text_weight=.5, cat_weight=.5,
                  threshold=inner["replace_text_half_cat"]["tuned"]["threshold"],
                  standalone_threshold=previous["standalone_threshold"],
                  baseline_threshold=previous["baseline_threshold"],
                  old_text_threshold=previous["old_text_threshold"],
                  inner_f1_gain=inner["replace_text_half_cat"]["tuned"]["f1"]-inner["baseline"]["tuned"]["f1"],
                  primary_additive_gate_passed=previous["run_oof"],
                  decision="Exploratory follow-up of the preregistered secondary replacement after a one-error inner gain; primary additive gate failed. No new estimator configs or weight search.",
                  selection_scope="inner only; weights and threshold frozen before OOF",
                  splitter=dict(n_splits=3,shuffle=True,random_state=2026),
                  sources={**provenance(), "experiments/v7_astra/run_oof.py":sha(__file__)},
                  data_sha256=previous["data_sha256"],
                  inner_results_sha256=sha(OUT / "inner_results.json"))
    assert frozen["inner_f1_gain"] > 0
    frozen_path = OUT / "oof_selection.json"
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == frozen
    else:
        write_json("oof_selection.json", frozen)
    cache = joblib.load(ROOT / "data/content_train.joblib")
    digest = data_digest(ROOT)
    assert digest == frozen["data_sha256"]
    indices = aligned_splits(ROOT,cache)["development"]
    assert len(indices) == 50977
    y = cache["labels"][indices]
    predictions = np.full(len(indices),np.nan)
    fold_ids = np.full(len(indices),-1,dtype=np.int8)
    covered = np.zeros(len(indices),dtype=np.int8)
    timings = []
    for fold,(tr,va) in enumerate(StratifiedKFold(**frozen["splitter"]).split(indices,y)):
        p,timing = fit_one(cache,indices[tr],indices[va],f"fold_{fold}",digest,frozen["sources"])
        predictions[va],fold_ids[va] = p,fold
        covered[va] += 1
        timings.append(timing)
    assert np.all(covered == 1) and np.isfinite(predictions).all()
    text_path = "experiments/html_astra/development_oof.npz"
    cat_path = "experiments/v5/cat_development_oof.npz"
    old_text = aligned_prediction(text_path,indices,y)
    old_cat = aligned_prediction(cat_path,indices,y)
    for path in (text_path,cat_path):
        with np.load(ROOT / path) as a:
            np.testing.assert_array_equal(a["folds"],fold_ids)
    baseline = .5*old_text + .5*old_cat
    combined = .5*predictions + .5*old_cat
    v6 = aligned_prediction("reports/v6/development_oof.npz",indices,y)
    summary = lambda yy,pp,cut:measure(yy,pp,cut)
    before = baseline >= frozen["baseline_threshold"]
    after = combined >= frozen["threshold"]
    report = dict(rows=len(indices),selection=frozen,
                  protocol="Original 3-fold OOF; fold-local vocabularies/IDF; frozen inner thresholds; no OOF search or refit",
                  limitation="Not nested model-selection evaluation: inner rows overlap original development. v6 probabilities were selected on development OOF and are a contextual comparator only.",
                  standalone=measure(y,predictions,frozen["standalone_threshold"]),
                  standalone_at_05=measure(y,predictions),
                  old_text=measure(y,old_text,frozen["old_text_threshold"]),
                  old_text_at_05=measure(y,old_text),
                  baseline=measure(y,baseline,frozen["baseline_threshold"]),
                  baseline_at_05=measure(y,baseline),
                  replacement=measure(y,combined,frozen["threshold"]),
                  replacement_at_05=measure(y,combined),v6_at_05=measure(y,v6),
                  fixed_errors=int(np.sum((before != y)&(after == y))),
                  new_errors=int(np.sum((before == y)&(after != y))),
                  folds=[dict(fold=f,baseline=summary(y[fold_ids==f],baseline[fold_ids==f],frozen["baseline_threshold"]),
                              replacement=summary(y[fold_ids==f],combined[fold_ids==f],frozen["threshold"]),
                              standalone=summary(y[fold_ids==f],predictions[fold_ids==f],frozen["standalone_threshold"])) for f in range(3)],
                  measured_fit_seconds=sum(t["fit_seconds"] for t in timings),
                  measured_fold_total_seconds=sum(t["total_seconds"] for t in timings),
                  comparison_sources_sha256={p:sha(ROOT / p) for p in (text_path,cat_path,"reports/v6/development_oof.npz")})
    np.savez_compressed(OUT / "development_oof.npz",indices=indices,labels=y,
                        probabilities=predictions,replacement=combined,baseline=baseline,folds=fold_ids)
    write_json("oof_results.json",report)
    write_json("oof_execution.json",dict(seconds=time.perf_counter()-started,
               project_files_read=sorted(READS),threadpools=threadpool_info()))
    emit("oof_complete",**report)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
