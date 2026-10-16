"""Read-only validation of the user's saved v7 OOF integration, without search."""
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
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.evaluation import measure
from experiments.v7_astra.run import OUT, audit, sha, write_json


def main():
    sys.addaudithook(audit)
    started = time.perf_counter()
    paths = ["experiments/v7_astra_selection.py", "reports/v7/selection.json",
             "reports/v7/development_oof.npz", "reports/v6/development_oof.npz",
             "experiments/v7_astra/development_oof.npz",
             "experiments/v7_astra/oof_selection.json"]
    hashes = {p: sha(ROOT / p) for p in paths}
    report = json.loads((ROOT / "reports/v7/selection.json").read_text())
    selected, baseline = report["selected"], report["baseline_v6"]
    assert report["text_oof_sha256"] == hashes[report["text_oof"]]
    assert report["text_selection_sha256"] == hashes[report["text_selection"]]
    text_selection = json.loads((OUT / "oof_selection.json").read_text())
    assert report["train_cache_sha256"] == text_selection["data_sha256"]
    assert report["text_parameters"] == text_selection["parameters"]
    with np.load(ROOT / "reports/v7/development_oof.npz") as integrated, \
         np.load(ROOT / "reports/v6/development_oof.npz") as old, \
         np.load(OUT / "development_oof.npz") as text:
        y = text["labels"]
        assert len(y) == 50977
        for table in (integrated, old):
            np.testing.assert_array_equal(table["indices"], text["indices"])
            np.testing.assert_array_equal(table["labels"], y)
        np.testing.assert_array_equal(integrated["new_text"], text["probabilities"])
        np.testing.assert_array_equal(integrated["original"], old["probabilities"])
        np.testing.assert_array_equal(integrated["folds"], text["folds"])
        p = integrated["probabilities"]
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        assert measure(y, p, selected["threshold"]) == selected["metrics"]
        assert measure(y, p, baseline["threshold"]) == selected["metrics_at_v6_threshold"]
        assert measure(y, old["probabilities"], baseline["threshold"]) == baseline
        before = old["probabilities"] >= baseline["threshold"]
        after = p >= selected["threshold"]
        assert int(np.sum((before != y) & (after == y))) == selected["fixed_errors"]
        assert int(np.sum((before == y) & (after != y))) == selected["new_errors"]
        for fold, (_, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(y, y)):
            np.testing.assert_array_equal(text["folds"][va], np.full(len(va), fold))
            row = selected["fold_comparison"][fold]
            assert row["baseline_errors"] == int(np.sum(before[va] != y[va]))
            assert row["candidate_errors"] == int(np.sum(after[va] != y[va]))
            assert row["baseline_f1"] == measure(y[va], old["probabilities"][va], baseline["threshold"])["f1"]
            assert row["candidate_f1"] == measure(y[va], p[va], selected["threshold"])["f1"]
    assert hashes == {p: sha(ROOT / p) for p in paths}
    result = dict(selected=selected, baseline_v6=baseline,
                  text_parameters=report["text_parameters"],
                  hierarchy_parameters=report["hierarchy_parameters"],
                  source_sha256=hashes, seconds=time.perf_counter()-started,
                  rows=50977, indices_labels_folds_match=True,
                  exact_text_probabilities_match=True, metrics_recomputed=True,
                  errors_and_fold_counts_recomputed=True,
                  scope="Saved original-development OOF and specified integration source only; no fit, selection search, holdout/test or integration-script execution.",
                  limitation="Weights and threshold were selected on OOF, so this is optimistic selection evidence. The gain includes hierarchy and threshold changes, not an isolated text effect. Full pipeline replay was performed by the user, not repeated here.")
    write_json("integration_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "text_parameters", "hierarchy_parameters")}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
