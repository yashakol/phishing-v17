"""Record observed screening cost and a clearly labeled OOF projection."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import json
import joblib


def main():
    result = json.loads((OUT / "results.json").read_text())
    model = joblib.load(OUT / "screen_selected.joblib")
    # Three conservative outer-refit proxies plus prediction on all development.
    projection = (3 * result["outer_refit_seconds"] + result["validation_predict_seconds"] *
                  result["alignment"]["development_rows"] / result["alignment"]["validation_rows"])
    oof_path = OUT / "oof_results.json"
    oof = json.loads(oof_path.read_text()) if oof_path.exists() else None
    oof_status = ("complete" if oof else "running" if (OUT / "oof/run_guard.json").exists()
                  else "authorized; waiting for completed screening")
    review = {
        "initial_aborted_attempt": {
            "configuration": "vocabulary fitted on all 34663 inner training records",
            "stage": "sklearn TfidfVectorizer._count_vocab, before any candidate or validation metrics",
            "reason": "high host memory load on 16 GB machine; avoid collecting all rare ngrams",
            "evidence": "screen_initial_unbounded.log contains KeyboardInterrupt traceback",
            "elapsed_seconds": None,
            "elapsed_note": "interrupted stage timing was not persisted; no invented duration",
            "peak_process_ram_bytes": None,
            "memory_note": "host vm_stat was inspected; process peak RAM was not measured"
        },
        "bounded_choice": {
            "vocabulary_fit_rows": model.vocabulary_fit_rows,
            "actual_vocabulary_fit_rows": model.vocabulary_fit_rows_,
            "transform_batch_size": model.transform_batch_size,
            "fit_scope": "vocabulary and IDF on deterministic training-only sample; classifier on all fit rows",
            "threads": model.threads,
            "outer_refit_view_seconds": model.view_fit_seconds_,
        },
        "measured": {key: result[key] for key in
                     ("feature_seconds", "outer_refit_seconds", "validation_predict_seconds",
                      "serialized_bytes", "total_seconds")},
        "three_fold_oof": {
            "status": oof_status,
            "authorization": "User authorized model improvement and Astra; the main agent requested comparable development OOF within that scope.",
            "estimated_seconds_proxy": projection,
            "estimate_is_not_measurement": True,
            "method": "3 measured outer refits + full-development inference scaled from measured validation",
            "limitations": "folds use fewer fit rows; host contention, I/O, and memory pressure can change runtime"
        },
        "observed_validation": result["validation"],
        "selection_at_screening": result["selection"],
    }
    if oof:
        review["three_fold_oof"]["measured_fold_seconds"] = oof["measured_fold_seconds"]
        review["three_fold_oof"]["measured_run_seconds"] = oof["total_seconds_this_run"]
    components_path = OUT / "component_results.json"
    if components_path.exists():
        components = json.loads(components_path.read_text())
        review["final_components"] = {key: components[key] for key in
                                       ("status", "components", "total_seconds", "scope")
                                       if key in components}
    (OUT / "cost_review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
