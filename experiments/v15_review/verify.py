"""Bounded, offline v15 artifact review. No classifier fits or parameter search.

Run from the project root:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v15_review/verify.py

Only synthetic ResourceContext counting is fitted. Saved classifiers are replayed
on their own eligible cross-fitted validation rows, never on diagnostic/test data.
"""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import types

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments/v15_resources"
ALLOWED = {
    "phishing_url/v15_resources.py", "phishing_url/v12_expert.py",
    "phishing_url/content_checkpoints.py", "phishing_url/content_validation.py",
    "phishing_url/evaluation.py", "reports/v10/selection.json",
    "experiments/v10_margin/selected_oof.npz", "reports/split_indices.npz",
    "reports/v3/split_indices.npz",
}
READS = set()
BLOCKED = []


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind")):
        BLOCKED.append(event)
        raise PermissionError("Offline review: network is forbidden")
    if event in ("subprocess.Popen", "os.system"):
        BLOCKED.append(event)
        raise PermissionError("Review cannot spawn subprocesses")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
               (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
    if writing and not p.is_relative_to(OUT):
        BLOCKED.append(str(p))
        raise PermissionError("Write outside review directory")
    if not p.is_relative_to(ROOT) or p.is_relative_to(ROOT / ".venv"):
        return  # Installed runtime/library reads only, no project data here.
    rel = p.relative_to(ROOT).as_posix()
    if not (rel in ALLOWED or p.is_relative_to(EXP) or p.is_relative_to(OUT)):
        BLOCKED.append(rel)
        raise PermissionError("Project read outside explicit review allowlist: " + rel)
    if not writing:
        READS.add(rel)


sys.addaudithook(audit)
# Avoid importing unrelated project __init__ code while unpickling the helper.
package = types.ModuleType("phishing_url")
package.__path__ = [str(ROOT / "phishing_url")]
sys.modules["phishing_url"] = package
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_info, threadpool_limits
# Execute only the explicitly allowed source, bypassing Python's optional
# __pycache__ reads. The initial review attempt blocked those reads correctly.
for _module in ("v15_resources", "v12_expert", "content_checkpoints", "evaluation"):
    _path = ROOT / "phishing_url" / (_module + ".py")
    _qualified = "phishing_url." + _module
    _loaded = types.ModuleType(_qualified)
    _loaded.__file__ = str(_path)
    _loaded.__package__ = "phishing_url"
    sys.modules[_qualified] = _loaded
    exec(compile(_path.read_text(), str(_path), "exec"), _loaded.__dict__)
from phishing_url.v15_resources import ResourceContext, KEYS, MISSING, add_resource_evidence
from phishing_url.v12_expert import eligibility, blend_expert
from phishing_url.content_checkpoints import load_component
from phishing_url.evaluation import measure


FIT_ATTEMPTS = 0


def forbidden_fit(*args, **kwargs):
    global FIT_ATTEMPTS
    FIT_ATTEMPTS += 1
    raise AssertionError("No classifier fitting in independent review")


for klass in (CatBoost, CatBoostClassifier, CatBoostRegressor):
    klass.fit = forbidden_fit


RESULT = {"status": "running", "checks": [], "folds": [], "limitations": [
    "Source inspection plus internally consistent metadata is not an execution trace or proof of historical training membership.",
    "Raw URLs/resource keys per row and original caches were deliberately not read. URL+label ordered guard digests, raw cache hashes and upstream v13 hashes cannot be independently recomputed.",
    "Saved resource columns are fold-validation encodings, not classifier-training LOO encodings. Replay verifies classifier outputs from the saved 354-column inputs, not reconstruction of those inputs from raw keys.",
    "The context guard lacks an independently recomputable ordered-index digest. Its row count, prior, configuration and count consistency are verified; row-level resource provenance remains source/metadata evidence.",
    "Not fully nested meta cross-validation: base OOF classifiers for specialist-training rows may have seen specialist-validation rows. Reused development selection is not an independent performance estimate.",
    "Selected recorded weight and thresholds are only replayed. No threshold search, new classifier fit, final inference or diagnostic evaluation is performed.",
]}


def check(name, condition, detail=None):
    entry = {"name": name, "passed": bool(condition)}
    if detail is not None:
        entry["detail"] = detail
    RESULT["checks"].append(entry)
    if not condition:
        raise AssertionError(name)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def equal(name, a, b):
    check(name, np.array_equal(a, b))


def expect_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def synthetic_checks():
    # A is mixed; B and C become unsupported after removing their only row.
    key = ["A", "A", "B", "C", "", "EMPTY", "INVALID"]
    labels = np.array([0, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    frame = pd.DataFrame({k: key for k in KEYS})
    context = ResourceContext().fit(frame, labels)
    before = copy.deepcopy(context.__dict__)
    query = pd.DataFrame({k: ["", "EMPTY", "INVALID", "UNSEEN"] for k in KEYS})
    neutral = context.transform(query).to_numpy()
    check("synthetic_missing_and_unseen_neutral", np.allclose(neutral, 0, rtol=0, atol=1e-14))
    counts = np.array([[1, 1], [1, 1], [0, 1], [1, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
    leave = counts.copy()
    leave[np.arange(4), labels[:4]] -= 1

    def expected(c):
        zero, one = c.T
        total = zero + one
        return np.column_stack([np.log1p(total),
            logit((one + 10 * context.prior_) / (total + 10)) - logit(context.prior_),
            ((total > 0) & ((zero == 0) | (one == 0))).astype(float)])

    full = context.transform(frame).to_numpy()
    loo = context.transform(frame, labels).to_numpy()
    check("synthetic_formula_query", np.allclose(full, np.tile(expected(counts), (1, 3)), rtol=0, atol=1e-14))
    check("synthetic_formula_leave_own_row_out", np.allclose(loo, np.tile(expected(leave), (1, 3)), rtol=0, atol=1e-14))
    check("synthetic_singletons_neutral_after_loo", np.allclose(loo[[2, 3]], 0, rtol=0, atol=1e-14))
    check("synthetic_missing_loo_does_not_subtract", np.allclose(loo[4:], 0, rtol=0, atol=1e-14))
    check("synthetic_unseen_loo_rejected", expect_error(lambda: context.transform(query.iloc[[3]], np.array([0]))))
    check("synthetic_wrong_class_loo_rejected", expect_error(lambda: context.transform(frame.iloc[[2]], np.array([0]))))
    check("synthetic_misaligned_loo_rejected", expect_error(lambda: context.transform(frame, labels[:-1])))
    check("synthetic_invalid_loo_rejected", expect_error(lambda: context.transform(frame, np.full(len(frame), 2))))
    check("synthetic_empty_query", context.transform(frame.iloc[:0]).shape == (0, 9))
    check("synthetic_empty_loo", context.transform(frame.iloc[:0], labels[:0]).shape == (0, 9))
    check("synthetic_counts_prior_rows_nonmutating", context.__dict__ == before)
    base = pd.DataFrame({"base": np.arange(len(frame))}, index=np.arange(len(frame)) + 20)
    joined = add_resource_evidence(base, frame.set_axis(np.arange(len(frame)) + 30), context, labels)
    equal("synthetic_join_aligns_by_row_position", joined.to_numpy(), np.column_stack([np.arange(len(frame)), loo]))
    check("synthetic_join_rejects_misalignment", expect_error(lambda: add_resource_evidence(base.iloc[:-1], frame, context)))
    duplicate = pd.DataFrame({KEYS[0] + "_pure": np.zeros(len(frame))})
    check("synthetic_join_rejects_duplicate_columns", expect_error(lambda: add_resource_evidence(duplicate, frame, context)))
    bad = base.astype(float); bad.iloc[0, 0] = np.inf
    check("synthetic_join_rejects_nonfinite", expect_error(lambda: add_resource_evidence(bad, frame, context)))
    edge = np.array([0., .049, .05, .5, .95, .951, 1.])
    equal("eligibility_inclusive_boundaries", eligibility(edge), [False, False, True, True, True, False, False])
    mixed = blend_expert(edge, np.full(len(edge), .4), .75)
    equal("blend_preserves_ineligible_including_hard_endpoints", mixed[~eligibility(edge)], edge[~eligibility(edge)])
    RESULT["synthetic"] = {"resource_context_fits": 1, "classifier_fits": 0,
        "neutral_tolerance": 1e-14, "counts_unchanged": context.__dict__ == before}


def static_checks(source, initial):
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    by_assign = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    by_assign[target.id] = ast.unparse(n.value)
    exact = {
        "context": "ResourceContext().fit(numeric.iloc[tr], y[tr])",
        "train_frame": "add_resource_evidence(X.iloc[fit], numeric.iloc[fit], context, y[fit])",
        "query_frame": "add_resource_evidence(X.iloc[predict], numeric.iloc[predict], context)",
        "model": "make_expert().fit(train_frame, y[fit])",
    }
    for name, value in exact.items():
        check("source_" + name + "_scope", by_assign[name] == value,
              {"line": next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)), "expression": by_assign[name]})
    tuple_scope = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and [getattr(x, 'id', None) for x in t.elts] == ['fit', 'predict'] for t in n.targets)]
    check("source_uncertain_train_and_validation_scopes", tuple_scope == ["fit, predict = (tr[eligible[tr]], va[eligible[va]])"])
    encoding_calls = [ast.unparse(n) for n in calls if ast.unparse(n.func) == "add_resource_evidence"]
    check("source_only_training_encoder_gets_labels", len(encoding_calls) == 3 and sum(len(n.args) == 4 for n in calls if ast.unparse(n.func) == "add_resource_evidence") == 1,
          encoding_calls)
    transforms = [ast.unparse(n) for n in calls if ast.unparse(n.func) == "context.transform"]
    check("source_validation_context_transform_no_labels", "context.transform(numeric.iloc[va])" in transforms and all(len(n.args) == 1 and not n.keywords for n in calls if ast.unparse(n.func) == "context.transform"))
    check("source_context_guard_complete_train", 'guard(cache_small, dev[tr]' in source)
    check("source_model_guard_uncertain_train", 'guard(cache_small, dev[fit]' in source)
    try:
        compile(initial, "initial_source.txt", "exec")
    except SyntaxError as err:
        initial_error = {"message": err.msg, "line": err.lineno}
    else:
        initial_error = None
    check("initial_source_fails_before_execution", initial_error == {"message": "unmatched '}'", "line": 115}, initial_error)
    compile(source, "search.py", "exec")
    check("current_source_compiles_without_execution", True)
    recovery = read_json(EXP / "recovery.json")
    check("recovery_records_zero_initial_fits_reads", recovery["classifier_fits_from_failed_run"] == 0 and recovery["data_reads_or_metrics_from_failed_run"] == 0)
    check("initial_failure_log_agrees", "SyntaxError: unmatched '}'" in (EXP / "initial_run.log").read_text())
    RESULT["initial_failure"] = {"compile_reproduced": initial_error,
        "scope": "This preserved script cannot execute imports/data reads/fits because parsing fails; this is not a global historical execution audit.",
        "loo_present_in_hashed_successful_source": True}


def main():
    files = sorted([ROOT / p for p in ALLOWED] + [EXP / p for p in (
        "search.py", "initial_source.txt", "initial_run.log", "recovery.json", "preregistered.json",
        "selection.json", "run.log", "training_frame.npz", "development_expert.npz", "selected_oof.npz",
        "models/fold_0.joblib", "models/fold_1.joblib", "models/fold_2.joblib")])
    # The split archives are hashed as bytes; only approved development/mapping
    # arrays are loaded. No holdout labels, scores or predictions are accessed.
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    RESULT["input_artifact_sha256"] = hashes
    protocol = read_json(EXP / "preregistered.json")
    selection = read_json(EXP / "selection.json")
    v10 = read_json(ROOT / "reports/v10/selection.json")
    source = (EXP / "search.py").read_text()
    static_checks(source, (EXP / "initial_source.txt").read_text())
    synthetic_checks()
    for field, name in (("source_sha256", "experiments/v15_resources/search.py"),
                        ("expert_source_sha256", "phishing_url/v15_resources.py"),
                        ("base_expert_source_sha256", "phishing_url/v12_expert.py")):
        check("protocol_" + field, protocol[field] == hashes[name])
    check("selection_protocol_exact_copy", selection["protocol"] == protocol)
    check("baseline_input_hash", protocol["input_sha256"]["experiments/v10_margin/selected_oof.npz"] == hashes["experiments/v10_margin/selected_oof.npz"])
    check("declared_cache_hash_consistency", protocol["input_sha256"]["data/content_train.joblib"] == selection["train_cache_sha256"] == v10["train_cache_sha256"])
    RESULT["unrecomputed_preregistered_input_hashes"] = {k: v for k, v in protocol["input_sha256"].items() if k != "experiments/v10_margin/selected_oof.npz"}
    with np.load(EXP / "training_frame.npz", allow_pickle=False) as z:
        indices, labels, values, columns, eligible_saved = [z[k] for k in ("indices", "labels", "values", "columns", "eligibility")]
    # Retain a single 138 MiB dense matrix; validation prediction batches are small.
    frame = pd.DataFrame(values, columns=columns, copy=False)
    check("frame_shape_unique_finite", values.shape == (50977, 354) and len(set(columns)) == 354 and np.isfinite(values).all())
    check("frame_columns_match_selection", columns.tolist() == selection["feature_columns"])
    check("indices_unique_and_labels_binary", len(np.unique(indices)) == len(indices) and np.array_equal(np.unique(labels), [0, 1]))
    loaded = {}
    for name, path in (("expert", EXP / "development_expert.npz"),
                       ("selected", EXP / "selected_oof.npz"),
                       ("baseline", ROOT / "experiments/v10_margin/selected_oof.npz")):
        with np.load(path, allow_pickle=False) as z:
            loaded[name] = {k: z[k] for k in z.files}
        equal(name + "_indices_alignment", indices, loaded[name]["indices"])
        equal(name + "_labels_alignment", labels, loaded[name]["labels"])
    baseline = loaded["baseline"]["probabilities"]
    expert = loaded["expert"]["probabilities"]
    selected = loaded["selected"]["probabilities"]
    eligible = eligibility(baseline)
    equal("baseline_frame_column", frame["baseline_probability"].to_numpy(), baseline)
    equal("baseline_selected_copy", loaded["selected"]["baseline"], baseline)
    equal("eligibility_training_frame", eligible, eligible_saved)
    equal("eligibility_expert_archive", eligible, loaded["expert"]["eligibility"])
    check("eligibility_report_count", int(eligible.sum()) == selection["eligible_rows"] == 5885)
    with np.load(ROOT / "reports/split_indices.npz", allow_pickle=False) as z:
        original_development = z["development"]
    with np.load(ROOT / "reports/v3/split_indices.npz", allow_pickle=False) as z:
        development, mapping = z["development"], z["old_to_cache"]
    equal("original_development_mapping", indices, mapping[original_development])
    equal("v3_development_indices", indices, development)
    RESULT["split_metadata_arrays_loaded"] = {"reports/split_indices.npz": ["development"], "reports/v3/split_indices.npz": ["development", "old_to_cache"]}
    folds = list(StratifiedKFold(**protocol["splitter"]).split(np.zeros(len(labels)), labels))
    reconstructed_folds = np.full(len(labels), -1, dtype=np.int8)
    replay = baseline.copy()
    resource_columns = [k + suffix for k in KEYS for suffix in ("_log_support", "_log_evidence", "_pure")]
    check("last_nine_columns_resource_schema", columns[-9:].tolist() == resource_columns)
    logs = [json.loads(line) for line in (EXP / "run.log").read_text().splitlines() if line.strip()]
    for fold, (tr, va) in enumerate(folds):
        fit, predict = tr[eligible[tr]], va[eligible[va]]
        reconstructed_folds[va] = fold
        check(f"fold_{fold}_disjoint_and_complete", len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(labels))
        path = EXP / f"models/fold_{fold}.joblib"
        payload = joblib.load(path)
        guard = payload["guard"]
        config = guard["config"]
        context_guard = config["context_guard"]
        model, context = payload["model"]["classifier"], payload["model"]["context"]
        check(f"fold_{fold}_classifier_guard_rows", guard["training_rows"] == len(fit))
        check(f"fold_{fold}_classifier_ordered_indices_sha256", guard["ordered_training_indices_sha256"] == hashlib.sha256(indices[fit].tobytes()).hexdigest())
        check(f"fold_{fold}_context_complete_train_rows", context.training_rows_ == context_guard["training_rows"] == len(tr))
        check(f"fold_{fold}_context_prior_exact", context.prior_ == float(labels[tr].mean()))
        check(f"fold_{fold}_sources_and_protocol", config["source_sha256"] == protocol["expert_source_sha256"] and
              config["base_expert_source_sha256"] == protocol["base_expert_source_sha256"] and
              config["protocol_sha256"] == hashes["experiments/v15_resources/preregistered.json"] and
              context_guard["config"] == {"source_sha256": protocol["expert_source_sha256"]})
        check(f"fold_{fold}_guard_data_hash_consistency", guard["data_sha256"] == context_guard["data_sha256"] == selection["train_cache_sha256"])
        check(f"fold_{fold}_classifier_parameters_and_schema", model.get_params() == config["parameters"] == protocol["classifier"] and
              config["frame_columns"] == model.feature_names_ == columns.tolist() and model.tree_count_ == protocol["classifier"]["iterations"])
        check(f"fold_{fold}_counts_kinds", set(context.counts_) == set(KEYS))
        prior = context.prior_
        label_totals = np.bincount(labels[tr], minlength=2)
        counts_summary = {}
        max_loo_prior_delta = 0.
        for kind in KEYS:
            counts = context.counts_[kind]
            c = np.array(list(counts.values()), dtype=np.int64)
            check(f"fold_{fold}_{kind}_count_consistency", c.ndim == 2 and c.shape[1] == 2 and (c >= 0).all() and
                  (c.sum(axis=1) > 0).all() and (c.sum(axis=0) <= label_totals).all() and not any(k in MISSING for k in counts))
            counts_summary[kind] = {"keys": len(counts), "class_totals": c.sum(axis=0).tolist(), "missing_rows_implied": int(len(tr) - c.sum())}
            # Validate feature triples against the possible count states of the
            # saved context. This is not a proof of row-to-resource-key alignment.
            cf = np.concatenate([np.zeros((1, 2), dtype=np.int64), c])
            cf = np.unique(cf, axis=0)
            observed = frame.iloc[va][[kind + suffix for suffix in ("_log_support", "_log_evidence", "_pure")]].to_numpy()
            total = np.rint(np.expm1(observed[:, 0]))
            post = 1 / (1 + np.exp(-(observed[:, 1] + logit(prior))))
            one = np.rint(post * (total + 10) - 10 * prior)
            zero = total - one
            reconstructed = np.column_stack([zero, one]).astype(np.int64)
            bucket_set = set(map(tuple, cf.tolist()))
            check(f"fold_{fold}_{kind}_validation_states_in_context", all(tuple(row) in bucket_set for row in reconstructed.tolist()))
            calc = np.column_stack([np.log1p(total), logit((one + 10 * prior)/(total + 10)) - logit(prior),
                ((total > 0) & ((zero == 0) | (one == 0))).astype(float)])
            check(f"fold_{fold}_{kind}_validation_formula", np.allclose(observed, calc, rtol=0, atol=1e-12))
            for label in (0, 1):
                left = c[c[:, label] > 0].copy()
                left[:, label] -= 1
                total_left = left.sum(axis=1)
                actual = logit((left[:, 1] + 10 * prior)/(total_left + 10)) - logit(prior)
                strict_prior = (int(label_totals[1]) - label) / (len(tr) - 1)
                strict = logit((left[:, 1] + 10 * strict_prior)/(total_left + 10)) - logit(strict_prior)
                max_loo_prior_delta = max(max_loo_prior_delta, float(np.max(np.abs(actual - strict), initial=0)))
        snapshots = copy.deepcopy(context.__dict__)
        unseen = pd.DataFrame({k: ["", "EMPTY", "INVALID"] for k in KEYS})
        check(f"fold_{fold}_missing_context_neutral", np.allclose(context.transform(unseen).to_numpy(), 0, rtol=0, atol=1e-14))
        for chunk_start in range(0, len(predict), 512):
            batch = predict[chunk_start:chunk_start + 512]
            replay[batch] = model.predict_proba(frame.iloc[batch], thread_count=2)[:, 1]
        equal(f"fold_{fold}_validation_predictions_exact", replay[predict], expert[predict])
        check(f"fold_{fold}_context_nonmutating", context.__dict__ == snapshots)
        # Use an independently rebuilt subset of guard fields; raw URL digest
        # must remain stored metadata, explicitly not presented as reconstructed.
        expected = copy.deepcopy(guard)
        expected["training_rows"] = len(fit)
        expected["ordered_training_indices_sha256"] = hashlib.sha256(indices[fit].tobytes()).hexdigest()
        restored = load_component(path, expected)
        equal(f"fold_{fold}_guarded_reload_prediction", restored["classifier"].predict_proba(frame.iloc[predict[:8]], thread_count=2)[:, 1], expert[predict[:8]])
        mismatch = copy.deepcopy(expected); mismatch["training_rows"] += 1
        check(f"fold_{fold}_guard_rejects_model_scope_mismatch", expect_error(lambda: load_component(path, mismatch)))
        mismatch = copy.deepcopy(expected); mismatch["config"]["context_guard"]["training_rows"] += 1
        check(f"fold_{fold}_guard_rejects_context_scope_mismatch", expect_error(lambda: load_component(path, mismatch)))
        entry = next(e for e in logs if e.get("event") == "expert_fold_started" and e.get("fold") == fold)
        check(f"fold_{fold}_log_counts", entry["fit_rows"] == len(fit) and entry["eligible_validation_rows"] == len(predict))
        # Holding all other labels fixed, flipping one own label changes the
        # global prior by 1/N. |d evidence/d prior| <= 1/(prior*(1-prior)).
        low, high = prior - 1 / len(tr), prior + 1 / len(tr)
        bound = 1 / len(tr) / min(low * (1-low), high * (1-high))
        RESULT["folds"].append({"fold": fold, "context_training_rows": len(tr),
            "context_positive_labels": int(label_totals[1]), "context_prior": prior,
            "classifier_training_rows": len(fit), "validation_rows": len(va),
            "eligible_validation_rows_replayed": len(predict), "prediction_max_abs_difference": float(np.max(np.abs(replay[predict] - expert[predict]))),
            "classifier_indices_hash_recomputed": expected["ordered_training_indices_sha256"],
            "context_url_label_guard_hash_stored_only": context_guard["ordered_training_sha256"],
            "classifier_url_label_guard_hash_stored_only": guard["ordered_training_sha256"],
            "resource_counts_summary": counts_summary,
            "max_evidence_change_if_prior_also_loo_over_saved_buckets": max_loo_prior_delta,
            "single_label_prior_flip_evidence_abs_upper_bound": bound})
        del restored, payload, model, context, snapshots
    equal("reconstructed_fold_ids_expert", reconstructed_folds, loaded["expert"]["folds"])
    equal("reconstructed_fold_ids_selected", reconstructed_folds, loaded["selected"]["folds"])
    equal("expert_full_probability_replay", replay, expert)
    equal("ineligible_expert_probabilities_preserved", expert[~eligible], baseline[~eligible])
    chosen = selection["selected"]
    check("selected_weight_predeclared", chosen["weight"] == .75 and chosen["weight"] in protocol["weights"])
    blended = blend_expert(baseline, replay, chosen["weight"])
    equal("selected_blend_exact", blended, selected)
    equal("ineligible_selected_probabilities_preserved", selected[~eligible], baseline[~eligible])
    hard = (baseline == 0) | (baseline == 1)
    equal("hard_endpoints_preserved", selected[hard], baseline[hard])
    old_cut, new_cut = v10["selected"]["threshold"], chosen["threshold"]
    check("reported_thresholds_within_unchanged_confident_regions", .05 < old_cut < .95 and .05 < new_cut < .95)
    equal("ineligible_decisions_preserved", selected[~eligible] >= new_cut, baseline[~eligible] >= old_cut)
    baseline_metrics = measure(labels, baseline, old_cut)
    selected_metrics = measure(labels, blended, new_cut)
    frozen_metrics = measure(labels, blended, old_cut)
    check("baseline_metrics_exact", baseline_metrics == v10["selected"]["metrics"] == selection["baseline_v10"])
    check("selected_metrics_exact", selected_metrics == chosen["metrics"])
    check("frozen_metrics_exact", frozen_metrics == chosen["metrics_at_baseline_threshold"])
    before, after, frozen = baseline >= old_cut, blended >= new_cut, blended >= old_cut
    base_errors, errors, frozen_errors = [int((pred != labels).sum()) for pred in (before, after, frozen)]
    fixed = int(((before != labels) & (after == labels)).sum())
    introduced = int(((before == labels) & (after != labels)).sum())
    comparisons = [{"fold": f, "before_errors": int((before[va] != labels[va]).sum()), "after_errors": int((after[va] != labels[va]).sum())} for f, (_, va) in enumerate(folds)]
    check("fold_error_arithmetic_exact", comparisons == chosen["fold_comparison"])
    check("fixed_new_net_arithmetic", (fixed, introduced, fixed - introduced, base_errors - frozen_errors) ==
          (chosen["fixed_errors"], chosen["new_errors"], chosen["net_errors"], chosen["frozen_net"]))
    gate_terms = {"selected_net_gain_at_least_8": base_errors - errors >= 8,
                  "frozen_net_gain_at_least_8": base_errors - frozen_errors >= 8,
                  "higher_f1": selected_metrics["f1"] > baseline_metrics["f1"],
                  "two_improved_folds": sum(c["after_errors"] < c["before_errors"] for c in comparisons) >= 2,
                  "no_fold_more_than_3_worse": all(c["after_errors"] <= c["before_errors"] + 3 for c in comparisons)}
    check("original_gate_recomputed_passed", all(gate_terms.values()) == chosen["gate_passed"] == selection["gate_passed"] == True)
    check("recorded_selected_candidate_present", chosen in selection["candidates"])
    check("recorded_candidate_weights_match_preregistration", [c["weight"] for c in selection["candidates"]] == protocol["weights"])
    check("three_logged_completions", [e["fold"] for e in logs if e.get("event") == "expert_fold_complete"] == [0, 1, 2])
    RESULT["selected_replay"] = {"rows": len(labels), "features": len(columns), "eligible_rows": int(eligible.sum()),
        "ineligible_probabilities_and_decisions_preserved": int((~eligible).sum()), "hard_endpoints_preserved": int(hard.sum()),
        "weight": chosen["weight"], "recorded_threshold": new_cut, "frozen_v10_threshold": old_cut,
        "baseline_errors": base_errors, "selected_errors": errors, "frozen_threshold_errors": frozen_errors,
        "net_gain": base_errors - errors, "frozen_gain": base_errors - frozen_errors, "fixed": fixed,
        "introduced": introduced, "baseline_f1": baseline_metrics["f1"], "selected_f1": selected_metrics["f1"],
        "fold_comparison": comparisons, "gate_terms": gate_terms, "gate_passed": all(gate_terms.values())}
    RESULT["prior_assessment"] = {
        "formula": "logit((n1+10*p)/(n0+n1+10))-logit(p), with n0,n1 leave-own-row-out for classifier training",
        "meaning": "p is one fold-wide constant, not a row-varying own-label feature. Counts remove the direct row contribution. The prior is not strictly LOO: one own-label flip changes p by 1/N and shifts supported evidence by a very small amount; missing/unseen/LOO-singleton evidence stays neutral.",
        "evidence_derivative_bound": "-1/(p*(1-p)) <= d(evidence)/dp <= 0",
        "conclusion": "No meaningful direct per-row self-label feature is identified from the globally constant prior. Do not claim mathematical independence from the own label: the shared prior retains a tiny aggregate dependency. No changes to parent source or parameters are proposed by this review.",
    }
    check("no_classifier_fit_attempts", FIT_ATTEMPTS == 0)
    check("no_blocked_access_attempts", not BLOCKED)
    check("review_inputs_unchanged_during_run", all(sha(ROOT / p) == digest for p, digest in hashes.items()))
    RESULT["status"] = "passed_with_stated_provenance_and_validation_limits"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    try:
        with threadpool_limits(limits=2):
            main()
            RESULT["threadpools"] = [{k: p[k] for k in ("user_api", "internal_api", "num_threads")} for p in threadpool_info()]
            check("numerical_threadpools_at_most_two", all(p["num_threads"] <= 2 for p in RESULT["threadpools"]))
    except Exception as exc:
        RESULT["status"] = "failed"
        RESULT["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        RESULT["classifier_fit_attempts"] = FIT_ATTEMPTS
        RESULT["project_files_read"] = sorted(READS)
        RESULT["blocked_access_attempts"] = BLOCKED
        RESULT["peak_rss_platform_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        RESULT["peak_rss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
        RESULT["verification_script_sha256"] = sha(Path(__file__))
        (OUT / "verification.json").write_text(json.dumps(RESULT, indent=2) + "\n")
    print(json.dumps({"status": RESULT["status"], "checks": len(RESULT["checks"]), "selected_replay": RESULT["selected_replay"], "folds": RESULT["folds"]}, indent=2))
