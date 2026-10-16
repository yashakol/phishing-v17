"""One preregistered, offline, fold-local C16 neighbor-residual experiment."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import csv
import gc
import hashlib
import json
import time
from datetime import datetime, timezone

READS, WRITES = set(), set()
INPUTS = {
    "data/content_train.joblib", "reports/clean_train.csv", "reports/split_indices.npz",
    "reports/v15/selection.json", "experiments/v10_margin/selected_oof.npz",
    "experiments/v15_resources/selected_oof.npz", "experiments/v7_astra/development_oof.npz",
    "phishing_url/__init__.py", "phishing_url/v7_text.py", "phishing_url/html_text.py",
    "phishing_url/content_checkpoints.py", "phishing_url/evaluation.py",
    "phishing_url/content_validation.py", "experiments/v7_astra/run.py",
    "experiments/v7_astra/run_oof.py",
}
for f in range(3):
    INPUTS.update({f"experiments/v7_astra/guards/fold_{f}.json", f"experiments/v7_astra/models/fold_{f}.joblib"})


def audit(event, args):
    if event.startswith(("socket.connect", "socket.getaddrinfo", "socket.bind", "subprocess.Popen", "os.system")):
        raise PermissionError("Offline, no child processes")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing:
        if not path.is_relative_to(OUT):
            raise PermissionError(f"Write outside experiment: {path}")
        WRITES.add(path.relative_to(ROOT).as_posix())
        return
    if not path.is_relative_to(ROOT):
        return  # Installed runtime and libraries, not project inputs.
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(".venv/"):
        return
    if any(x in rel.lower() for x in ("external/", "reports/v2/", "reference", "submission", "holdout", "test")):
        raise PermissionError(f"Forbidden input: {rel}")
    # Reading installed bytecode for the explicitly permitted own modules is OK.
    permitted_bytecode = rel.startswith("phishing_url/__pycache__/") and ("phishing_url/" + path.name.split(".")[0] + ".py") in INPUTS
    if rel not in INPUTS and not path.is_relative_to(OUT) and not permitted_bytecode:
        raise PermissionError(f"Input outside allowlist: {rel}")
    READS.add(rel)


sys.addaudithook(audit)
import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v7_text import FullVocabularyTextModel


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def emit(event, **data):
    print(json.dumps({"event": event, **data}, allow_nan=False), flush=True)


def prepare():
    assert not (OUT / "preregistered.json").exists(), "No retrospective registration"
    selection = json.loads((ROOT / "reports/v15/selection.json").read_text())
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "c16_fold_local_train_residual_knn8", "variant_count": 1,
        "splitter": {"n_splits": 3, "shuffle": True, "random_state": 2026},
        "representation": "Exact guarded C16 word/char/markup/URL TF-IDF; concatenate unchanged blocks then L2 normalize for cosine only",
        "eligible_probability": [0.05, 0.95], "neighbors": 8,
        "minimum_similarity": 0.55, "minimum_support": 2,
        "power": 4, "smoothing": 0.5, "residual_weight": 1.0, "maximum_absolute_delta": 0.1,
        "formula": "r_j = y_train_j - C16_fold.predict(train_j); delta = clip(sum(s_j^4*r_j)/(0.5+sum(s_j^4)), -0.1, 0.1); p_new=clip(p_base+delta,0,1); delta=0 with support<2 or p_base outside [0.05,0.95]",
        "neighbor_ties": "Stable original training-index order; sparse_dot_topn selection; dense verification accepts equivalent cosine ties",
        "query_batch": 128, "transform_batch": 1024, "threads": 2,
        "classifier_fits": 0, "vectorizer_fits": 0, "nonparametric_index_fits": 3,
        "primary_baseline": "v15", "secondary_baseline": "v10; same fixed correction, contextual comparison only",
        "frozen_v15_threshold": selection["selected"]["threshold"],
        "frozen_v10_threshold": selection["baseline_v10"]["threshold"],
        "threshold_tuning": "Each prediction vector separately: max pooled development F1, tie nearest 0.5. No neighbor-parameter search. Report same candidate threshold and separately tuned baselines.",
        "gate": {"net_fewer_errors_vs_v15_at_each_mode": 5, "higher_f1_at_each_mode": True,
                 "improved_folds_at_each_mode": 2, "maximum_fold_error_increase_at_each_mode": 2,
                 "modes": ["frozen_v15", "individually_tuned"]},
        "scope": "Only original aligned development rows; per fold search database and residual labels are exclusively outer-train; no same-row neighbors; validation labels used only in evaluation and original stratified fold reconstruction",
        "limitations": ["Reused development with prior model/threshold selection; not an independent score estimate.",
                        "Inherited v15 OOF specialist is not fully nested, as documented in its selection; this experiment does not repair that baseline limitation.",
                        "C16 training residuals are in-sample and likely shrunken; residual transfer to v15/v10 is a hypothesis, not calibrated ensemble residual estimation.",
                        "The training cache is deserialized once; only development records/labels are retained and accessed by the experiment. Original split archive only development member is read.",
                        "No raw HTML reparse, new representation fit, full-data retraining, or public-score inference."],
        "source_sha256": sha(__file__),
        "input_sha256": {p: sha(ROOT / p) for p in sorted(INPUTS)},
    }
    with (OUT / "preregistered.json").open("x") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True)
        stream.write("\n")
    emit("preregistered", sha256=sha(OUT / "preregistered.json"), variant=protocol["variant"])


def features(similarities, labels, train_probability, cfg):
    valid = (labels >= 0) & (similarities >= cfg["minimum_similarity"])
    weights = np.where(valid, np.clip(similarities, 0, 1) ** cfg["power"], 0.)
    total = weights.sum(axis=1)
    denominator = cfg["smoothing"] + total
    label_mean = (weights * np.maximum(labels, 0)).sum(axis=1) / denominator
    prediction_mean = (weights * train_probability).sum(axis=1) / denominator
    residual = label_mean - prediction_mean
    support = valid.sum(axis=1)
    delta = np.clip(cfg["residual_weight"] * residual, -cfg["maximum_absolute_delta"], cfg["maximum_absolute_delta"])
    delta[support < cfg["minimum_support"]] = 0
    return dict(delta=delta, support=support, total_weight=total,
                smoothed_label_mean=label_mean, smoothed_prediction_mean=prediction_mean,
                residual=residual, maximum_similarity=similarities.max(axis=1))


def validate_formula(cfg):
    s = np.array([[1., 1.], [1., 1.], [0., 0.], [1., 0.]])
    y = np.array([[1, 1], [0, 0], [-1, -1], [1, -1]])
    p = np.array([[.9, .9], [.1, .1], [0., 0.], [0., 0.]])
    np.testing.assert_allclose(features(s, y, p, cfg)["delta"], [.08, -.08, 0, 0], atol=1e-12)
    assert features(np.ones((1, 2)), np.ones((1, 2)), np.zeros((1, 2)), cfg)["delta"][0] == .1


def original_alignment(cache, dev, y):
    # Do not call aligned_splits: it also materializes excluded split labels.
    with np.load(ROOT / "reports/split_indices.npz", allow_pickle=False) as archive:
        old_dev = archive["development"]
    wanted = set(map(int, old_dev))
    own = {url: i for i, url in enumerate(cache["urls"])}
    assert len(own) == len(cache["urls"])
    mapping, labels = {}, {}
    with (ROOT / "reports/clean_train.csv").open(newline="", encoding="utf-8") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if i in wanted:
                mapping[i], labels[i] = own[row["url"]], int(row["result"])
    np.testing.assert_array_equal(dev, np.array([mapping[int(i)] for i in old_dev]))
    np.testing.assert_array_equal(y, np.array([labels[int(i)] for i in old_dev]))
    np.testing.assert_array_equal(y, cache["labels"][dev])
    assert len(dev) == len(set(map(int, dev))) == 50977


def guard_for_fold(urls, y, dev, tr, stored, digest):
    sample = hashlib.sha256()
    for i in tr:
        sample.update(urls[int(i)].encode("utf-8")); sample.update(b"\x00")
        sample.update(bytes([int(y[int(i)])]))
    expected = {"training_rows": len(tr), "ordered_training_sha256": sample.hexdigest(),
                "ordered_training_indices_sha256": array_sha(dev[tr]), "data_sha256": digest,
                "config": stored["config"]}
    assert stored == expected, "Training scope mismatch"
    assert stored["config"]["parameters"] == FullVocabularyTextModel().get_params()
    for path, digest in stored["config"]["sources"].items():
        assert path in INPUTS and sha(ROOT / path) == digest, path
    return expected


def transform_train(model, records, tr, cfg):
    parts, probability = [], []
    for start in range(0, len(tr), cfg["transform_batch"]):
        indices = tr[start:start + cfg["transform_batch"]]
        matrix = model._matrix([records[int(i)] for i in indices])
        probability.append(model.classifier_.predict_proba(matrix)[:, 1])
        parts.append(normalize(matrix, norm="l2", copy=False))
        if start % (8 * cfg["transform_batch"]) == 0:
            emit("training_transform", completed=min(start + len(indices), len(tr)), total=len(tr))
    return vstack(parts, format="csr"), np.concatenate(probability)


def comparison(y, before_p, after_p, before_cut, after_cut, dev, fold_ids):
    before, after = before_p >= before_cut, after_p >= after_cut
    fixed = (before != y) & (after == y)
    new = (before == y) & (after != y)
    def metrics(p, cut):
        m = measure(y, p, cut)
        return {**m, "errors": m["fp"] + m["fn"]}
    folds = []
    for f in range(3):
        m = fold_ids == f
        folds.append({"fold": f, "rows": int(m.sum()), "before_errors": int((before[m] != y[m]).sum()),
                      "after_errors": int((after[m] != y[m]).sum()), "fixed": int(fixed[m].sum()),
                      "new": int(new[m].sum()), "before_f1": measure(y[m], before_p[m], before_cut)["f1"],
                      "after_f1": measure(y[m], after_p[m], after_cut)["f1"]})
    return {"before": metrics(before_p, before_cut), "after": metrics(after_p, after_cut),
            "fixed_errors": int(fixed.sum()), "new_errors": int(new.sum()), "net_fewer_errors": int(fixed.sum() - new.sum()),
            "fixed_indices": dev[fixed].tolist(), "new_error_indices": dev[new].tolist(), "folds": folds}


def main():
    started = time.perf_counter()
    assert not (OUT / "results.json").exists(), "No silent rescreening"
    cfg = json.loads((OUT / "preregistered.json").read_text())
    assert cfg["source_sha256"] == sha(__file__), "Source changed after preregistration"
    for p, h in cfg["input_sha256"].items():
        assert sha(ROOT / p) == h, p
    write("started.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(OUT / "preregistered.json"), "source_sha256": sha(__file__)})
    validate_formula(cfg)
    with np.load(ROOT / "experiments/v15_resources/selected_oof.npz", allow_pickle=False) as data:
        dev, y = data["indices"], data["labels"]
        baseline = {"v15": data["probabilities"]}
        recorded_folds = data["folds"]
        v15_base = data["baseline"]
    with np.load(ROOT / "experiments/v10_margin/selected_oof.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(dev, data["indices"]); np.testing.assert_array_equal(y, data["labels"])
        baseline["v10"] = data["probabilities"]
    np.testing.assert_array_equal(v15_base, baseline["v10"])
    with np.load(ROOT / "experiments/v7_astra/development_oof.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(dev, data["indices"]); np.testing.assert_array_equal(y, data["labels"])
        np.testing.assert_array_equal(recorded_folds, data["folds"])
        original_c16 = data["probabilities"]
    cache = joblib.load(ROOT / "data/content_train.joblib")
    original_alignment(cache, dev, y)
    records, urls = [cache["records"][int(i)] for i in dev], [cache["urls"][int(i)] for i in dev]
    del cache
    gc.collect()
    folds = list(StratifiedKFold(**cfg["splitter"]).split(dev, y))
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    origins = []
    for f, (tr, va) in enumerate(folds):
        assert not set(dev[tr]) & set(dev[va]) and len(tr) + len(va) == len(dev)
        fold_ids[va] = f
        stored = json.loads((ROOT / f"experiments/v7_astra/guards/fold_{f}.json").read_text())
        expected = guard_for_fold(urls, y, dev, tr, stored, cfg["input_sha256"]["data/content_train.joblib"])
        origins.append({"fold": f, "guard": expected, "train_indices_sha256": array_sha(dev[tr]), "validation_indices_sha256": array_sha(dev[va])})
    np.testing.assert_array_equal(fold_ids, recorded_folds)
    write("fold_scopes_before_retrieval.json", {"origins": origins, "development_indices_sha256": array_sha(dev)})
    eligible = {name: (p >= cfg["eligible_probability"][0]) & (p <= cfg["eligible_probability"][1]) for name, p in baseline.items()}
    query_mask = eligible["v15"] | eligible["v10"]
    delta = np.zeros(len(y)); coverage = np.zeros(len(y), dtype=np.int8)
    reports = []
    with threadpool_limits(limits=2):
        for f, (tr, va) in enumerate(folds):
            tick = time.perf_counter()
            q = va[query_mask[va]]
            emit("fold_start", fold=f, train_rows=len(tr), validation_rows=len(va), uncertain_queries=len(q))
            stored = joblib.load(ROOT / f"experiments/v7_astra/models/fold_{f}.joblib")
            assert stored["guard"] == origins[f]["guard"]
            model = stored["model"]; del stored
            assert model.get_params() == FullVocabularyTextModel().get_params()
            assert model.training_rows_ == model.vocabulary_fit_rows_ == len(tr)
            assert model.classifier_.C == 16 and model.ratio_ is None and not model.nb
            fingerprint = joblib.hash(([(k, v.vocabulary_, v.idf_) for k, v in model.vectorizers_.items()], model.classifier_.coef_, model.classifier_.intercept_))
            X, train_p = transform_train(model, records, tr, cfg)
            assert X.shape == (len(tr), model.n_features_)
            transpose = X.T.tocsr()
            k = cfg["neighbors"]
            sims = np.zeros((len(q), k), dtype=np.float32)
            neighbors = np.full((len(q), k), -1, dtype=np.int64)
            neighbor_y = np.full((len(q), k), -1, dtype=np.int8)
            neighbor_p = np.zeros((len(q), k))
            reproduced = []
            dense_rows = 0
            for start in range(0, len(q), cfg["query_batch"]):
                positions = q[start:start + cfg["query_batch"]]
                Q = model._matrix([records[int(i)] for i in positions])
                cp = model.classifier_.predict_proba(Q)[:, 1]
                np.testing.assert_allclose(cp, original_c16[positions], atol=1e-12, rtol=0)
                reproduced.extend(cp.tolist())
                Q = normalize(Q, norm="l2", copy=False)
                found = sp_matmul_topn(Q, transpose, top_n=k, threshold=cfg["minimum_similarity"], sort=True, n_threads=2)
                for row in range(len(positions)):
                    a, b = found.indptr[row:row + 2]
                    scores, indices = found.data[a:b], found.indices[a:b]
                    assert len(indices) <= k and np.all(scores[:-1] >= scores[1:])
                    assert len(set(indices.tolist())) == len(indices)
                    dest = start + row
                    sims[dest, :len(indices)] = scores
                    neighbors[dest, :len(indices)] = dev[tr[indices]]
                    neighbor_y[dest, :len(indices)] = y[tr[indices]]
                    neighbor_p[dest, :len(indices)] = train_p[indices]
                    assert dev[positions[row]] not in neighbors[dest]
                    if dense_rows < 8:
                        dense = (Q[row] @ transpose).toarray().ravel()
                        expected = np.sort(dense[dense > cfg["minimum_similarity"]])[::-1][:k]
                        np.testing.assert_allclose(scores, expected, atol=2e-6, rtol=0)
                        dense_rows += 1
                if start % (cfg["query_batch"] * 8) == 0:
                    emit("retrieval_progress", fold=f, completed=start + len(positions), total=len(q))
            valid = neighbors >= 0
            assert np.isin(neighbors[valid], dev[tr]).all()
            assert not np.isin(neighbors[valid], dev[va]).any()
            feat = features(sims, neighbor_y, neighbor_p, cfg)
            delta[q] = feat["delta"]
            coverage[va] += 1
            assert fingerprint == joblib.hash(([(k, v.vocabulary_, v.idf_) for k, v in model.vectorizers_.items()], model.classifier_.coef_, model.classifier_.intercept_))
            np.savez_compressed(OUT / f"fold_{f}_retrieval.npz", train_indices=dev[tr], validation_indices=dev[va],
                                query_indices=dev[q], neighbor_indices=neighbors, similarities=sims,
                                neighbor_labels=neighbor_y, neighbor_c16_probability=neighbor_p,
                                query_c16_probability=np.array(reproduced), **feat)
            report = {"fold": f, "train_rows": len(tr), "validation_rows": len(va), "queries": len(q),
                      "supported_queries": int((feat["support"] >= cfg["minimum_support"]).sum()),
                      "nonzero_deltas": int((feat["delta"] != 0).sum()), "maximum_absolute_delta": float(abs(feat["delta"]).max()),
                      "matrix_shape": list(X.shape), "matrix_nnz": int(X.nnz), "dense_check_queries": dense_rows,
                      "train_absolute_residual_mean": float(abs(y[tr] - train_p).mean()),
                      "train_residual_quantiles": np.quantile(y[tr] - train_p, [0, .01, .5, .99, 1]).tolist(),
                      "zero_train_validation_overlap": True, "zero_self_neighbors": True, "all_neighbor_labels_from_train": True,
                      "frozen_c16_predictions_reproduced": True, "representation_and_coefficients_unchanged": True,
                      "seconds": time.perf_counter() - tick, "threadpools": threadpool_info()}
            assert all(p["num_threads"] <= 2 for p in report["threadpools"])
            reports.append(report); write(f"fold_{f}_execution.json", report)
            emit("fold_complete", **{k: report[k] for k in ("fold", "queries", "supported_queries", "maximum_absolute_delta", "seconds")})
            del X, transpose, model, train_p, Q, found, neighbors, neighbor_y, neighbor_p, sims
            gc.collect()
    assert (coverage == 1).all()
    candidate = {name: np.where(eligible[name], np.clip(p + delta, 0, 1), p) for name, p in baseline.items()}
    for name, p in candidate.items():
        np.testing.assert_array_equal(p[~eligible[name]], baseline[name][~eligible[name]])
        assert np.isfinite(p).all() and np.all(abs(p - baseline[name]) <= .10000000001)
    cuts = {name: best_threshold(y, p)[0] for name, p in {**baseline, **{f"neighbor_{k}": v for k, v in candidate.items()}}.items()}
    comparisons = {}
    for name, p in candidate.items():
        base = baseline[name]
        comparisons[name] = {
            "frozen_v15": comparison(y, base, p, cfg["frozen_v15_threshold"], cfg["frozen_v15_threshold"], dev, fold_ids),
            "individually_tuned": comparison(y, base, p, cuts[name], cuts[f"neighbor_{name}"], dev, fold_ids),
            "same_candidate_tuned_threshold": comparison(y, base, p, cuts[f"neighbor_{name}"], cuts[f"neighbor_{name}"], dev, fold_ids),
            "frozen_v10": comparison(y, base, p, cfg["frozen_v10_threshold"], cfg["frozen_v10_threshold"], dev, fold_ids),
        }
    cross = {
        "frozen_v15": comparison(y, baseline["v10"], candidate["v15"], cfg["frozen_v15_threshold"], cfg["frozen_v15_threshold"], dev, fold_ids),
        "individually_tuned": comparison(y, baseline["v10"], candidate["v15"], cuts["v10"], cuts["neighbor_v15"], dev, fold_ids),
    }
    decisions = []
    for mode in cfg["gate"]["modes"]:
        row = comparisons["v15"][mode]
        decisions.append(row["net_fewer_errors"] >= 5 and row["after"]["f1"] > row["before"]["f1"] and
                         sum(r["after_errors"] < r["before_errors"] for r in row["folds"]) >= 2 and
                         all(r["after_errors"] <= r["before_errors"] + 2 for r in row["folds"]))
    for p, h in cfg["input_sha256"].items():
        assert sha(ROOT / p) == h, "Input changed during run: " + p
    np.savez_compressed(OUT / "development_oof.npz", indices=dev, labels=y, folds=fold_ids,
                        baseline_v15=baseline["v15"], baseline_v10=baseline["v10"],
                        neighbor_v15=candidate["v15"], neighbor_v10=candidate["v10"],
                        retrieval_delta=delta, queried=query_mask, eligible_v15=eligible["v15"], eligible_v10=eligible["v10"])
    write("results.json", {"variant": cfg["variant"], "gate_passed": all(decisions), "rows": len(y),
                           "comparisons": comparisons, "neighbor_v15_vs_v10": cross,
                           "thresholds": cuts, "folds": reports, "limitations": cfg["limitations"],
                           "eligible_rows": {k: int(v.sum()) for k, v in eligible.items()},
                           "protocol_sha256": sha(OUT / "preregistered.json"), "source_sha256": sha(__file__),
                           "status": "development gate passed" if all(decisions) else "negative development result; retain v15",
                           "seconds": time.perf_counter() - started})
    write("execution.json", {"project_files_read": sorted(READS), "project_files_written": sorted(WRITES | {"experiments/v16_neighbor/execution.json"}),
                             "all_inputs_unchanged": True, "original_development_alignment_verified": True,
                             "classifier_fits": 0, "vectorizer_fits": 0, "nonparametric_index_fits": 3,
                             "all_rows_covered_once": True, "formula_checks_passed": True,
                             "network_used": False, "read_scope": "Exact allowlist enforced before imports and dataset reads",
                             "seconds": time.perf_counter() - started})
    emit("complete", gate_passed=all(decisions), comparisons={name: {mode: {k: row[k] for k in ("net_fewer_errors", "fixed_errors", "new_errors")} for mode, row in modes.items()} for name, modes in comparisons.items()})


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    elif sys.argv[1:] == []:
        main()
    else:
        raise SystemExit("Use --prepare once, then run without arguments")
