"""Independent replay of saved own-development evidence, without models/cache."""
import sys
sys.dont_write_bytecode = True
import run as experiment  # Reuse the enforced input/write/network boundary only.
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

OUT, ROOT = experiment.OUT, experiment.ROOT


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_comparison(row, labels, base, candidate, indices, fold_ids):
    def check_metrics(saved, y, p):
        predicted = p >= saved["threshold"]
        actual = {"tp": int(((predicted == 1) & (y == 1)).sum()),
                  "tn": int(((predicted == 0) & (y == 0)).sum()),
                  "fp": int(((predicted == 1) & (y == 0)).sum()),
                  "fn": int(((predicted == 0) & (y == 1)).sum())}
        assert all(saved[k] == v for k, v in actual.items())
        assert saved["errors"] == actual["fp"] + actual["fn"]
        f1 = 2 * actual["tp"] / (2 * actual["tp"] + actual["fp"] + actual["fn"])
        assert abs(f1 - saved["f1"]) < 1e-14
        return predicted
    before = check_metrics(row["before"], labels, base)
    after = check_metrics(row["after"], labels, candidate)
    fixed, new = (before != labels) & (after == labels), (before == labels) & (after != labels)
    assert row["fixed_indices"] == indices[fixed].tolist()
    assert row["new_error_indices"] == indices[new].tolist()
    assert row["fixed_errors"] == int(fixed.sum()) and row["new_errors"] == int(new.sum())
    assert row["net_fewer_errors"] == int(fixed.sum()) - int(new.sum())
    for fold in row["folds"]:
        m = fold_ids == fold["fold"]
        assert fold["before_errors"] == int((before[m] != labels[m]).sum())
        assert fold["after_errors"] == int((after[m] != labels[m]).sum())
        assert fold["fixed"] == int(fixed[m].sum()) and fold["new"] == int(new[m].sum())


def main():
    cfg = json.loads((OUT / "preregistered.json").read_text())
    result = json.loads((OUT / "results.json").read_text())
    execution = json.loads((OUT / "execution.json").read_text())
    assert digest(OUT / "preregistered.json") == result["protocol_sha256"]
    assert digest(OUT / "run.py") == cfg["source_sha256"] == result["source_sha256"]
    data = dict(np.load(OUT / "development_oof.npz", allow_pickle=False))
    dev, y, fold_ids = data["indices"], data["labels"], data["folds"]
    lookup = {int(i): position for position, i in enumerate(dev)}
    delta = np.zeros(len(y))
    queries_seen = np.zeros(len(y), dtype=np.int8)
    max_replay_error, neighbor_count = 0., 0
    fold_checks = []
    for f, (tr, va) in enumerate(StratifiedKFold(**cfg["splitter"]).split(dev, y)):
        assert (fold_ids[va] == f).all()
        a = dict(np.load(OUT / f"fold_{f}_retrieval.npz", allow_pickle=False))
        np.testing.assert_array_equal(a["train_indices"], dev[tr])
        np.testing.assert_array_equal(a["validation_indices"], dev[va])
        expected_q = va[data["queried"][va]]
        np.testing.assert_array_equal(a["query_indices"], dev[expected_q])
        queries_seen[expected_q] += 1
        good = a["neighbor_indices"] >= 0
        assert np.isin(a["neighbor_indices"][good], dev[tr]).all()
        assert not np.isin(a["neighbor_indices"][good], dev[va]).any()
        assert not (a["neighbor_indices"] == a["query_indices"][:, None]).any()
        expected_y = np.array([y[lookup[int(i)]] for i in a["neighbor_indices"][good]], dtype=np.int8)
        np.testing.assert_array_equal(expected_y, a["neighbor_labels"][good])
        assert (a["neighbor_labels"][~good] == -1).all()
        assert np.all(a["similarities"][good] >= cfg["minimum_similarity"])
        neighbor_count += int(good.sum())
        # Independently accumulate each weighted residual, not difference of means.
        for q in range(len(expected_q)):
            mask = good[q]
            weights = a["similarities"][q, mask].astype(float) ** cfg["power"]
            residuals = a["neighbor_labels"][q, mask].astype(float) - a["neighbor_c16_probability"][q, mask]
            raw = float(np.dot(weights, residuals) / (cfg["smoothing"] + weights.sum()))
            correction = float(np.clip(cfg["residual_weight"] * raw, -cfg["maximum_absolute_delta"], cfg["maximum_absolute_delta"])) if int(mask.sum()) >= cfg["minimum_support"] else 0.
            max_replay_error = max(max_replay_error, abs(correction - float(a["delta"][q])))
            assert abs(correction - float(a["delta"][q])) < 2e-7
            assert int(mask.sum()) == a["support"][q]
        delta[expected_q] = a["delta"]
        fold_checks.append({"fold": f, "query_rows": len(expected_q), "neighbor_edges": int(good.sum()), "scope_and_labels_verified": True})
    np.testing.assert_array_equal(queries_seen, data["queried"].astype(np.int8))
    np.testing.assert_array_equal(delta, data["retrieval_delta"])
    comparison_count = 0
    for name in ("v15", "v10"):
        p = data[f"baseline_{name}"]
        eligible = (p >= cfg["eligible_probability"][0]) & (p <= cfg["eligible_probability"][1])
        np.testing.assert_array_equal(eligible, data[f"eligible_{name}"])
        replay = np.where(eligible, np.clip(p + delta, 0, 1), p)
        np.testing.assert_array_equal(replay, data[f"neighbor_{name}"])
        for comparison in result["comparisons"][name].values():
            check_comparison(comparison, y, p, replay, dev, fold_ids)
            comparison_count += 1
    for comparison in result["neighbor_v15_vs_v10"].values():
        check_comparison(comparison, y, data["baseline_v10"], data["neighbor_v15"], dev, fold_ids)
        comparison_count += 1
    selected = json.loads((ROOT / "reports/v15/selection.json").read_text())
    actual = result["comparisons"]["v15"]["frozen_v15"]["before"]
    for field in ("tp", "fp", "tn", "fn", "rows", "threshold", "f1"):
        assert actual[field] == selected["selected"]["metrics"][field]
    actual = result["comparisons"]["v10"]["frozen_v10"]["before"]
    for field in ("tp", "fp", "tn", "fn", "rows", "threshold", "f1"):
        assert actual[field] == selected["baseline_v10"][field]
    gate = cfg["gate"]
    passes = []
    for mode in gate["modes"]:
        row = result["comparisons"]["v15"][mode]
        passes.append(row["net_fewer_errors"] >= gate["net_fewer_errors_vs_v15_at_each_mode"] and
                      row["after"]["f1"] > row["before"]["f1"] and
                      sum(f["after_errors"] < f["before_errors"] for f in row["folds"]) >= gate["improved_folds_at_each_mode"] and
                      all(f["after_errors"] - f["before_errors"] <= gate["maximum_fold_error_increase_at_each_mode"] for f in row["folds"]))
    assert result["gate_passed"] == all(passes)
    assert all(Path(p).is_relative_to("experiments/v16_neighbor") for p in execution["project_files_written"])
    assert all(p in experiment.INPUTS or p.startswith("experiments/v16_neighbor/") or p.startswith("phishing_url/__pycache__/") for p in execution["project_files_read"])
    assert sum(f["dense_check_queries"] for f in result["folds"]) == 24
    assert all(all(t["num_threads"] <= 2 for t in f["threadpools"]) for f in result["folds"])
    report = {"status": "passed", "neighbor_edges_verified": neighbor_count,
              "all_queried_rows_replayed": int(queries_seen.sum()), "folds": fold_checks,
              "independent_float64_residual_max_abs_difference": max_replay_error,
              "saved_feature_probability_replay_bitwise_identical": True,
              "metric_comparisons_verified": comparison_count, "baseline_matches_v15_selection": True,
              "no_train_validation_or_self_neighbors": True, "neighbor_labels_match_own_train_indices": True,
              "gate_recomputed": all(passes), "thread_limit_verified": True, "write_scope_verified": True,
              "protocol_sha256": digest(OUT / "preregistered.json"), "verifier_sha256": digest(__file__),
              "artifact_sha256": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ("verification.json", "REPORT.md")}}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
