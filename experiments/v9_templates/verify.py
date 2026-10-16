"""Independent saved-artifact checks; no fit, feature tuning or dataset loads."""
import run as screen  # Installs the same audit policy and two-thread limits.
import gzip
import json
import hashlib
import numpy as np
from templates import TemplateEvidence

OUT = screen.OUT


def read(name):
    return json.loads((OUT / name).read_text())


def check():
    frozen = read("frozen_run.json")
    for path, expected in frozen["sources"].items():
        assert screen.sha(screen.ROOT / path) == expected
    protocol = frozen["protocol"]
    assert len(protocol["configurations"]) == 3 and protocol["classifiers_fit"] == 0
    with np.load(OUT / "inner_split.npz", allow_pickle=False) as a:
        tr, va = a["inner_train"], a["inner_validation"]
    assert len(tr) == 34663 and len(va) == 6118 and set(tr).isdisjoint(va)
    saved = read("best_model.json")
    selection = read("selection.json")
    provenance = read("provenance.json")
    assert saved["guard"]["provenance_sha256"] == screen.sha(OUT / "provenance.json")
    assert saved["guard"]["protocol_sha256"] == screen.sha(OUT / "protocol.json")
    assert saved["guard"]["split"] == provenance["split"]
    restored = TemplateEvidence.from_dict(saved["model"])
    with gzip.open(OUT / "training_key_ledger.json.gz", "rt", encoding="utf-8") as handle:
        ledger = json.load(handle)
    train_set = set(int(i) for i in tr)
    row_url = {}
    reasons = {}
    for key, entry in ledger.items():
        ids = entry["training_indices"]
        assert set(ids) <= train_set and len(ids) == len(set(ids)) == entry["support"]
        assert len(entry["training_url_sha256"]) == len(set(entry["training_url_sha256"])) == entry["support"]
        assert len(entry["training_domain_sha256"]) == len(set(entry["training_domain_sha256"])) == entry["domain_count"]
        assert sum(entry["counts"]) == entry["support"]
        for rid, url_sha in zip(ids, entry["training_url_sha256"], strict=True):
            assert row_url.setdefault(rid, url_sha) == url_sha
        expected = "conflict" if min(entry["counts"]) > 0 else "urls" if entry["support"] < 3 else "domains" if entry["domain_count"] < 2 else "accepted"
        assert entry["status"] == expected
        if expected == "accepted": assert entry["label"] == int(entry["counts"][1] > 0)
        reasons[expected] = reasons.get(expected, 0) + 1
    for key, entry in restored.evidence.items():
        assert ledger[key] == entry and entry["status"] == "accepted"
    expected_keys = {k for k, e in ledger.items() if e["status"] == "accepted" and k.split(":")[0] in restored.config["families"]}
    assert set(restored.evidence) == expected_keys
    results = read("results.json")
    base_metric = results["baseline"]
    t = base_metric["threshold"]
    reference_base = None
    for result in results["candidates"]:
        with np.load(OUT / (result["name"] + "_inner.npz"), allow_pickle=False) as a:
            np.testing.assert_array_equal(a["indices"], va)
            y, p, base = a["labels"], a["probabilities"], a["baseline"]
            if reference_base is None: reference_base = base.copy()
            np.testing.assert_array_equal(base, reference_base)
            for policy, threshold in (("fixed", t), ("optimum", result["optimum"]["threshold"])):
                pred = p >= threshold
                tp = int(((y == 1) & pred).sum())
                fp = int(((y == 0) & pred).sum())
                fn = int(((y == 1) & ~pred).sum())
                assert result[policy]["errors"] == fp + fn
                assert abs(result[policy]["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-14
                assert result[policy]["fp"] == fp and result[policy]["fn"] == fn
            fixed = p >= t
            changed = fixed != (base >= t)
            assert result["corrected"] == int((changed & (fixed == y)).sum())
            assert result["damaged"] == int((changed & ((base >= t) == y)).sum())
            assert result["net_fewer_errors"] == result["corrected"] - result["damaged"]
            exact = (base == 0) | (base == 1)
            np.testing.assert_array_equal(p[exact], base[exact])
            np.testing.assert_array_equal(p[a["conflict"]], base[a["conflict"]])
            assert result["gate_passed"] == bool(base_metric["errors"] - result["fixed"]["errors"] >= 2 and result["fixed"]["f1"] >= base_metric["f1"])
    winner = min(results["candidates"], key=lambda r: (r["fixed"]["errors"], -r["fixed"]["f1"], r["order"]))
    assert selection["winner"] == winner["name"] and selection["gate_passed"] == winner["gate_passed"]
    assert selection["winner_result"] == winner
    reload = read("reload_check.json")
    assert reload["rows"] == len(va) and reload["probability_bitwise_equal"] and reload["regenerated_real_keys"]
    assert reload["model_sha256"] == screen.sha(OUT / "best_model.json")
    execution = read("execution.json")
    assert execution["original_inputs_and_sources_unchanged"]
    diagnostic = read("import_audit_diagnostic.json")
    assert execution["denied_attempts"] == diagnostic["denied_by_run_audit"] == ["socket.__new__"]
    assert len(diagnostic["events"]) == 1
    assert "_has_ipv6" in "".join(diagnostic["events"][0]["stack"])
    assert not any(event in execution["denied_attempts"] for event in ("socket.connect", "socket.getaddrinfo", "socket.bind"))
    assert execution["classifier_fits"] == 0 and execution["configuration_count"] == 3
    assert all(pool["num_threads"] <= 2 for pool in execution["threadpools"])
    # Direct synthetic policy events only. No attempted filesystem/network I/O.
    checked = 0
    for event, args in (("open", (str(screen.ROOT / "external/forbidden"), "r", 0)), ("open", (str(screen.ROOT / "outside-sidecar"), "w", 1)), ("socket.connect", (None, None)), ("subprocess.Popen", (None, None, None, None))):
        try:
            screen.audit(event, args)
        except PermissionError:
            checked += 1
        else:
            raise AssertionError("Audit allowed prohibited synthetic event")
    result = dict(status="passed", training_keys_checked=len(ledger), key_status_counts=reasons, winner_evidence_keys=len(restored.evidence), referenced_training_rows=len(row_url), candidates_recomputed=3, serialized_guard_and_frozen_sources_rechecked=True, synthetic_audit_events_rejected=checked, blocked_import_ipv6_probe_confirmed=True, initial_verifier_failure="Overstrict empty-denial-list assertion; corrected after import-only stack diagnosis, without changing frozen experiment", no_dataset_reload=True, no_new_fit=True, no_new_selection=True)
    screen.write("verification.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    with screen.threadpool_limits(limits=2): check()
