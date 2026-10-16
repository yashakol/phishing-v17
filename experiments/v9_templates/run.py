"""Run the single frozen, offline inner screen; never proceed to OOF."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gc
import gzip
import hashlib
import json
import resource
import time

INPUTS = {
    "data/content_train.joblib", "data/extra_html_train.joblib",
    "reports/clean_train.csv", "reports/split_indices.npz",
    "experiments/v7_astra/run.py", "experiments/v7_astra/inner_full_vocabulary.npz",
    "experiments/v7_astra/inner_split.npz", "experiments/v7_astra/guards/inner_full_vocabulary.json",
    "experiments/v5/cat_inner_1.npz", "experiments/v5/cat_inner_1.json",
    "experiments/v5/screen_catboost.py", "experiments/v8_astra_regularization/inner_results.json"
}
READS = set()
DENIED = []


def audit(event, args):
    if event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn")) or event in ("os.system", "os.fork", "pty.spawn"):
        DENIED.append(event)
        raise PermissionError("Offline sidecar prohibits network/process execution")
    if event in ("os.remove", "os.rmdir", "os.mkdir", "os.rename"):
        for raw in args[:2] if event == "os.rename" else args[:1]:
            if isinstance(raw, (str, bytes, os.PathLike)) and not Path(os.fsdecode(raw)).resolve().is_relative_to(OUT):
                raise PermissionError("Filesystem mutation outside sidecar")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)): return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing and not path.is_relative_to(OUT): raise PermissionError("Write outside sidecar: " + str(path))
    if path.is_relative_to(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(".venv/"): return
        own = path.is_relative_to(OUT)
        core = rel.startswith("phishing_url/") and path.suffix in (".py", ".pyc") and "reference" not in rel
        if not own and not core and rel not in INPUTS:
            DENIED.append(rel)
            raise PermissionError("Input outside allowlist: " + rel)
        if not writing: READS.add(rel)


sys.addaudithook(audit)
import joblib
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.content_validation import aligned_splits
from phishing_url.content_checkpoints import guard
from phishing_url.evaluation import best_threshold, measure
from phishing_url.features import group_for_url
from templates import TemplateEvidence, signature_keys, normalize, digest, FAMILIES


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)


def measure_errors(y, p, threshold):
    metric = measure(y, p, threshold)
    metric["errors"] = metric["fp"] + metric["fn"]
    return metric


def metrics(y, p, threshold):
    optimum, _ = best_threshold(y, p)
    return dict(fixed=measure_errors(y, p, threshold), optimum=measure_errors(y, p, optimum))


def load_prediction(name, indices, labels, training=None):
    with np.load(ROOT / name, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["indices"], indices)
        np.testing.assert_array_equal(archive["labels"], labels)
        if training is not None: np.testing.assert_array_equal(archive["train_indices"], training)
        p = archive["probabilities"].copy()
    assert p.shape == labels.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def contract_checks():
    # Synthetic invariants only, no dataset labels or extra screened configurations.
    first = 'nonce="abc"; user="alice"; const u="https://one.example/a123.js?v=120&email=a@one.example"; var token="secretA"; margin:12px;'
    second = 'nonce="xyz"; user="bob"; const u="https://two.example/a987.js?v=987&email=b@two.example"; var token="secretB"; margin:99px;'
    assert normalize(first) == normalize(second)
    assert signature_keys({"url": "https://a.example", "title": "Welcome", "markup": "tag_html tag_body"}, {"url": "https://a.example"}) == {}
    keys = [{"title_js": "title_js:synthetic"}] * 4
    model = TemplateEvidence(families=["title_js"])
    model.fit(keys[:3], ["a", "b", "c"], ["a.example", "b.example", "c.example"], [1, 1, 1], [0, 1, 2])
    base = np.array([0., 1., .5, .5])
    out, _, _ = model.predict_keys(keys, base, [False, False, True, False])
    np.testing.assert_array_equal(out[:3], base[:3])
    assert out[3] > .5
    model.fit(keys[:3], ["a", "b", "c"], ["a.example", "b.example", "c.example"], [1, 0, 1], [0, 1, 2])
    assert not model.evidence and model.statistics["title_js"]["conflicting_keys"] == 1
    return dict(normalization_invariance=True, empty_boilerplate_excluded=True, exact_hard_and_explicit_mask_preserved=True, contradictory_training_key_excluded=True)


def main():
    started = time.perf_counter()
    if (OUT / "frozen_run.json").exists(): raise RuntimeError("No adaptive rerun: frozen_run.json already exists")
    protocol = json.loads((OUT / "protocol.json").read_text())
    own_sources = {f"experiments/v9_templates/{name}": sha(OUT / name) for name in ("protocol.json", "templates.py", "run.py")}
    # Persist code and policy hashes BEFORE reading any data or saved predictions.
    frozen = dict(protocol=protocol, sources=own_sources, unix_time=time.time(), order="Frozen before dataset loading, normalization, fitting and scoring")
    write("frozen_run.json", frozen)
    checks = contract_checks()
    emit("frozen", configurations=3, classifiers=0, contract_checks=checks)
    inputs = {name: sha(ROOT / name) for name in sorted(INPUTS)}
    cache = joblib.load(ROOT / "data/content_train.joblib")
    assert len(cache["records"]) == len(cache["labels"]) == len(cache["urls"]) == 63900
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    assert len(tr) == 34663 and len(va) == 6118
    assert len(set(tr)) == len(tr) and len(set(va)) == len(va) and not set(tr) & set(va)
    assert set(tr) | set(va) <= set(splits["development"])
    assert not (set(tr) | set(va)) & (set(splits["holdout"]) | set(splits["added"]))
    with np.load(ROOT / "experiments/v7_astra/inner_split.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["inner_train"], tr)
        np.testing.assert_array_equal(saved["inner_validation"], va)
    g = json.loads((ROOT / "experiments/v7_astra/guards/inner_full_vocabulary.json").read_text())
    expected_guard = guard(cache, tr, g["config"], inputs["data/content_train.joblib"])
    expected_guard["ordered_training_indices_sha256"] = hashlib.sha256(tr.tobytes()).hexdigest()
    assert g == expected_guard and g["config"]["parameters"]["c"] == 16.
    source_hashes = {name: sha(ROOT / name) for name in g["config"]["sources"]}
    assert source_hashes == g["config"]["sources"]
    for name in ("phishing_url/features.py", "phishing_url/v7_signatures.py", "phishing_url/v8_resource_text.py"):
        source_hashes[name] = sha(ROOT / name)
    ytr, y = cache["labels"][tr].copy(), cache["labels"][va].copy()
    text = load_prediction(protocol["baseline"]["text"], va, y, tr)
    cat = load_prediction(protocol["baseline"]["numeric"], va, y)
    base = .5 * text + .5 * cat
    threshold, _ = best_threshold(y, base)
    baseline = measure_errors(y, base, threshold)
    prior_baseline = json.loads((ROOT / "experiments/v8_astra_regularization/inner_results.json").read_text())["baseline"]["mix"]
    assert baseline == prior_baseline, (baseline, prior_baseline)
    all_urls = list(cache["urls"])
    selected = np.concatenate([tr, va])
    content = [cache["records"][int(i)] for i in selected]
    selected_raw_hashes = [cache["hashes"][int(i)] for i in selected]
    for i, record in zip(selected, content, strict=True): assert record["url"] == all_urls[int(i)]
    source_metadata = dict(content=cache["source"])
    split_provenance = dict(train_rows=len(tr), validation_rows=len(va), development_rows=len(splits["development"]), disjoint=True, excluded_rows_unused=True, train_sha256=hashlib.sha256(tr.tobytes()).hexdigest(), validation_sha256=hashlib.sha256(va.tobytes()).hexdigest(), train_labels_sha256=hashlib.sha256(ytr.tobytes()).hexdigest(), validation_labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(), selected_html_sha256=digest("\n".join(selected_raw_hashes)))
    del cache, splits, selected_raw_hashes
    gc.collect()
    emit("baseline", **baseline)
    extra_cache = joblib.load(ROOT / "data/extra_html_train.joblib")
    assert len(extra_cache["records"]) == 63900 and list(extra_cache["urls"]) == all_urls
    np.testing.assert_array_equal(extra_cache["labels"][tr], ytr)
    np.testing.assert_array_equal(extra_cache["labels"][va], y)
    extra = [extra_cache["records"][int(i)] for i in selected]
    source_metadata["extra"] = extra_cache["source"]
    assert source_metadata["extra"]["original_raw_hashes_verified"] is True
    del extra_cache, all_urls
    gc.collect()
    load_seconds = time.perf_counter() - started
    key_started = time.perf_counter()
    keys = []
    for i, (record, additional) in enumerate(zip(content, extra, strict=True)):
        keys.append(signature_keys(record, additional))
        if (i + 1) % 5000 == 0: emit("keys", rows=i + 1, total=len(content), seconds=time.perf_counter()-key_started)
    key_seconds = time.perf_counter() - key_started
    train_keys, valid_keys = keys[:len(tr)], keys[len(tr):]
    training_urls = [record["url"] for record in content[:len(tr)]]
    validation_urls = [record["url"] for record in content[len(tr):]]
    assert len(set(training_urls)) == len(tr) and not set(training_urls) & set(validation_urls)
    # PSL lookup is offline and uses only inner_train URLs for domain support.
    domains = [group_for_url(url) for url in training_urls]
    coverage = {family: dict(train_rows=sum(family in row for row in train_keys), validation_rows=sum(family in row for row in valid_keys), unique_train_keys=len({row[family] for row in train_keys if family in row})) for family in FAMILIES}
    provenance = dict(inputs=inputs, sources={**source_hashes, **own_sources}, c16_guard=g, split=split_provenance, extractor_metadata=source_metadata, constraints=protocol, baseline_origin="saved aligned C16 probabilities; original source/data/training guard verified; saved classifier not loaded or rerun", numeric_origin="saved aligned deep numeric predictions; no training-scope guard saved in that archive; source screen_catboost.py uses same aligned_splits inner train/validation and validation early stopping", reused_validation=True)
    write("provenance.json", provenance)
    write("key_coverage.json", coverage)
    np.savez_compressed(OUT / "inner_split.npz", inner_train=tr, inner_validation=va)
    results = []
    best_model, best_output, best_flags, best_matches, best_result = None, None, None, None, None
    ledger_written = False
    for order, spec in enumerate(protocol["configurations"]):
        spec = dict(spec)
        name = spec.pop("name")
        fit_started = time.perf_counter()
        model = TemplateEvidence(**protocol["shared_policy"], **spec).fit(train_keys, training_urls, domains, ytr, tr)
        fit_seconds = time.perf_counter()-fit_started
        # All retained/rejected evidence references only training indices.
        allowed = set(int(i) for i in tr)
        assert all(set(entry["training_indices"]) <= allowed for entry in model.ledger.values())
        if len(spec["families"]) == len(FAMILIES) and not ledger_written:
            with gzip.open(OUT / "training_key_ledger.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(model.ledger, handle, sort_keys=True)
            ledger_written = True
        predict_started = time.perf_counter()
        p, flags, matches = model.predict_keys(valid_keys, base)
        prediction_seconds = time.perf_counter()-predict_started
        scores = metrics(y, p, threshold)
        changed = (p >= threshold) != (base >= threshold)
        corrected = changed & ((p >= threshold) == y)
        damaged = changed & ((base >= threshold) == y)
        assert int(corrected.sum()-damaged.sum()) == baseline["errors"]-scores["fixed"]["errors"]
        family_coverage = {f: dict(accepted_keys=model.statistics[f]["accepted_keys"], validation_matching_rows=sum(row.get(f) in model.evidence for row in valid_keys)) for f in spec["families"]}
        result = dict(name=name, order=order, configuration=model.config, **scores, training_statistics=model.statistics, family_coverage=family_coverage, flags={k: int(v.sum()) for k, v in flags.items()}, corrected=int(corrected.sum()), damaged=int(damaged.sum()), changed_labels=int(changed.sum()), net_fewer_errors=baseline["errors"]-scores["fixed"]["errors"], gate_passed=bool(baseline["errors"]-scores["fixed"]["errors"] >= 2 and scores["fixed"]["f1"] >= baseline["f1"]), diagnostic_optimum_gate=bool(baseline["errors"]-scores["optimum"]["errors"] >= 2 and scores["optimum"]["f1"] >= baseline["f1"]), fit_seconds=fit_seconds, prediction_seconds=prediction_seconds)
        results.append(result)
        np.savez_compressed(OUT / f"{name}_inner.npz", indices=va, labels=y, probabilities=p, baseline=base, **flags)
        write(f"{name}_results.json", result)
        write("results.json", dict(baseline=baseline, candidates=results))
        rank = lambda row: (row["fixed"]["errors"], -row["fixed"]["f1"], row["order"])
        if best_result is None or rank(result) < rank(best_result):
            best_model, best_output, best_flags, best_matches, best_result = model.to_dict(), p, flags, matches, result
        del model
        gc.collect()
        emit("candidate", name=name, fixed=scores["fixed"], optimum=scores["optimum"], corrected=result["corrected"], damaged=result["damaged"], flags=result["flags"], gate_passed=result["gate_passed"])
    write("best_model.json", dict(model=best_model, guard=dict(provenance_sha256=sha(OUT/"provenance.json"), protocol_sha256=own_sources["experiments/v9_templates/protocol.json"], split=split_provenance, frozen_sources=own_sources)))
    reload_started = time.perf_counter()
    saved = json.loads((OUT / "best_model.json").read_text())
    assert saved["guard"]["provenance_sha256"] == sha(OUT/"provenance.json")
    restored = TemplateEvidence.from_dict(saved["model"])
    # Regenerate every validation key from real static HTML views, not key cache.
    after, after_flags, after_matches = restored.predict(content[len(tr):], extra[len(tr):], base)
    np.testing.assert_array_equal(after, best_output)
    for name in best_flags: np.testing.assert_array_equal(after_flags[name], best_flags[name])
    assert after_matches == best_matches
    matched_examples = []
    for j, hit in enumerate(best_matches):
        if hit:
            matched_examples.append(dict(validation_index=int(va[j]), url_sha256=digest(validation_urls[j]), keys=hit, baseline=float(base[j]), candidate=float(best_output[j]), baseline_label=int(base[j] >= threshold), candidate_label=int(best_output[j] >= threshold), validation_label=int(y[j]), adjusted=bool(best_flags["adjusted"][j]), conflict=bool(best_flags["conflict"][j])))
    write("validation_match_provenance.json", matched_examples)
    reload_check = dict(rows=len(va), matched_rows=len(matched_examples), regenerated_real_keys=True, probability_bitwise_equal=True, flags_equal=True, matched_keys_equal=True, max_abs_difference=float(np.abs(after-best_output).max()), seconds=time.perf_counter()-reload_started, model_sha256=sha(OUT/"best_model.json"))
    write("reload_check.json", reload_check)
    selection = dict(gate_passed=best_result["gate_passed"], winner=best_result["name"], baseline=baseline, winner_result=best_result, threshold_policy="fixed baseline optimum; candidate optimum diagnostics only", classifier_fits=0, screened_configurations=3, original_inner_only=True, oof_authorized=False, public_improvement_claimed=False, model="experiments/v9_templates/best_model.json", protocol_sha256=own_sources["experiments/v9_templates/protocol.json"], status="eligible for parent-requested OOF only" if best_result["gate_passed"] else "negative screen retained; do not integrate", limitations=protocol["limitations"])
    write("selection.json", selection)
    # Rehash all originals, protocol and code after scoring; do not mutate parents.
    for name, expected in {**inputs, **source_hashes, **own_sources}.items(): assert sha(ROOT/name) == expected, name
    pools = threadpool_info()
    assert all(pool["num_threads"] <= 2 for pool in pools)
    execution = dict(seconds=time.perf_counter()-started, loading_and_guard_seconds=load_seconds, normalization_seconds=key_seconds, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, threadpools=pools, project_files_read=sorted(READS), denied_attempts=DENIED, original_inputs_and_sources_unchanged=True, contract_checks=checks, classifier_fits=0, configuration_count=3, validation_rows_reloaded=len(va), selected_inner_rows_processed=len(content), source_cache_rows=63900, full_cache_deserialization_limitation=protocol["audit"], no_oof=True)
    write("execution.json", execution)
    lines = ["# Normalized static template inner screen", "", f"Gate: {'PASS' if selection['gate_passed'] else 'FAIL'}. Winner: {selection['winner']}.", "", "| Configuration | Fixed errors | Fixed F1 | Fixed FP/FN | Optimum errors | Optimum F1 | Corrected/damaged |", "|---|---:|---:|---|---:|---:|---|", f"| C16 text + deep numeric (0.5/0.5) | {baseline['errors']} | {baseline['f1']:.12f} | {baseline['fp']}/{baseline['fn']} | {baseline['errors']} | {baseline['f1']:.12f} | - |"]
    for row in results:
        lines.append(f"| {row['name']} | {row['fixed']['errors']} | {row['fixed']['f1']:.12f} | {row['fixed']['fp']}/{row['fixed']['fn']} | {row['optimum']['errors']} | {row['optimum']['f1']:.12f} | {row['corrected']}/{row['damaged']} |")
    lines += ["", f"Frozen baseline threshold: {threshold!r}. The candidate optimum is a secondary diagnostic and does not select keys, configuration or gate.", "", "All signatures require 3 distinct training URLs and 2 registered domains. Contradictory training keys and conflicting prediction evidence abstain. Exact 0/1 baseline predictions and explicit hard masks are protected. No preexisting hard-memory rules were fitted or reselected.", "", f"Fit rows: {len(tr)}; validation rows: {len(va)}; disjointness, source hashes, C16 training/data guard, cache URL alignment and prior v8 baseline metrics verified. No new classifiers. All {len(va)} real validation predictions reproduced bitwise after JSON reload and key regeneration.", "", f"Elapsed seconds: {execution['seconds']:.2f}; normalization seconds: {key_seconds:.2f}; peak RSS bytes: {execution['peak_process_rss_bytes']}. CPU library threads <=2.", "", "Artifacts: protocol.json and frozen_run.json freeze policy/code; templates.py is the reusable count model; provenance.json records input/source hashes and row guards; training_key_ledger.json.gz retains accepted/rejected keys with training provenance; key_coverage.json and *_results.json give coverage/conflict counts; selection.json contains the gate; best_model.json is serialized even if the gate fails; reload_check.json verifies inference; validation_match_provenance.json records evidence matches; execution.json lists audited reads and timing.", "", "Distinct from v7 exact component hashing and v8 character-resource classifier: request-specific variation is removed deterministically and independent static views form compound hashes; no learned text representation or new classifier is fitted.", "", protocol["limitations"], "", "Input limitations: full organizer caches necessarily deserialize excluded rows; aligned_splits checks original-row mapping/labels as metadata. Only the 40,781 inner rows undergo template extraction; only 34,663 train labels sign evidence. No excluded-row fitting/scoring. Saved numeric predictions have aligned indices/labels but no equivalent C16-style training guard. No baseline model was loaded or regenerated. No OOF, holdout, test, network or public leaderboard evaluation occurred.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    emit("complete", gate_passed=selection["gate_passed"], winner=selection["winner"], baseline_errors=baseline["errors"], winner_errors=best_result["fixed"]["errors"], seconds=execution["seconds"], reload=reload_check)


if __name__ == "__main__":
    with threadpool_limits(limits=2): main()
