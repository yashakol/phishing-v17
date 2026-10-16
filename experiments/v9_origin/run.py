"""One frozen three-configuration origin-affinity inner screen, offline."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gc, gzip, hashlib, json, resource, time, traceback

INPUTS = {
    "data/content_train.joblib", "data/extra_html_train.joblib",
    "reports/clean_train.csv", "reports/split_indices.npz",
    "experiments/v7_astra/run.py", "experiments/v7_astra/inner_split.npz",
    "experiments/v7_astra/inner_full_vocabulary.npz", "experiments/v7_astra/guards/inner_full_vocabulary.json",
    "experiments/v5/cat_inner_1.npz", "experiments/v5/cat_inner_1.json", "experiments/v5/screen_catboost.py",
    "experiments/v8_astra_regularization/inner_results.json"
}
READS, DENIED = set(), []


def audit(event, args):
    if event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn")) or event in ("os.system", "os.fork", "pty.spawn"):
        DENIED.append(dict(event=event, stack=traceback.format_stack(limit=8)))
        raise PermissionError("Offline origin screen rejects socket/process execution")
    if event in ("os.remove", "os.rmdir", "os.mkdir", "os.rename"):
        for raw in args[:2] if event == "os.rename" else args[:1]:
            if isinstance(raw, (str, bytes, os.PathLike)) and not Path(os.fsdecode(raw)).resolve().is_relative_to(OUT): raise PermissionError("Mutation outside origin sidecar")
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)): return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing and not path.is_relative_to(OUT): raise PermissionError("Write outside origin sidecar: " + str(path))
    if path.is_relative_to(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(".venv/"): return
        own = path.is_relative_to(OUT)
        core = rel.startswith("phishing_url/") and path.suffix in (".py", ".pyc") and "reference" not in rel
        if not own and not core and rel not in INPUTS: raise PermissionError("Read outside origin allowlist: " + rel)
        if not writing: READS.add(rel)


sys.addaudithook(audit)
import joblib
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.content_validation import aligned_splits
from phishing_url.content_checkpoints import guard
from phishing_url.evaluation import best_threshold, measure
from phishing_url.features import group_for_url
from origin import OriginAffinity, prepare_row, training_digest, index_digest, normalize_title, valid_title, FAMILIES


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_gzip(name, value):
    with gzip.open(OUT / name, "wt", encoding="utf-8") as f: json.dump(value, f, sort_keys=True)


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)


def metric(y, p, threshold):
    m = measure(y, p, threshold)
    m["errors"] = m["fp"] + m["fn"]
    return m


def predictions(name, indices, labels, tr=None):
    with np.load(ROOT/name, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["indices"], indices)
        np.testing.assert_array_equal(a["labels"], labels)
        if tr is not None: np.testing.assert_array_equal(a["train_indices"], tr)
        p = a["probabilities"].copy()
    assert p.shape == labels.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def contract_checks():
    assert normalize_title("  A&#38;B  Secure  Access  ") == "a&b secure access"
    assert not valid_title(normalize_title("Welcome to the home page"))
    assert normalize_title("Example access 123") != normalize_title("Example access 456")
    # Wrong row scope must be rejected before even trying to coerce labels.
    class NoLabels:
        def __array__(self, *args, **kwargs): raise AssertionError("Forbidden labels were accessed")
    try: OriginAffinity().fit([dict(index=99)], NoLabels(), expected_indices=[1], forbidden_indices=[99])
    except ValueError as exc: assert "row guard" in str(exc)
    else: raise AssertionError("Validation row was accepted")
    # Synthetic records only: verify actual conditional transfer and origin trust.
    domains = ["source.example"]*3 + ["copy-a.example"]*2 + ["copy-b.example"]
    toy = [prepare_row(i, "https://"+d+"/"+str(i), "Synthetic document service", d, "source.example", "source.example", "cdn.example") for i,d in enumerate(domains)]
    toy_labels = [0,0,0,1,1,1]
    model = OriginAffinity().fit(toy, toy_labels, list(range(6)), forbidden_indices=[100])
    assert {g["family"] for g in model.evidence.values()} == {"title", "canonical_title", "og_title"}
    assert any(g["status"] == "untrusted_declared_origin" for g in model.ledger.values())
    sample_domains = ["source.example", "new-copy.example", "copy-a.example", "new-copy.example", "new-copy.example"]
    sample = [prepare_row(100+i, "https://"+d+"/new", "Synthetic document service", d, "source.example", "source.example", "cdn.example") for i,d in enumerate(sample_domains)]
    adjusted, flags, _ = model.predict(sample, [.9,.1,.1,0.,1.])
    assert adjusted[0] < .5 and adjusted[1] > .5 and adjusted[2] == .1
    np.testing.assert_array_equal(adjusted[3:], [0.,1.])
    assert flags["new_foreign"][1] and flags["known_foreign_only"][2]
    mixed = toy + [prepare_row(6, "https://source.example/conflict", "Synthetic document service", "source.example", "source.example", "source.example", "cdn.example")]
    assert not OriginAffinity().fit(mixed, toy_labels+[1], list(range(7))).evidence
    return dict(full_title_retained=True, generic_title_excluded=True, validation_row_rejected_before_label_access=True, conditional_source_and_new_foreign_transfer=True, known_foreign_abstention=True, untrusted_cdn_origin_excluded=True, mixed_source_excluded=True, exact_hard_answers_preserved=True)


def main():
    started = time.perf_counter()
    if (OUT/"frozen_run.json").exists(): raise RuntimeError("Screen already frozen; no adaptive rerun")
    protocol = json.loads((OUT/"protocol.json").read_text())
    sources = {f"experiments/v9_origin/{name}": sha(OUT/name) for name in ("protocol.json", "origin.py", "run.py")}
    write("frozen_run.json", dict(protocol=protocol, sources=sources, unix_time=time.time(), order="before dataset loads, fitting or scoring"))
    checks = contract_checks()
    emit("frozen", configurations=3, classifier_fits=0, checks=checks)
    # Saved predictions are not opened or hashed until all evidence fits finish.
    inputs = {name: sha(ROOT/name) for name in sorted(INPUTS) if name not in (protocol["baseline"]["text"], protocol["baseline"]["numeric"], "experiments/v8_astra_regularization/inner_results.json")}
    cache = joblib.load(ROOT/"data/content_train.joblib")
    assert len(cache["records"]) == len(cache["labels"]) == len(cache["urls"]) == 63900
    splits = aligned_splits(ROOT, cache)
    tr, va = splits["inner_train"], splits["inner_validation"]
    assert len(tr) == 34663 and len(va) == 6118 and not set(tr)&set(va)
    assert set(tr)|set(va) <= set(splits["development"])
    assert not (set(tr)|set(va)) & (set(splits["holdout"])|set(splits["added"]))
    with np.load(ROOT/"experiments/v7_astra/inner_split.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["inner_train"], tr); np.testing.assert_array_equal(a["inner_validation"], va)
    original_guard = json.loads((ROOT/protocol["baseline"]["guard"]).read_text())
    expected = guard(cache, tr, original_guard["config"], inputs["data/content_train.joblib"])
    expected["ordered_training_indices_sha256"] = index_digest(tr)
    assert original_guard == expected and original_guard["config"]["parameters"]["c"] == 16.
    for name, expected_sha in original_guard["config"]["sources"].items():
        assert sha(ROOT/name) == expected_sha
        sources[name] = expected_sha
    for name in ("phishing_url/features.py", "phishing_url/extra_html.py", "phishing_url/soft_context.py"):
        sources[name] = sha(ROOT/name)
    selected = np.concatenate([tr, va])
    ytrain = cache["labels"][tr].copy()
    validation_truth = cache["labels"][va].copy()  # quarantined from every fit API
    all_urls = list(cache["urls"])
    minimal = []
    for i in selected:
        record = cache["records"][int(i)]
        assert record["url"] == all_urls[int(i)]
        minimal.append(dict(index=int(i), url=record["url"], title=record.get("title", "")))
    selected_html_hash = hashlib.sha256("\n".join(cache["hashes"][int(i)] for i in selected).encode()).hexdigest()
    metadata = dict(content=cache["source"])
    del cache, record, splits
    gc.collect()
    extra = joblib.load(ROOT/"data/extra_html_train.joblib")
    assert len(extra["records"]) == 63900 and list(extra["urls"]) == all_urls
    np.testing.assert_array_equal(extra["labels"][tr], ytrain)
    np.testing.assert_array_equal(extra["labels"][va], validation_truth)
    cols = ["cat_extra_canonical_domain", "cat_extra_og_domain", "cat_extra_main_resource_domain"]
    feature_rows = extra["features"].iloc[selected][cols].values.tolist()
    metadata["extra"] = extra["source"]
    assert metadata["extra"]["original_raw_hashes_verified"] is True
    del extra, all_urls
    gc.collect()
    rows = [prepare_row(r["index"], r["url"], r["title"], group_for_url(r["url"]), *f) for r, f in zip(minimal, feature_rows, strict=True)]
    train_rows, validation_rows = rows[:len(tr)], rows[len(tr):]
    assert {r["url"] for r in train_rows}.isdisjoint(r["url"] for r in validation_rows)
    scope = dict(training_rows=len(tr), training_indices_sha256=index_digest(tr), ordered_training_sha256=training_digest(train_rows, ytrain))
    assert scope["ordered_training_sha256"] == original_guard["ordered_training_sha256"]
    write_gzip("training_rows.json.gz", [dict(row=r, label=int(y)) for r, y in zip(train_rows, ytrain, strict=True)])
    write_gzip("validation_keys.json.gz", validation_rows)
    np.savez_compressed(OUT/"inner_split.npz", inner_train=tr, inner_validation=va)
    write("key_coverage.json", {f: dict(train_rows=sum(f in r["keys"] for r in train_rows), validation_rows=sum(f in r["keys"] for r in validation_rows), unique_training_keys=len({r["keys"][f] for r in train_rows if f in r["keys"]})) for f in FAMILIES})
    load_seconds = time.perf_counter()-started
    fitted, timings, model_hashes = {}, {}, {}
    train_index_set = set(tr)
    for spec in protocol["configurations"]:
        config = dict(spec); name = config.pop("name")
        fit_start = time.perf_counter()
        model = OriginAffinity(**protocol["shared_policy"], **config).fit(train_rows, ytrain, tr, forbidden_indices=va)
        assert model.scope == scope
        assert all(set(g["training_indices"]) <= train_index_set for g in model.ledger.values())
        timings[name] = time.perf_counter()-fit_start
        fitted[name] = model.to_dict()
        write(name+"_model.json", fitted[name])
        model_hashes[name] = sha(OUT/(name+"_model.json"))
        if name == "all_origins_098": write_gzip("training_group_ledger.json.gz", model.ledger)
        emit("fit_complete", name=name, accepted_keys=len(model.evidence), fit_seconds=timings[name], statistics=model.statistics)
        del model
    # This record is persisted before reading baseline probabilities or scoring.
    write("fits_before_scoring.json", dict(unix_time=time.time(), candidate_model_hashes=model_hashes, scope=scope, all_three_fits_complete=True, validation_labels_passed_to_fit=False))
    for name in (protocol["baseline"]["text"], protocol["baseline"]["numeric"], "experiments/v8_astra_regularization/inner_results.json"):
        inputs[name] = sha(ROOT/name)
    y = validation_truth
    ptext = predictions(protocol["baseline"]["text"], va, y, tr)
    pcat = predictions(protocol["baseline"]["numeric"], va, y)
    base = .5*ptext + .5*pcat
    threshold, _ = best_threshold(y, base)
    baseline = metric(y, base, threshold)
    assert baseline == json.loads((ROOT/"experiments/v8_astra_regularization/inner_results.json").read_text())["baseline"]["mix"]
    assert baseline["errors"] == protocol["baseline"]["expected_errors"]
    emit("baseline", **baseline)
    provenance = dict(inputs=inputs, sources=sources, original_c16_guard=original_guard, scope=scope, validation_indices_sha256=index_digest(va), source_cache_rows=63900, transformed_rows=len(rows), selected_html_sha256=selected_html_hash, disjoint_original_inner_only=True, extractor_metadata=metadata, validation_truth_sha256=hashlib.sha256(y.tobytes()).hexdigest(), baseline_origin="Aligned saved guarded C16 and deep numeric predictions; no classifier load or fit", numeric_guard_limitation="Saved numeric archive has indices/labels but no training scope guard; originating source uses aligned_splits with validation early stopping", training_rows_sha256=sha(OUT/"training_rows.json.gz"), validation_keys_sha256=sha(OUT/"validation_keys.json.gz"))
    write("provenance.json", provenance)
    results, outputs = [], {}
    for order, spec in enumerate(protocol["configurations"]):
        name = spec["name"]
        model = OriginAffinity.from_dict(fitted[name], expected_scope=scope)
        predict_start = time.perf_counter()
        p, flags, traces = model.predict(validation_rows, base)
        predict_seconds = time.perf_counter()-predict_start
        optimum, _ = best_threshold(y, p)
        fixed, tuned = metric(y, p, threshold), metric(y, p, optimum)
        changed = (p>=threshold)!=(base>=threshold)
        corrected, damaged = changed&((p>=threshold)==y), changed&((base>=threshold)==y)
        breakdown = {side: dict(eligible=int(flags[side].sum()), adjusted=int((flags[side]&flags["adjusted"]).sum()), corrected=int((flags[side]&corrected).sum()), damaged=int((flags[side]&damaged).sum())) for side in ("source", "new_foreign", "globally_unseen_foreign")}
        result = dict(name=name, order=order, configuration=model.config, fixed=fixed, optimum=tuned, corrected=int(corrected.sum()), damaged=int(damaged.sum()), changed_labels=int(changed.sum()), net_fewer_errors=baseline["errors"]-fixed["errors"], flags={k:int(v.sum()) for k,v in flags.items()}, conditional_breakdown=breakdown, accepted_keys=len(model.evidence), training_statistics=model.statistics, family_validation_matching_rows={f:sum(r["keys"].get(f) in model.evidence for r in validation_rows) for f in model.config["families"]}, gate_passed=bool(baseline["errors"]-fixed["errors"]>=2 and fixed["f1"]>=baseline["f1"]), secondary_optimum_gate=bool(baseline["errors"]-tuned["errors"]>=2 and tuned["f1"]>=baseline["f1"]), fit_seconds=timings[name], predict_seconds=predict_seconds)
        assert result["net_fewer_errors"] == result["corrected"]-result["damaged"]
        results.append(result); outputs[name] = (p, flags, traces)
        write(name+"_results.json", result)
        np.savez_compressed(OUT/(name+"_inner.npz"), indices=va, labels=y, probabilities=p, baseline=base, **flags)
        write_gzip(name+"_validation_evidence.json.gz", traces)
        emit("candidate", name=name, fixed=fixed, optimum=tuned, flags=result["flags"], conditional_breakdown=breakdown, gate_passed=result["gate_passed"])
    winner = min(results, key=lambda r:(r["fixed"]["errors"], -r["fixed"]["f1"], r["order"]))
    name = winner["name"]
    write("best_model.json", dict(model=fitted[name], guard=dict(scope=scope, provenance_sha256=sha(OUT/"provenance.json"), sources=sources, candidate_model_sha256=model_hashes[name])))
    saved = json.loads((OUT/"best_model.json").read_text())
    restored = OriginAffinity.from_dict(saved["model"], expected_scope=scope)
    # Rebuild all real validation rows from original minimal cache fields.
    rebuilt = [prepare_row(r["index"],r["url"],r["title"],group_for_url(r["url"]),*f) for r,f in zip(minimal[len(tr):],feature_rows[len(tr):],strict=True)]
    assert rebuilt == validation_rows
    after, after_flags, after_traces = restored.predict(rebuilt, base)
    before, flags, traces = outputs[name]
    np.testing.assert_array_equal(before, after)
    for flag in flags: np.testing.assert_array_equal(flags[flag], after_flags[flag])
    assert traces == after_traces
    hard = np.zeros(len(va), bool); hard[::17] = True
    masked, _, _ = restored.predict(rebuilt, base, hard)
    np.testing.assert_array_equal(masked[hard], base[hard])
    wrong_scope_rejected = False
    try: OriginAffinity.from_dict(saved["model"], expected_scope=dict(scope, training_rows=-1))
    except ValueError: wrong_scope_rejected = True
    assert wrong_scope_rejected
    write("reload_check.json", dict(rows=len(va), probability_bitwise_equal=True, keys_regenerated_from_real_minimal_cache_fields=True, flags_equal=True, traces_equal=True, max_abs_difference=float(np.abs(before-after).max()), explicit_hard_mask_preserved=True, wrong_scope_rejected=True, model_sha256=sha(OUT/"best_model.json")))
    selection = dict(winner=name, gate_passed=winner["gate_passed"], baseline=baseline, winner_result=winner, fixed_threshold=threshold, classifier_fits=0, configurations=3, no_oof=True, no_public_improvement_claim=True, status="preserved for separate parent integration only" if winner["gate_passed"] else "negative result retained; no integration", parent_routing_used=False, limitations=protocol["limitations"])
    write("selection.json", selection); write("results.json", dict(baseline=baseline,candidates=results))
    for path, expected_sha in {**inputs, **sources}.items(): assert sha(ROOT/path) == expected_sha, path
    for name, expected_sha in model_hashes.items(): assert sha(OUT/(name+"_model.json")) == expected_sha
    pools = threadpool_info()
    assert all(p["num_threads"]<=2 for p in pools)
    execution = dict(seconds=time.perf_counter()-started, load_and_prepare_seconds=load_seconds, fit_seconds=timings, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, threadpools=pools, project_read_open_events=sorted(READS), denied_events=DENIED, original_sources_inputs_unchanged=True, fitted_models_unchanged_after_scoring=True, classifier_fits=0, configurations=3, guard_checks=checks, selected_rows_transformed=len(rows), validation_rows_reproduced=len(va), no_oof=True)
    write("execution.json", execution)
    lines = ["# Conditional content-origin inner screen", "", f"Gate: {'PASS' if winner['gate_passed'] else 'FAIL'}. Winner: {winner['name']}; frozen baseline threshold {threshold!r}.", "", "| Configuration | Fixed errors | Fixed F1 | FP/FN | Tuned errors | Tuned F1 | Corrected/damaged |", "|---|---:|---:|---|---:|---:|---|", f"| C16/Cat 0.5/0.5 | {baseline['errors']} | {baseline['f1']:.12f} | {baseline['fp']}/{baseline['fn']} | {baseline['errors']} | {baseline['f1']:.12f} | - |"]
    for r in results: lines.append(f"| {r['name']} | {r['fixed']['errors']} | {r['fixed']['f1']:.12f} | {r['fixed']['fp']}/{r['fixed']['fn']} | {r['optimum']['errors']} | {r['optimum']['f1']:.12f} | {r['corrected']}/{r['damaged']} |")
    lines += ["", "The hypothesis is evaluated conditionally on URL-domain membership in training-proven legitimate sources, not unconditional title purity. Each accepted key needs both legitimate and phishing examples; declared/resource origins must be observed benign source domains under that same key. Foreign inference is limited to a URL domain unseen within the key. All thresholds, smoothing constants, count weights, families, source limits and support requirements were frozen before fitting. The candidate optimum is diagnostic only. No routing results were used.", ""]
    for r in results: lines.append(f"{r['name']}: {r['accepted_keys']} accepted keys; flags={json.dumps(r['flags'],sort_keys=True)}; source/foreign breakdown={json.dumps(r['conditional_breakdown'],sort_keys=True)}.")
    lines += ["", "All three evidence models were fitted and serialized before opening classifier predictions or scoring. Full train rows and validation keys are saved with original aligned indices; fit API rejects forbidden indices before accessing labels. Original C16 data, training and source guards match; numeric archive row/label alignment matches, with its inherited guard/early-stopping limitations documented. Best-model JSON reload regenerated all 6118 real validation keys and reproduced every probability, flag and evidence trace bitwise; explicit hard masks and wrong-scope rejection passed.", "", f"Execution: {execution['seconds']:.2f} seconds, peak RSS {execution['peak_process_rss_bytes']} bytes, library threads <=2, zero classifier fits. Full cache deserialization includes excluded rows, but only selected inner URL/title/three cached origin fields were transformed. No raw HTML/script/style normalization, excluded-row scoring, full OOF or test evaluation occurred.", "", "Audit: every socket/process event is rejected. Import-time urllib3 IPv6 socket initialization, if attempted, is blocked before socket creation and retained with its stack in execution.json. No allowed network operations exist.", "", protocol["limitations"], "", "Artifacts: protocol.json/frozen_run.json; origin.py/run.py; training_rows.json.gz and validation_keys.json.gz; training_group_ledger.json.gz; *_model.json and best_model.json; *_inner.npz and *_validation_evidence.json.gz; provenance.json/selection.json/results.json/reload_check.json/execution.json. Independent verification is saved separately in verification.json.", ""]
    (OUT/"REPORT.md").write_text("\n".join(lines))
    emit("complete", gate_passed=winner["gate_passed"], winner=winner["name"], baseline_errors=baseline["errors"], winner_errors=winner["fixed"]["errors"], seconds=execution["seconds"])


if __name__ == "__main__":
    with threadpool_limits(limits=2): main()
