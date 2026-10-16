"""Preregistered own-fold raw-label neighbor vote; no new models or retrieval."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'experiments/v16_neighbor'
sys.path.insert(0, str(ROOT))
import csv
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
READS, WRITES = set(), set()
INPUTS = {'data/content_train.joblib', 'reports/clean_train.csv', 'reports/split_indices.npz',
          'phishing_url/__init__.py', 'phishing_url/evaluation.py'}


def audit(event, args):
    if event.startswith(('socket.connect', 'socket.getaddrinfo', 'socket.bind', 'subprocess.Popen', 'os.system')):
        raise PermissionError('Offline experiment; child processes disabled')
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing:
        if not p.is_relative_to(OUT):
            raise PermissionError(f'Write outside experiment: {p}')
        WRITES.add(p.relative_to(ROOT).as_posix())
        return
    if not p.is_relative_to(ROOT):
        return
    rel = p.relative_to(ROOT).as_posix()
    if rel.startswith('.venv/'):
        return
    if any(s in rel.lower() for s in ('external/', 'reports/v2/', 'reference', 'submission', 'holdout', 'test', 'publicsql')):
        raise PermissionError(f'Forbidden input: {rel}')
    bytecode = rel.startswith('phishing_url/__pycache__/') and 'phishing_url/' + p.name.split('.')[0] + '.py' in INPUTS
    if rel not in INPUTS and not p.is_relative_to(OUT) and not p.is_relative_to(OLD) and not bytecode:
        raise PermissionError(f'Input outside allowlist: {rel}')
    READS.add(rel)


sys.addaudithook(audit)
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.evaluation import best_threshold, measure


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def emit(event, **value):
    print(json.dumps(dict(event=event, **value)), flush=True)


def prepare():
    assert not (OUT / 'preregistered.json').exists(), 'No re-registration'
    old = json.loads((OLD / 'preregistered.json').read_text())
    cfg = {k: old[k] for k in ('eligible_probability', 'neighbors', 'minimum_similarity', 'minimum_support', 'power', 'smoothing', 'splitter', 'gate', 'frozen_v15_threshold', 'representation')}
    cfg.update(created_utc=datetime.now(timezone.utc).isoformat(), variant='exactfold_knn8_smoothed_raw_train_vote',
               weights=[.25, .5, .75], threads=2, classifier_fits=0, vectorizer_fits=0, retrieval_fits=0,
               formula='q=(sum(sim**4*y_train_neighbor)+0.5*outer_train_prior)/(sum(sim**4)+0.5); p=(1-weight)*frozen_v15+weight*q only with original support>=2 and baseline within [0.05,0.95]; otherwise exact baseline',
               numeric_policy='Cast saved float32 similarities to float64 before exponentiation; no new clipping, residual cap, retrieval, neighbor or eligibility changes',
               prior='Mean raw own-train labels over complete original outer-training side, not over nearest labels or validation rows',
               threshold='Maximum pooled development F1, ties nearest 0.5; baseline and each candidate tuned separately; also report common candidate threshold',
               selection='Among gate-passing candidates choose highest tuned F1, then largest frozen net fixes, then lower weight. If none pass selected_config=null; report best_observed by same order without adoption.',
               limitations=['Reused original development and previously selected baseline, not an independent estimate.',
                            'Inherited v15 specialist OOF is not fully nested; strict new neighbor scope does not repair the baseline.',
                            'Only raw-label vote mixtures of frozen v15 are evaluated; no full refit or public-score claim.'],
               sources_sha256={name: sha(OUT / name) for name in ('run.py', 'verify.py', 'PROTOCOL.md')},
               input_sha256={p: sha(ROOT / p) for p in sorted(INPUTS)},
               v16_snapshot_sha256={p.name: sha(p) for p in sorted(OLD.iterdir()) if p.is_file()})
    for p in ('data/content_train.joblib', 'reports/clean_train.csv', 'reports/split_indices.npz', 'phishing_url/evaluation.py'):
        assert cfg['input_sha256'][p] == old['input_sha256'][p]
    with (OUT / 'preregistered.json').open('x') as f:
        json.dump(cfg, f, indent=2, sort_keys=True); f.write('\n')
    emit('preregistered', sha256=sha(OUT / 'preregistered.json'), weights=cfg['weights'])


def check_frozen(cfg):
    for p, h in cfg['input_sha256'].items():
        assert sha(ROOT / p) == h, p
    for p, h in cfg['sources_sha256'].items():
        assert sha(OUT / p) == h, p
    assert sorted(p.name for p in OLD.iterdir() if p.is_file()) == sorted(cfg['v16_snapshot_sha256'])
    for p, h in cfg['v16_snapshot_sha256'].items():
        assert sha(OLD / p) == h, p


def compared(y, before, after, before_cut, after_cut, dev, folds):
    b, a = before >= before_cut, after >= after_cut
    fixed, new = (b != y) & (a == y), (b == y) & (a != y)
    def metric(yy, p, cut):
        m = measure(yy, p, cut)
        return {**m, 'errors': m['fp'] + m['fn']}
    rows = []
    for f in range(3):
        m = folds == f
        rows.append(dict(fold=f, before_errors=int((b[m] != y[m]).sum()), after_errors=int((a[m] != y[m]).sum()),
                         fixed=int(fixed[m].sum()), new=int(new[m].sum()),
                         before_f1=metric(y[m], before[m], before_cut)['f1'], after_f1=metric(y[m], after[m], after_cut)['f1']))
    return dict(before=metric(y, before, before_cut), after=metric(y, after, after_cut),
                fixed_errors=int(fixed.sum()), new_errors=int(new.sum()), net_fixes=int(fixed.sum())-int(new.sum()),
                fixed_indices=dev[fixed].tolist(), new_error_indices=dev[new].tolist(), folds=rows)


def gate_check(comparisons, cfg):
    g = cfg['gate']; checks = {}
    for mode in g['modes']:
        c = comparisons[mode]
        checks[mode] = dict(net_fixes=c['net_fixes'] >= g['net_fewer_errors_vs_v15_at_each_mode'],
                            higher_f1=c['after']['f1'] > c['before']['f1'],
                            improved_folds=sum(f['after_errors'] < f['before_errors'] for f in c['folds']) >= g['improved_folds_at_each_mode'],
                            no_fold_worse=all(f['after_errors'] <= f['before_errors'] + g['maximum_fold_error_increase_at_each_mode'] for f in c['folds']))
    return all(all(v.values()) for v in checks.values()), checks


def main():
    started = time.perf_counter()
    assert not (OUT / 'started.json').exists(), 'Single run only; no silent rescore'
    cfg = json.loads((OUT / 'preregistered.json').read_text())
    check_frozen(cfg)
    write('started.json', dict(utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(OUT / 'preregistered.json')))
    old_verify = json.loads((OLD / 'verification.json').read_text())
    for name in ('development_oof.npz', 'fold_scopes_before_retrieval.json', *(f'fold_{f}_retrieval.npz' for f in range(3))):
        assert sha(OLD / name) == old_verify['artifact_sha256'][name]
    with np.load(OLD / 'development_oof.npz', allow_pickle=False) as a:
        dev, y, folds, base = a['indices'], a['labels'], a['folds'], a['baseline_v15']
        queried, eligible = a['queried'], a['eligible_v15']
    np.testing.assert_array_equal(eligible, (base >= cfg['eligible_probability'][0]) & (base <= cfg['eligible_probability'][1]))
    cache = joblib.load(ROOT / 'data/content_train.joblib')
    raw_y = cache['labels'][dev].copy()
    urls = [cache['urls'][int(i)] for i in dev]
    np.testing.assert_array_equal(raw_y, y)
    assert len(set(urls)) == len(set(dev.tolist())) == len(dev)
    with np.load(ROOT / 'reports/split_indices.npz', allow_pickle=False) as a:
        original_dev = a['development']
    positions = {int(i): j for j, i in enumerate(original_dev)}
    with (ROOT / 'reports/clean_train.csv').open(newline='', encoding='utf-8') as f:
        count = 0
        for i, row in enumerate(csv.DictReader(f)):
            if i in positions:
                j = positions[i]
                assert row['url'] == urls[j] and int(row['result']) == int(raw_y[j])
                count += 1
    assert count == len(dev) == 50977
    del cache; gc.collect()
    np.savez_compressed(OUT / 'own_development_labels.npz', indices=dev, labels=raw_y)
    scopes = json.loads((OLD / 'fold_scopes_before_retrieval.json').read_text())['origins']
    vote = np.zeros(len(y)); support = np.zeros(len(y), dtype=np.int16); priors = np.zeros(len(y))
    coverage = np.zeros(len(y), dtype=np.int8); scope_reports = []
    with threadpool_limits(limits=2):
        for f, (tr, va) in enumerate(StratifiedKFold(**cfg['splitter']).split(dev, raw_y)):
            assert (folds[va] == f).all() and not np.isin(dev[tr], dev[va]).any()
            a = np.load(OLD / f'fold_{f}_retrieval.npz', allow_pickle=False)
            np.testing.assert_array_equal(a['train_indices'], dev[tr]); np.testing.assert_array_equal(a['validation_indices'], dev[va])
            qp = va[queried[va]]
            np.testing.assert_array_equal(a['query_indices'], dev[qp])
            ids, sim, labels = a['neighbor_indices'], a['similarities'], a['neighbor_labels']
            original_support = a['support']; a.close()
            assert ids.shape == sim.shape == labels.shape == (len(qp), cfg['neighbors'])
            good = ids >= 0
            np.testing.assert_array_equal(good, labels >= 0)
            assert np.isin(ids[good], dev[tr]).all() and not np.isin(ids[good], dev[va]).any()
            assert not (ids == dev[qp, None]).any()
            index_to_local = {int(i): j for j, i in enumerate(dev[tr])}
            expected = np.array([raw_y[tr[index_to_local[int(i)]]] for i in ids[good]], dtype=np.int8)
            np.testing.assert_array_equal(labels[good], expected)
            assert (sim[~good] == 0).all() and (sim[good] >= cfg['minimum_similarity']).all()
            assert (sim[good] <= 1.000002).all()
            valid = (labels >= 0) & (sim >= cfg['minimum_similarity'])
            np.testing.assert_array_equal(valid.sum(axis=1), original_support)
            for row in ids:
                assert len(set(row[row >= 0].tolist())) == int((row >= 0).sum())
            g = scopes[f]['guard']
            assert g['ordered_training_indices_sha256'] == hashlib.sha256(dev[tr].tobytes()).hexdigest()
            sample = hashlib.sha256()
            for j in tr:
                sample.update(urls[int(j)].encode()); sample.update(b'\0'); sample.update(bytes([int(raw_y[j])]))
            assert sample.hexdigest() == g['ordered_training_sha256']
            assert g['data_sha256'] == cfg['input_sha256']['data/content_train.joblib'] and g['training_rows'] == len(tr)
            prior = float(raw_y[tr].mean())
            weights = np.where(valid, sim.astype(np.float64) ** cfg['power'], 0.)
            q = (np.sum(weights * np.maximum(labels, 0), axis=1) + cfg['smoothing'] * prior) / (weights.sum(axis=1) + cfg['smoothing'])
            assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
            vote[va] = prior; vote[qp] = q; priors[va] = prior; support[qp] = original_support; coverage[va] += 1
            row = dict(fold=f, train_rows=len(tr), validation_rows=len(va), query_rows=len(qp), prior=prior,
                       neighbor_edges=int(good.sum()), support_ge_2=int((original_support >= cfg['minimum_support']).sum()),
                       eligible_supported=int((eligible[qp] & (original_support >= cfg['minimum_support'])).sum()),
                       raw_labels_match_cache_and_original_train=True, neighbors_strictly_outer_train=True,
                       original_c16_guard_matches=True, no_self_neighbors=True)
            scope_reports.append(row); emit('vote_fold_complete', **row)
    assert (coverage == 1).all()
    applied = eligible & (support >= cfg['minimum_support'])
    predictions = np.array([np.where(applied, (1-w)*base + w*vote, base) for w in cfg['weights']])
    assert np.isfinite(predictions).all()
    for p in predictions:
        np.testing.assert_array_equal(p[~applied], base[~applied])
    # Freeze and serialize every candidate before the first metric calculation.
    np.savez_compressed(OUT / 'development_vote.npz', indices=dev, labels=raw_y, folds=folds, baseline=base,
                        votes=vote, support=support, eligible=eligible, applied=applied, fold_train_prior=priors,
                        weights=np.array(cfg['weights']), probabilities=predictions)
    with np.load(OUT / 'development_vote.npz', allow_pickle=False) as a:
        for i, w in enumerate(a['weights']):
            replay = np.where(a['applied'], (1-w)*a['baseline'] + w*a['votes'], a['baseline'])
            np.testing.assert_array_equal(replay, a['probabilities'][i]); np.testing.assert_array_equal(predictions[i], replay)
    emit('all_predictions_serialized_before_scoring', candidate_count=len(predictions), applied_rows=int(applied.sum()))
    baseline_cut, _ = best_threshold(raw_y, base)
    rows = []
    for weight, p in zip(cfg['weights'], predictions):
        cut, _ = best_threshold(raw_y, p)
        comparisons = {
            'frozen_v15': compared(raw_y, base, p, cfg['frozen_v15_threshold'], cfg['frozen_v15_threshold'], dev, folds),
            'individually_tuned': compared(raw_y, base, p, baseline_cut, cut, dev, folds),
            'same_candidate_tuned_threshold': compared(raw_y, base, p, cut, cut, dev, folds),
        }
        passed, checks = gate_check(comparisons, cfg)
        rows.append(dict(weight=weight, threshold=cut, gate_passed=passed, gate_checks=checks, comparisons=comparisons))
    rank = lambda r: (r['comparisons']['individually_tuned']['after']['f1'], r['comparisons']['frozen_v15']['net_fixes'], -r['weight'])
    passed = [r for r in rows if r['gate_passed']]
    best = max(passed if passed else rows, key=rank)
    with np.load(OLD / 'development_oof.npz', allow_pickle=False) as a:
        residual_p = a['neighbor_v15']
    differences = [dict(weight=w, probability_rows_different_from_v16=int((p != residual_p).sum()),
                        max_abs_probability_difference=float(np.max(abs(p - residual_p))),
                        decisions_different_from_v16_at_frozen=int(((p >= cfg['frozen_v15_threshold']) != (residual_p >= cfg['frozen_v15_threshold'])).sum())) for w, p in zip(cfg['weights'], predictions)]
    check_frozen(cfg)
    write('results.json', dict(variant=cfg['variant'], decision='accept_development_candidate' if passed else 'reject_raw_vote_retain_v15',
                              gate_passed=bool(passed), selected_config=dict(weight=best['weight'], threshold=best['threshold']) if passed else None,
                              best_observed_config=dict(weight=best['weight'], threshold=best['threshold']), candidates=rows,
                              folds=scope_reports, vote_vs_v16_residual=differences, applied_rows=int(applied.sum()),
                              rows=len(dev), protocol_sha256=sha(OUT / 'preregistered.json'), source_sha256=sha(__file__),
                              limitations=cfg['limitations'], seconds=time.perf_counter()-started))
    write('execution.json', dict(project_files_read=sorted(READS), project_files_written=sorted(WRITES | {'experiments/v18_neighbor_vote/execution.json'}),
                                v16_unchanged=True, input_hashes_unchanged=True, classifier_fits=0, vectorizer_fits=0, retrieval_fits=0,
                                original_raw_label_alignment=True, serialization_mix_replay_bitwise_identical=True,
                                all_candidates_serialized_before_scoring=True, threadpools=threadpool_info(), network_used=False))
    emit('complete', gate_passed=bool(passed), best_weight=best['weight'], metrics=[dict(weight=r['weight'], passed=r['gate_passed'], frozen_net=r['comparisons']['frozen_v15']['net_fixes'], tuned_net=r['comparisons']['individually_tuned']['net_fixes']) for r in rows])


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        prepare()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Use --prepare once, then run without arguments')
