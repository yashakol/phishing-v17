"""One preregistered inner NB logistic fit, using frozen guarded v7 TF-IDF."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hashlib
import json
import time

READS = set()
WRITES = set()
ALLOWED = {
    'data/content_train.joblib', 'reports/clean_train.csv', 'reports/split_indices.npz',
    'experiments/v8_astra_regularization/resume.py',
    'experiments/v7_astra/run.py', 'experiments/v7_astra/inner_results.json',
    'experiments/v7_astra/inner_full_vocabulary.npz',
    'experiments/v7_astra/guards/inner_full_vocabulary.json',
    'experiments/v7_astra/models/inner_full_vocabulary.joblib',
    'experiments/v5/cat_inner_1.npz',
}


def audit(event, args):
    if event.startswith(('socket.connect', 'socket.getaddrinfo', 'socket.bind')):
        raise PermissionError('Offline experiment: ' + event)
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
               (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
    if writing:
        if not p.is_relative_to(OUT):
            raise PermissionError(f'Write outside scope: {p}')
        WRITES.add(str(p.relative_to(ROOT)))
        return
    if not p.is_relative_to(ROOT):
        return  # Interpreter / installed libraries only; no external data paths used.
    rel = p.relative_to(ROOT).as_posix()
    if rel.startswith('.venv/'):
        return
    if (p.is_relative_to(OUT) or rel in ALLOWED or
        (p.parent == ROOT / 'phishing_url' and p.suffix == '.py' and p.name in
         {'__init__.py', 'content_validation.py', 'content_checkpoints.py',
          'evaluation.py', 'html_text.py', 'v7_text.py'})):
        READS.add(rel)
        return
    # Python may probe a stale bytecode cache even with dont_write_bytecode.
    if '/__pycache__/' in rel and p.suffix == '.pyc':
        raise FileNotFoundError(f'Bytecode disabled: {rel}')
    raise PermissionError(f'Input outside exact allowlist: {rel}')


sys.addaudithook(audit)
import copy
import gc
import resource
import warnings
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.content_checkpoints import data_digest, load_component, save_component
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.html_text import ContentTextModel
# Import definitions only. The v8 OUT-specific audit/main are NOT installed/called.
from experiments.v8_astra_regularization.resume import checked_base, scope, semantic_hash


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def emit(event, **kw):
    print(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-START, **kw)), flush=True)


def metrics(y, p, threshold):
    m = measure(y, p, threshold)
    m['errors'] = m['fp'] + m['fn']
    return m


def provenance(cache, indices):
    record_hash = hashlib.sha256()
    for i in indices:
        record_hash.update(json.dumps(cache['records'][int(i)], sort_keys=True,
                                      ensure_ascii=True, separators=(',', ':')).encode())
        record_hash.update(b'\n')
    return dict(rows=len(indices), ordered_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                ordered_labels_sha256=hashlib.sha256(cache['labels'][indices].tobytes()).hexdigest(),
                ordered_records_sha256=record_hash.hexdigest(),
                ordered_urls_labels_sha256=scope(cache, indices, {}, DIGEST)['ordered_training_sha256'])


def prediction(path, va, y, tr=None):
    with np.load(ROOT / path, allow_pickle=False) as a:
        np.testing.assert_array_equal(a['indices'], va)
        np.testing.assert_array_equal(a['labels'], y)
        if tr is not None:
            np.testing.assert_array_equal(a['train_indices'], tr)
        p = a['probabilities'].copy()
    assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def main():
    global DIGEST, FITS
    # Exclusive markers prohibit a second fit or a silent parameter retry.
    with (OUT / 'attempt.json').open('x') as f:
        json.dump(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       script_sha256=sha(__file__)), f, indent=2)
    classifier = LogisticRegression(C=4, solver='liblinear', max_iter=600, random_state=20261008)
    protocol = dict(
        classifier=classifier.get_params(), threads=2, classifier_fit_budget=1,
        alpha=1.0, nb_formula='pos=sum(X[y_train==1],axis=0)+1.0; neg=sum(X[y_train==0],axis=0)+1.0; ratio=log((pos/pos.sum())/(neg/neg.sum())).astype(float32)',
        ratio_input='Frozen v7 nonnegative concatenated float32 TF-IDF; no binarization or extra normalization',
        ratio_labels='inner_train only', inner_train_rows=34663, inner_validation_rows=6118,
        splits='original aligned_splits; remaining cache rows only for split/index consistency metadata',
        representation='guarded inner_full_vocabulary v7 C16 vectorizers/IDF; transform only; zero vectorizer fits',
        baseline_formula='0.5 * v7_text + 0.5 * cat_inner_1',
        frozen_baseline_threshold=0.4939816339654343, expected_baseline_errors=51,
        replacement_weights=[0.25, 0.5, 1.0],
        mixture_formula='0.5 * ((1-w) * v7_text + w * nb_text) + 0.5 * cat_inner_1',
        tuning='Only existing best_threshold max F1, ties nearest 0.5, once per prespecified mixture plus standalone diagnostic',
        gate='The SAME weight must reduce baseline errors by >=2 at original frozen threshold AND its tuned threshold; F1 must exceed baseline at BOTH thresholds',
        selection='Among passing weights: lowest tuned errors, then highest tuned F1, then lowest weight; otherwise no winner',
        no_follow_on='No OOF, extra fits, added rows, further weights or tuning even if gate passes',
        evaluation_limitation='Reused inner selection; not unbiased independent evaluation; prior raw-vocabulary-sampled NB trials differ, no algorithm novelty claimed',
        numerical_limitation='Frozen transform path may differ numerically from original float32 fit_transform; no scratch-refit equivalence claimed',
        library_versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__, joblib=joblib.__version__),
        failure_policy='Preserve exact warnings/failures and attempt marker; no silent change/retry')
    with (OUT / 'preregistered.json').open('x') as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write('\n')
    protocol_sha = sha(OUT / 'preregistered.json')
    emit('protocol_frozen', sha256=protocol_sha)
    source_paths = ['experiments/v10_nb/run.py', 'experiments/v10_nb/preregistered.json',
                    'experiments/v8_astra_regularization/resume.py',
                    'experiments/v7_astra/inner_results.json',
                    'experiments/v7_astra/guards/inner_full_vocabulary.json',
                    'experiments/v7_astra/models/inner_full_vocabulary.joblib',
                    'experiments/v7_astra/inner_full_vocabulary.npz',
                    'experiments/v5/cat_inner_1.npz', 'phishing_url/__init__.py']
    original_guard = json.loads((ROOT / 'experiments/v7_astra/guards/inner_full_vocabulary.json').read_text())
    source_paths += list(original_guard['config']['sources'])
    originals = {p: sha(ROOT / p) for p in sorted(set(source_paths))}
    write('original_sources.json', originals)
    DIGEST = data_digest(ROOT)
    assert DIGEST == original_guard['data_sha256']
    cache = joblib.load(ROOT / 'data/content_train.joblib')
    assert len(cache['records']) == len(cache['labels']) == len(cache['urls']) == 63900
    splits = aligned_splits(ROOT, cache)
    counts = {k: len(splits[k]) for k in ('development', 'inner_train', 'inner_validation', 'holdout', 'added')}
    assert counts == dict(development=50977, inner_train=34663, inner_validation=6118, holdout=12745, added=178)
    tr, va = splits['inner_train'], splits['inner_validation']
    assert not set(tr) & set(va)
    assert set(tr) | set(va) <= set(splits['development'])
    assert not (set(tr) | set(va)) & (set(splits['holdout']) | set(splits['added']))
    base, origin = checked_base(cache, tr, 'inner_full_vocabulary', DIGEST)
    base_semantic = semantic_hash(base)
    base_views = joblib.hash({k: (v.vocabulary_, v.idf_) for k, v in base.vectorizers_.items()})
    ytr = cache['labels'][tr].copy()
    y = cache['labels'][va].copy()
    assert np.array_equal(np.unique(ytr), [0, 1])
    split_provenance = dict(train=provenance(cache, tr), validation=provenance(cache, va),
                            data_sha256=DIGEST, cache_rows=63900, split_counts=counts,
                            excluded_rows=63900-len(tr)-len(va),
                            limitation='Full cache deserialized; aligned_splits checks original labels and uses development labels for original stratification only. Excluded records are not transformed, fitted, predicted or scored.')
    write('provenance.json', split_provenance)
    np.savez_compressed(OUT / 'split_provenance.npz', train_indices=tr, train_labels=ytr,
                        validation_indices=va, validation_labels=y)
    b = prediction('experiments/v7_astra/inner_full_vocabulary.npz', va, y, tr)
    cat = prediction('experiments/v5/cat_inner_1.npz', va, y)
    frozen = protocol['frozen_baseline_threshold']
    old = .5*b + .5*cat
    baseline = metrics(y, old, frozen)
    assert baseline['errors'] == 51
    assert best_threshold(y, old)[0] == frozen
    write('baseline.json', baseline)
    emit('guarded_baseline', errors=baseline['errors'], f1=baseline['f1'])
    transform_start = time.perf_counter()
    model = ContentTextModel(c=4, nb=True, max_iter=600, random_state=20261008,
                             threads=2, vocabulary_fit_rows=len(tr), verbose=True)
    model.vectorizers_ = copy.deepcopy(base.vectorizers_)
    model.training_rows_ = model.vocabulary_fit_rows_ = len(tr)
    model.vocabulary_policy_ = 'frozen guarded v7 all inner-train rows'
    X = model._matrix(records_at(cache, tr), fit=False)
    V = model._matrix(records_at(cache, va), fit=False)
    assert X.shape[0] == len(tr) and V.shape[0] == len(va) and X.shape[1] == V.shape[1] == base.n_features_
    assert X.dtype == V.dtype == np.float32 and X.data.min() >= 0 and V.data.min() >= 0
    np.testing.assert_array_equal(base.classifier_.predict_proba(V)[:, 1], b)
    transform_seconds = time.perf_counter() - transform_start
    positive = np.asarray(X[ytr == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(X[ytr == 0].sum(axis=0)).ravel() + 1.0
    model.ratio_ = np.log((positive / positive.sum()) / (negative / negative.sum())).astype(np.float32)
    assert np.isfinite(model.ratio_).all()
    X = X.multiply(model.ratio_).tocsr()
    V = V.multiply(model.ratio_).tocsr()
    model.classifier_ = classifier
    model.n_features_ = model.n_features = X.shape[1]
    emit('fit_start', rows=len(tr), features=X.shape[1], transform_seconds=transform_seconds)
    with (OUT / 'fit_started.json').open('x') as f:
        json.dump(dict(classifier=classifier.get_params(), rows=len(tr), protocol_sha256=protocol_sha), f, indent=2)
    fit_start = time.perf_counter()
    FITS += 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier.fit(X, ytr)
    model.fit_seconds_ = model.fit_seconds = time.perf_counter() - fit_start
    model.classes_ = classifier.classes_
    assert np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all()
    assert classifier.get_params() == protocol['classifier']
    assert clone(model).get_params() == model.get_params() and not hasattr(clone(model), 'classifier_')
    p = classifier.predict_proba(V)[:, 1]
    assert p.shape == y.shape and np.isfinite(p).all()
    g = scope(cache, tr, dict(parameters=model.get_params(), classifier_parameters=classifier.get_params(),
                             sources=originals, representation_origin=origin, alpha=1.0,
                             validation_provenance=split_provenance['validation']), DIGEST)
    model.representation_origin_ = origin
    model.nb_training_provenance_ = split_provenance['train']
    model_path = OUT / 'models/inner_nb.joblib'
    save_component(model_path, model, g)
    write('guards/inner_nb.json', g)
    # Save predictions immediately, before checks that could fail after the sole fit.
    np.savez_compressed(OUT / 'inner_nb.npz', train_indices=tr, indices=va, labels=y, probabilities=p,
                        baseline_probabilities=old, replacement_weights=np.array(protocol['replacement_weights']),
                        mixture_probabilities=np.vstack([.5*((1-w)*b+w*p)+.5*cat for w in protocol['replacement_weights']]),
                        protocol_sha256=np.array(protocol_sha), model_sha256=np.array(sha(model_path)),
                        data_sha256=np.array(DIGEST))
    fit_report = dict(classifier_fits=FITS, vectorizer_fits=0, fit_seconds=model.fit_seconds_,
                      transform_seconds=transform_seconds, iterations=classifier.n_iter_.tolist(),
                      warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                      training_shape=list(X.shape), validation_shape=list(V.shape),
                      training_nnz=X.nnz, validation_nnz=V.nnz, ratio_dtype=str(model.ratio_.dtype),
                      ratio_sha256=hashlib.sha256(model.ratio_.tobytes()).hexdigest(),
                      model_sha256=sha(model_path), threadpools=threadpool_info())
    write('fit.json', fit_report)
    emit('fit_done', **fit_report)
    del X, V
    gc.collect()
    replay_start = time.perf_counter()
    restored = load_component(model_path, g)
    replay = restored.predict_proba(records_at(cache, va))
    np.testing.assert_array_equal(p, replay)
    with np.load(OUT / 'inner_nb.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['probabilities'], replay)
        np.testing.assert_array_equal(saved['indices'], va)
        np.testing.assert_array_equal(saved['train_indices'], tr)
        np.testing.assert_array_equal(saved['labels'], y)
    del restored
    gc.collect()
    rejection_checks = []
    bad_scopes = [('reversed_training_order', scope(cache, tr[::-1], g['config'], DIGEST)),
                  ('wrong_data_sha', dict(g, data_sha256='0'*64)),
                  ('wrong_training_rows', dict(g, training_rows=-1))]
    wrong_source = copy.deepcopy(g)
    wrong_source['config']['sources']['phishing_url/html_text.py'] = '0'*64
    bad_scopes.append(('wrong_original_source_sha', wrong_source))
    for reason, bad in bad_scopes:
        try:
            load_component(model_path, bad)
        except ValueError as error:
            rejection_checks.append(dict(case=reason, expected_guard=bad,
                                         exception_type=type(error).__name__, message=str(error)))
        else:
            raise AssertionError('Wrong guard accepted: ' + reason)
        gc.collect()
    write('verification.json', dict(serialization_rows=len(va), serialization_exact=True,
                                    serialization_max_abs_difference=0.0, npz_exact=True,
                                    replay_seconds=time.perf_counter()-replay_start,
                                    wrong_guard_rejections=rejection_checks))
    candidates = []
    for w in protocol['replacement_weights']:
        mixed = .5*((1-w)*b+w*p)+.5*cat
        fixed_m = metrics(y, mixed, frozen)
        tuned_m = metrics(y, mixed, best_threshold(y, mixed)[0])
        passed = all(m['errors'] <= baseline['errors']-2 and m['f1'] > baseline['f1'] for m in (fixed_m, tuned_m))
        candidates.append(dict(weight=w, frozen=fixed_m, tuned=tuned_m, gate_passed=passed))
    passing = [r for r in candidates if r['gate_passed']]
    winner = min(passing, key=lambda r: (r['tuned']['errors'], -r['tuned']['f1'], r['weight'])) if passing else None
    result = dict(baseline=baseline, candidates=candidates, gate_passed=bool(passing), winner=winner,
                  standalone=dict(at_05=metrics(y, p, .5), tuned=metrics(y, p, best_threshold(y, p)[0])),
                  classifier_fits=FITS, protocol_sha256=protocol_sha,
                  limitations=[protocol['evaluation_limitation'], protocol['numerical_limitation'], split_provenance['limitation']])
    write('results.json', result)
    assert semantic_hash(base) == base_semantic
    assert joblib.hash({k: (v.vocabulary_, v.idf_) for k, v in model.vectorizers_.items()}) == base_views
    assert data_digest(ROOT) == DIGEST
    for path, expected in originals.items():
        assert sha(ROOT / path) == expected, path
    assert FITS == 1
    pools = threadpool_info()
    assert all(t['num_threads'] <= 2 for t in pools)
    runtime = time.perf_counter()-START
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak_bytes *= 1024
    execution = dict(runtime_seconds=runtime, peak_rss_bytes=peak_bytes,
                     peak_rss_gib=peak_bytes/1024**3, classifier_fits=FITS,
                     vectorizer_fits=0, threadpools=pools, original_sources_and_inputs_unchanged=True,
                     original_model_semantically_unchanged=True, frozen_vocabulary_idf_unchanged=True,
                     project_files_read=sorted(READS), project_files_written=sorted(WRITES),
                     socket_connect_blocked=True, gate_passed=bool(passing))
    write('execution.json', execution)
    lines = ['# v10 NB: bounded inner experiment', '',
             f"Gate: {'PASS' if passing else 'FAIL'}. Classifier fits: 1. Vectorizer fits: 0. No OOF or follow-on fits.", '',
             f"Baseline: {baseline['errors']} errors, F1={baseline['f1']:.12f}, frozen threshold={frozen}.", '',
             '| Replacement weight | Frozen errors | Frozen F1 | Tuned errors | Tuned F1 | Tuned threshold | Gate |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in candidates:
        lines.append(f"| {r['weight']} | {r['frozen']['errors']} | {r['frozen']['f1']:.12f} | {r['tuned']['errors']} | {r['tuned']['f1']:.12f} | {r['tuned']['threshold']:.17g} | {r['gate_passed']} |")
    lines += ['', f"Standalone NB errors: at 0.5={result['standalone']['at_05']['errors']}; tuned={result['standalone']['tuned']['errors']}.", '',
              'Protocol: LogisticRegression(C=4, solver=liblinear, max_iter=600, random_state=20261008); threads=2; alpha=1.0. Complete defaults and versions are in preregistered.json.',
              'Mixture = 0.5*((1-w)*v7_text+w*nb_text)+0.5*cat_inner_1. The same weight must pass both thresholds with >=2 fewer errors and higher F1.',
              'NB weights use inner TRAIN labels only and the exact html_text.py normalized sum formula on frozen TF-IDF.', '',
              'Rows: 34663 fitted, 6118 scored. Other 23119 cache rows excluded from model operations. Full cache is deserialized; aligned_splits accesses original label metadata for consistency and stratification.',
              f"Runtime: {runtime:.2f} s; classifier fit: {model.fit_seconds_:.2f} s; peak process RSS: {peak_bytes/1024**3:.3f} GiB. Peak includes full cache, sparse transforms, deep copies and verification; it is not incremental model-only memory.", '',
              'Guards: checked_base and scope imported from v8 resume; its audit is not activated. Own strict project input allowlist, output boundary and socket-connect audit are active.',
              'Exact ordered train, data and original source guards checked before reuse. Original files pass final SHA checks; original model semantic hash and frozen vocabulary/IDF hashes are unchanged.',
              'All 6118 validation predictions replay exactly after guarded serialization. Reversed train order, wrong data SHA, row count and original-source SHA are rejected; exact exceptions and requested guards are in verification.json.',
              f"Warnings: {json.dumps(fit_report['warnings'], ensure_ascii=True)}.", '',
              'Limitations: reused inner selection, not an unbiased independent evaluation. Prior sampled-vocabulary NB trials differ; no algorithm novelty claimed. Frozen transform can differ numerically from scratch fit_transform. No forbidden external/reference/test/holdout predictions or labels used; no live URLs/network, routing, SVM, origin/template/memory models or stacking.',
              'Portable estimator: guarded models/inner_nb.joblib contains standard phishing_url.html_text.ContentTextModel, copied v7 vectorizers, ratio and classifier. It needs matching project sources and installed libraries; it does not depend on a custom __main__ class.', '',
              'Files: run.py, preregistered.json, attempt.json, fit_started.json, original_sources.json, provenance.json, split_provenance.npz, baseline.json, fit.json, guards/inner_nb.json, models/inner_nb.joblib, inner_nb.npz, verification.json, results.json, execution.json, run.log, REPORT.md. All writes are under experiments/v10_nb/.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    emit('complete', gate_passed=bool(passing), runtime_seconds=runtime, peak_rss_gib=peak_bytes/1024**3,
         errors=[dict(weight=r['weight'], frozen=r['frozen']['errors'], tuned=r['tuned']['errors']) for r in candidates])


START = time.perf_counter()
FITS = 0
if __name__ == '__main__':
    try:
        with threadpool_limits(limits=2):
            main()
    except BaseException as error:
        import traceback
        write('failure.json', dict(type=type(error).__name__, message=str(error),
                                   traceback=traceback.format_exc(), classifier_fits_attempted=FITS,
                                   runtime_seconds=time.perf_counter()-START,
                                   project_files_read=sorted(READS), project_files_written=sorted(WRITES)))
        (OUT / 'REPORT.md').write_text('Gate: NOT EVALUATED. Exact run failed; no silent parameter change or retry.\n'
                                      f'Classifier fits attempted: {FITS}.\n'
                                      f'{type(error).__name__}: {error}\nSee failure.json and run.log.\n')
        raise
