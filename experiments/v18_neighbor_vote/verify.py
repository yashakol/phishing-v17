"""Independent raw-vote replay, decision audit and human-readable report."""
import sys
sys.dont_write_bytecode = True
import run as boundary
import gc
import json
import math
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold

ROOT, OUT, OLD = boundary.ROOT, boundary.OUT, boundary.OLD


def metric(y, p, threshold):
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, errors=fp+fn, f1=2*tp/(2*tp+fp+fn))


def main():
    cfg = json.loads((OUT / 'preregistered.json').read_text())
    result = json.loads((OUT / 'results.json').read_text())
    execution = json.loads((OUT / 'execution.json').read_text())
    boundary.check_frozen(cfg)
    assert result['protocol_sha256'] == boundary.sha(OUT / 'preregistered.json')
    assert result['source_sha256'] == cfg['sources_sha256']['run.py']
    with np.load(OUT / 'development_vote.npz', allow_pickle=False) as source:
        a = {k: source[k] for k in source.files}
    dev, y, fold_ids = a['indices'], a['labels'], a['folds']
    cache = joblib.load(ROOT / 'data/content_train.joblib')
    np.testing.assert_array_equal(y, cache['labels'][dev])
    del cache; gc.collect()
    with np.load(OUT / 'own_development_labels.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(dev, b['indices']); np.testing.assert_array_equal(y, b['labels'])
    with np.load(OLD / 'development_oof.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(dev, b['indices']); np.testing.assert_array_equal(y, b['labels'])
        np.testing.assert_array_equal(a['baseline'], b['baseline_v15'])
        np.testing.assert_array_equal(a['eligible'], b['eligible_v15'])
        old_queried, residual_p = b['queried'], b['neighbor_v15']
    old_cfg = json.loads((OLD / 'preregistered.json').read_text())
    for k in ('eligible_probability', 'neighbors', 'minimum_similarity', 'minimum_support', 'power', 'smoothing', 'splitter', 'gate', 'frozen_v15_threshold'):
        assert cfg[k] == old_cfg[k]
    replay_vote = np.zeros(len(y)); replay_support = np.zeros(len(y), dtype=int)
    edges, queries, max_difference = 0, 0, 0.
    for f, (tr, va) in enumerate(StratifiedKFold(**cfg['splitter']).split(dev, y)):
        assert (fold_ids[va] == f).all()
        train_lookup = {int(index): int(y[pos]) for pos, index in zip(tr, dev[tr])}
        prior = sum(train_lookup.values()) / len(train_lookup)
        np.testing.assert_array_equal(a['fold_train_prior'][va], np.full(len(va), prior))
        assert result['folds'][f]['prior'] == prior
        replay_vote[va] = prior
        with np.load(OLD / f'fold_{f}_retrieval.npz', allow_pickle=False) as b:
            qp = va[old_queried[va]]
            np.testing.assert_array_equal(b['query_indices'], dev[qp])
            np.testing.assert_array_equal(b['train_indices'], dev[tr]); np.testing.assert_array_equal(b['validation_indices'], dev[va])
            for row, pos in enumerate(qp):
                terms, weights = [], []
                for index, similarity, label in zip(b['neighbor_indices'][row], b['similarities'][row], b['neighbor_labels'][row]):
                    if index < 0:
                        assert label == -1 and similarity == 0
                        continue
                    assert int(index) in train_lookup and int(index) != int(dev[pos])
                    assert train_lookup[int(index)] == int(label)
                    assert float(similarity) >= cfg['minimum_similarity']
                    w = float(similarity) ** cfg['power']
                    weights.append(w); terms.append(w * int(label)); edges += 1
                value = (math.fsum(terms) + cfg['smoothing']*prior) / (math.fsum(weights) + cfg['smoothing'])
                replay_vote[pos] = value; replay_support[pos] = len(weights); queries += 1
                assert len(weights) == int(b['support'][row])
    np.testing.assert_allclose(replay_vote, a['votes'], atol=2e-15, rtol=0)
    max_difference = float(np.max(abs(replay_vote - a['votes'])))
    np.testing.assert_array_equal(replay_support, a['support'])
    applied = (a['baseline'] >= cfg['eligible_probability'][0]) & (a['baseline'] <= cfg['eligible_probability'][1]) & (replay_support >= cfg['minimum_support'])
    np.testing.assert_array_equal(applied, a['applied'])
    np.testing.assert_array_equal(a['weights'], cfg['weights'])
    metric_checks, gate_passes = 0, []
    for i, c in enumerate(result['candidates']):
        w = cfg['weights'][i]; assert c['weight'] == w
        p = np.where(applied, (1-w)*a['baseline'] + w*a['votes'], a['baseline'])
        np.testing.assert_array_equal(p, a['probabilities'][i])
        np.testing.assert_array_equal(p[~applied], a['baseline'][~applied])
        for mode, row in c['comparisons'].items():
            for key, probabilities in (('before', a['baseline']), ('after', p)):
                saved = row[key]
                for k, value in metric(y, probabilities, saved['threshold']).items():
                    assert abs(saved[k]-value) < 1e-14
            if mode == 'frozen_v15':
                assert row['before']['threshold'] == row['after']['threshold'] == cfg['frozen_v15_threshold']
            before, after = a['baseline'] >= row['before']['threshold'], p >= row['after']['threshold']
            fixed, new = (before != y) & (after == y), (before == y) & (after != y)
            assert row['fixed_indices'] == dev[fixed].tolist() and row['new_error_indices'] == dev[new].tolist()
            assert row['fixed_errors'] == int(fixed.sum()) and row['new_errors'] == int(new.sum())
            assert row['net_fixes'] == int(fixed.sum()-new.sum())
            for f in row['folds']:
                mask = fold_ids == f['fold']
                assert f['before_errors'] == int((before[mask] != y[mask]).sum())
                assert f['after_errors'] == int((after[mask] != y[mask]).sum())
                assert f['fixed'] == int(fixed[mask].sum()) and f['new'] == int(new[mask].sum())
            metric_checks += 1
        g = cfg['gate']; required = []
        for mode in g['modes']:
            row = c['comparisons'][mode]
            required.extend([row['net_fixes'] >= g['net_fewer_errors_vs_v15_at_each_mode'], row['after']['f1'] > row['before']['f1'],
                             sum(f['after_errors'] < f['before_errors'] for f in row['folds']) >= g['improved_folds_at_each_mode'],
                             max(f['after_errors'] - f['before_errors'] for f in row['folds']) <= g['maximum_fold_error_increase_at_each_mode']])
        assert c['gate_passed'] == all(required); gate_passes.append(all(required))
        d = result['vote_vs_v16_residual'][i]
        assert d['probability_rows_different_from_v16'] == int((p != residual_p).sum())
        assert d['max_abs_probability_difference'] == float(abs(p-residual_p).max())
        assert d['decisions_different_from_v16_at_frozen'] == int(((p >= cfg['frozen_v15_threshold']) != (residual_p >= cfg['frozen_v15_threshold'])).sum())
    assert result['gate_passed'] == any(gate_passes)
    pool = [c for c in result['candidates'] if c['gate_passed']] or result['candidates']
    chosen = max(pool, key=lambda c: (c['comparisons']['individually_tuned']['after']['f1'], c['comparisons']['frozen_v15']['net_fixes'], -c['weight']))
    config = dict(weight=chosen['weight'], threshold=chosen['threshold'])
    assert result['best_observed_config'] == config
    assert result['selected_config'] == (config if any(gate_passes) else None)
    assert all(p.startswith('experiments/v18_neighbor_vote/') for p in execution['project_files_written'])
    assert all(t['num_threads'] <= 2 for t in execution['threadpools'])
    boundary.check_frozen(cfg)
    files = sorted({p.name for p in OUT.iterdir() if p.is_file()} | {'verification.json', 'REPORT.md'})
    report = dict(status='passed', neighbor_edges_verified=edges, query_rows_replayed=queries,
                  max_independent_vote_difference=max_difference, raw_labels_directly_rechecked_against_train_cache=True,
                  original_retrieval_parameters_unchanged=True, train_only_prior_and_neighbors=True, no_self_neighbors=True,
                  serialization_mix_replay_bitwise_identical=True, metric_comparisons_verified=metric_checks,
                  selection_gate_recomputed=True, v16_unchanged=True, all_source_and_input_hashes_match=True,
                  threads_le_2=True, classifier_fits=0, retrieval_fits=0,
                  changed_paths=[str(OUT / name) for name in files],
                  output_sha256={p.name: boundary.sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('verification.json', 'REPORT.md')})
    boundary.write('verification.json', report)
    lines = ['# v18_neighbor_vote: raw train-label vote', '',
             f"Decision: **{result['decision']}**. Gate passed: **{result['gate_passed']}**.", '',
             f"Selected config: `{json.dumps(result['selected_config'])}`. Best observed config: `{json.dumps(result['best_observed_config'])}`. A failed candidate is not selected for use.", '',
             '## Results', '', '| Weight | Threshold mode | Errors before -> after | F1 before -> after | Fixes / new | Net fixes | Gate |',
             '|---:|---|---:|---:|---:|---:|---|']
    for c in result['candidates']:
        for mode in cfg['gate']['modes']:
            r = c['comparisons'][mode]
            lines.append(f"| {c['weight']} | {mode} | {r['before']['errors']} -> {r['after']['errors']} | {r['before']['f1']:.10f} -> {r['after']['f1']:.10f} | {r['fixed_errors']} / {r['new_errors']} | {r['net_fixes']} | {c['gate_passed']} |")
    lines += ['', f"Frozen v15 threshold: `{cfg['frozen_v15_threshold']}`. Candidate thresholds and all fixed/new own cache indices are in results.json. No conditions changed after scoring.", '',
              '## Best observed folds', '', '| Mode | Fold | Errors before -> after | Fixes / new |', '|---|---:|---:|---:|']
    for mode in cfg['gate']['modes']:
        for f in chosen['comparisons'][mode]['folds']:
            lines.append(f"| {mode} | {f['fold']} | {f['before_errors']} -> {f['after_errors']} | {f['fixed']} / {f['new']} |")
    lines += ['', 'Gate requires >=5 net fixes in BOTH frozen and separately tuned modes, higher F1 in both, >=2 improved folds in both, and no fold more than two errors worse. Detailed Boolean gate checks are saved for every candidate.', '',
              '## Verified distinction from C16 residual correction', '',
              'This variant uses weighted raw TRAIN labels and the complete outer-train class prior. Neighbor C16 probabilities and in-sample residuals are not inputs to the vote. Same links, similarities, support and baseline uncertainty as v16. No residual cap; exactly the requested convex mixtures.', '',
              '| Weight | Probability rows different from v16 | Max absolute difference | Different frozen decisions vs v16 |', '|---:|---:|---:|---:|']
    for d in result['vote_vs_v16_residual']:
        lines.append(f"| {d['weight']} | {d['probability_rows_different_from_v16']} | {d['max_abs_probability_difference']:.10f} | {d['decisions_different_from_v16_at_frozen']} |")
    lines += ['', '## Checks and scope', '',
              f"Original development: {len(y)} rows, three folds. {queries} cached query rows and {edges} neighbor edges replayed. {int(applied.sum())} rows receive a vote mixture; all other baseline probabilities remain bitwise identical.", '',
              f"Direct raw-label match against own train cache passed in both runner and verifier. Original development URL/label alignment, fold train indices and ordered C16 training guard checked before scoring. Every neighbor is outer-train and never the query or validation. Priors computed independently from each outer-train side. Float64 scalar vote replay maximum difference: {max_difference:.3g}.", '',
              f"All three mixtures replayed bitwise after serialization; {metric_checks} metric comparisons, fixed/new indices and the selection decision independently verified. All v16 files retain their preregistered hashes. Thread count <=2. No new classifier, vectorizer or retrieval fits. Runtime: {result['seconds']:.2f} seconds.", '',
              'No prohibited evaluation data/files, network, live URLs, full refit or release used. The train cache is deserialized to compare development labels; excluded labels are not indexed or scored. Reused development and inherited incomplete nesting of v15 remain limitations. No public improvement claim.', '',
              '## Exact changed paths', '']
    lines += [f'- `{OUT / name}`' for name in files]
    lines.append('')
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in report.items() if k not in ('output_sha256', 'changed_paths')}, indent=2))


if __name__ == '__main__':
    main()
