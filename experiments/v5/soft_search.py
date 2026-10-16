"""Tune gentle context evidence on the same original development OOF only."""

import gc
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from phishing_url.content_checkpoints import data_digest
from phishing_url.content_validation import aligned_splits,records_at
from phishing_url.evaluation import best_threshold,measure
from phishing_url.soft_context import apply_count_context,context_keys,fit_context,lookup_counts


def main():
    out=ROOT/'reports/v5'
    old=np.load(ROOT/'reports/v3/development_oof.npz')
    cache=joblib.load(ROOT/'data/content_train.joblib')
    dev=aligned_splits(ROOT,cache)['development']
    np.testing.assert_array_equal(dev,old['indices'])
    y=cache['labels'][dev]
    np.testing.assert_array_equal(y,old['labels'])
    keys=context_keys(records_at(cache,dev))
    digest=data_digest(ROOT)
    del cache
    gc.collect()
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,y))
    counts={kind:np.zeros((len(dev),2),dtype=np.int32) for kind in keys}
    prior=np.zeros(len(dev))
    for tr,va in folds:
        context=fit_context({kind:[values[int(i)] for i in tr] for kind,values in keys.items()},y[tr])
        actual=lookup_counts(context,{kind:[values[int(i)] for i in va] for kind,values in keys.items()})
        for kind in counts:
            counts[kind][va]=actual[kind]
        prior[va]=context['prior']
    base=.8*(.5*old['text_probabilities']+.5*old['numeric_probabilities'])+.2*old['url_probabilities']
    memory=joblib.load(ROOT/'experiments/v5/development_memory.joblib')['values']
    from experiments.v5.select_ensemble import apply_values
    baseline=apply_values(base,memory)
    np.testing.assert_allclose(baseline,old['probabilities'],rtol=0,atol=1e-14)
    # Every old hard rule is retained with its original eligibility decision.
    hard=(baseline!=base)
    old_cut=.46183375900143303
    old_prediction=baseline>=old_cut
    rows=[]
    best_p=None
    for alpha in (5.,15.,40.):
        for domain_weight in (0.,.3,.7):
            for title_weight in (0.,.15,.3,.5):
                config={'alpha':alpha,'domain_weight':domain_weight,'title_weight':title_weight,
                        'title_minimum_support':2}
                p=np.zeros(len(y))
                for _,va in folds:
                    p[va]=apply_count_context(base[va],{kind:value[va] for kind,value in counts.items()},
                                              float(prior[va[0]]),config)
                p[hard]=baseline[hard]
                cut,_=best_threshold(y,p)
                metric=measure(y,p,cut)
                pred=p>=cut
                row={'context_config':config,'metrics':metric,'threshold':cut,
                     'fixed_baseline_errors':int(np.sum((old_prediction!=y)&(pred==y))),
                     'new_errors':int(np.sum((old_prediction==y)&(pred!=y))),
                     'fold_comparison':[{'fold':f,'v3_f1':measure(y[va],baseline[va],old_cut)['f1'],
                       'v5_f1':measure(y[va],p[va],cut)['f1'],
                       'v3_errors':int(np.sum(old_prediction[va]!=y[va])),
                       'v5_errors':int(np.sum(pred[va]!=y[va]))} for f,(_,va) in enumerate(folds)]}
                rows.append(row)
                if best_p is None or metric['f1']>max(r['metrics']['f1'] for r in rows[:-1]):
                    best_p=p.copy()
    chosen=max(rows,key=lambda row:row['metrics']['f1'])
    chosen['config']={'text_weight':.5,'aux_url_weight':.2,'leaf_weight':0.,'cat_weight':0.,'hash_weight':0.,
                      'memory_minimum_support':3,'domain_minimum_support':5}
    result={'selected':chosen,'baseline_v3':measure(y,baseline,old_cut),'candidates':rows,
            'included_oof_components':[],'train_cache_sha256':digest,
            'protocol':'Frozen v3 base and hard rules. Smoothed domain/title class counts fitted on each development fold training only. Coarse context weights and threshold selected on original OOF; holdout/test/added rows excluded.',
            'external_test_labels_used':False,'new_astra_calls':0}
    (out/'soft_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'soft_development_oof.npz',indices=dev,labels=y,probabilities=best_p,
                        base=base,domain_counts=counts['domain'],title_counts=counts['title'],prior=prior)
    joblib.dump({'indices':dev,'keys':keys,'data_sha256':digest},ROOT/'experiments/v5/soft_keys.joblib',compress=3)
    print(json.dumps({'event':'soft_context_selected','selected':chosen,'baseline':result['baseline_v3']}),flush=True)


if __name__=='__main__':
    main()
