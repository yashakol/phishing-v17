"""Compare a few pre-existing OOF mixtures with smoothed training context."""

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.v5.select_ensemble import apply_values
from phishing_url.evaluation import best_threshold,measure
from phishing_url.soft_context import apply_count_context
from phishing_url.v5_bundle import mix_probabilities


def main():
    out=ROOT/'reports/v5'
    current=json.loads((out/'selection.json').read_text())
    fixed=json.loads((ROOT/'experiments/v5/hybrid_base_config.json').read_text())
    assert fixed['train_cache_sha256']==current['train_cache_sha256']
    old=np.load(ROOT/'reports/v3/development_oof.npz')
    hashed=np.load(ROOT/'experiments/v5/hash_development_oof.npz')
    cat=np.load(ROOT/'experiments/v5/cat_development_oof.npz')
    stats=np.load(out/'soft_development_oof.npz')
    for table in (hashed,cat,stats):
        np.testing.assert_array_equal(table['indices'],old['indices'])
        np.testing.assert_array_equal(table['labels'],old['labels'])
    y=old['labels']
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(y,y))
    memory=joblib.load(ROOT/'experiments/v5/development_memory.joblib')['values']
    old_cut=.46183375900143303
    old_prediction=old['probabilities']>=old_cut
    base_config={'text_weight':.5,'aux_url_weight':.2,'leaf_weight':0.,'cat_weight':0.,'hash_weight':0.,
                 'memory_minimum_support':3,'domain_minimum_support':5}
    configurations=[base_config,{**base_config,'hash_weight':.25},{**base_config,'hash_weight':.5},
                    fixed['config']]
    rows=[]
    best=None
    best_p=None
    for config in configurations:
        base=mix_probabilities(old['text_probabilities'],old['numeric_probabilities'],old['url_probabilities'],
                               None,cat['probabilities'],config,hashed['probabilities'])
        hard=apply_values(base,memory)
        hits=hard!=base
        for dw in (0.,.3,.7):
            for tw in (0.,.15,.3):
                context={'alpha':5.,'domain_weight':dw,'title_weight':tw,'title_minimum_support':2}
                p=np.zeros(len(y))
                for _,va in folds:
                    counts={'domain':stats['domain_counts'][va],'title':stats['title_counts'][va]}
                    p[va]=apply_count_context(base[va],counts,float(stats['prior'][va[0]]),context)
                p[hits]=hard[hits]
                cut,_=best_threshold(y,p)
                metric=measure(y,p,cut)
                prediction=p>=cut
                row={'config':config,'context_config':context,'threshold':cut,'metrics':metric,
                     'fixed_baseline_errors':int(np.sum((old_prediction!=y)&(prediction==y))),
                     'new_errors':int(np.sum((old_prediction==y)&(prediction!=y))),
                     'fold_comparison':[{'fold':f,'v3_f1':measure(y[va],old['probabilities'][va],old_cut)['f1'],
                                          'v5_f1':measure(y[va],p[va],cut)['f1'],
                                          'v3_errors':int(np.sum(old_prediction[va]!=y[va])),
                                          'v5_errors':int(np.sum(prediction[va]!=y[va]))} for f,(_,va) in enumerate(folds)]}
                rows.append(row)
                if best is None or metric['f1']>best['metrics']['f1']:
                    best=row
                    best_p=p.copy()
    result={'selected':best,'baseline_v3':measure(y,old['probabilities'],old_cut),'candidates':rows,
            'train_cache_sha256':current['train_cache_sha256'],'included_oof_components':['hash','cat'],
            'protocol':'Four fixed OOF probability mixtures and nine gentle context combinations. Training-only context counts from the original folds. Threshold and weights selected on development OOF; no holdout/test/added-row labels loaded.',
            'external_test_labels_used':False,'new_astra_calls':0}
    (out/'hybrid_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'hybrid_development_oof.npz',indices=old['indices'],labels=y,probabilities=best_p,
                        text=old['text_probabilities'],numeric=old['numeric_probabilities'],url=old['url_probabilities'],
                        leaf=np.zeros(len(y)),cat=cat['probabilities'],hashed=hashed['probabilities'])
    print(json.dumps({'event':'hybrid_selected','selected':best,'baseline':result['baseline_v3']}),flush=True)


if __name__=='__main__':
    main()
