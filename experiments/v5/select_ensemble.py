"""Select a small ensemble grid on development OOF, retaining the v3 memory rules."""

import argparse
import gc
import json
import sys
from itertools import product
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from phishing_url.content_checkpoints import data_digest
from phishing_url.content_memory import text_fingerprint
from phishing_url.content_validation import aligned_splits,records_at
from phishing_url.evaluation import best_threshold,measure
from phishing_url.features import group_for_url
from phishing_url.v5_bundle import mix_probabilities


def apply_values(base,values):
    raw,txt,dom=values['raw'],values['text'],values['domain']
    page=np.where(raw>=0,raw,txt)
    page_present=page>=0
    domain_eligible=(dom>=0)&(page_present|((base>=.05)&(base<=.95)))
    conflict=((raw>=0)&(txt>=0)&(raw!=txt))|(domain_eligible&page_present&(dom!=page))
    result=base.copy()
    take_page=page_present&~conflict
    take_domain=domain_eligible&~page_present&~conflict
    result[take_page]=page[take_page]
    result[take_domain]=dom[take_domain]
    return result


def memory_values(keys,y,folds):
    values={}
    for kind,support in [('raw',3),('text',3),('domain',5)]:
        encoded,unique=pd.factorize(np.asarray(keys[kind],dtype=object),sort=False)
        result=np.full(len(y),-1,dtype=np.int8)
        for tr,va in folds:
            zero=np.bincount(encoded[tr],weights=y[tr]==0,minlength=len(unique))
            one=np.bincount(encoded[tr],weights=y[tr]==1,minlength=len(unique))
            code=encoded[va]
            result[va[(zero[code]>=support)&(one[code]==0)]]=0
            result[va[(one[code]>=support)&(zero[code]==0)]]=1
        values[kind]=result
    return values


def main(include_cat,include_hash):
    out=ROOT/'reports/v5'
    out.mkdir(exist_ok=True)
    old=np.load(ROOT/'reports/v3/development_oof.npz')
    leaf=np.load(ROOT/'experiments/v5/leaf_development_oof.npz')
    cat=np.load(ROOT/'experiments/v5/cat_development_oof.npz') if include_cat else None
    hashed=np.load(ROOT/'experiments/v5/hash_development_oof.npz') if include_hash else None
    cache=joblib.load(ROOT/'data/content_train.joblib')
    dev=aligned_splits(ROOT,cache)['development']
    y=cache['labels'][dev]
    for table in (old,leaf,cat,hashed):
        if table is not None:
            np.testing.assert_array_equal(table['indices'],dev)
            np.testing.assert_array_equal(table['labels'],y)
    keys={'raw':[cache['hashes'][int(i)] for i in dev],
          'text':[text_fingerprint(record) for record in records_at(cache,dev)],
          'domain':[group_for_url(cache['urls'][int(i)]) for i in dev]}
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,y))
    for fold,(_,va) in enumerate(folds):
        assert np.all(leaf['folds'][va]==fold)
        if cat is not None:
            assert np.all(cat['folds'][va]==fold)
        if hashed is not None:
            assert np.all(hashed['folds'][va]==fold)
    values=memory_values(keys,y,folds)
    digest=data_digest(ROOT)
    joblib.dump({'indices':dev,'keys':keys,'values':values,'data_sha256':digest},
                ROOT/'experiments/v5/development_memory.joblib',compress=3)
    del cache,keys
    gc.collect()
    text,numeric,url=[old[name] for name in ('text_probabilities','numeric_probabilities','url_probabilities')]
    old_config=json.loads((ROOT/'reports/v3/ensemble_selection.json').read_text())['selected']
    baseline_config={'text_weight':.5,'aux_url_weight':.2,'leaf_weight':0.,'cat_weight':0.}
    baseline=apply_values(mix_probabilities(text,numeric,url,None,None,baseline_config),values)
    np.testing.assert_allclose(baseline,old['probabilities'],rtol=0,atol=1e-14)
    baseline_prediction=baseline>=old_config['threshold']
    baseline_metrics=measure(y,baseline,old_config['threshold'])
    candidates=[]
    predictions={}
    for tw in (.5,.7,.85):
        for uw in (.1,.2):
            for lw,cw,hw in product(
                    (0.,.1,.2) if include_cat or include_hash else (0.,.1,.2,.3,.4,.5),
                    (0.,.15,.3) if include_cat else (0.,),
                    (0.,.25,.5,.75,1.) if include_hash else (0.,)):
                    config={'text_weight':tw,'aux_url_weight':uw,'leaf_weight':lw,'cat_weight':cw,
                            'hash_weight':hw,'memory_minimum_support':3,'domain_minimum_support':5}
                    base=mix_probabilities(text,numeric,url,leaf['probabilities'],
                                           cat['probabilities'] if cat is not None else None,config,
                                           hashed['probabilities'] if hashed is not None else None)
                    p=apply_values(base,values)
                    cut,_=best_threshold(y,p)
                    metrics=measure(y,p,cut)
                    prediction=p>=cut
                    fold_comparison=[]
                    for fold,(_,va) in enumerate(folds):
                        before=measure(y[va],baseline[va],old_config['threshold'])
                        after=measure(y[va],p[va],cut)
                        fold_comparison.append({'fold':fold,'v3_f1':before['f1'],'v5_f1':after['f1'],
                                                 'v3_errors':before['fp']+before['fn'],
                                                 'v5_errors':after['fp']+after['fn']})
                    result={'config':config,'metrics':metrics,'threshold':cut,
                            'fixed_baseline_errors':int(np.sum((baseline_prediction!=y)&(prediction==y))),
                            'new_errors':int(np.sum((baseline_prediction==y)&(prediction!=y))),
                            'fold_comparison':fold_comparison}
                    candidates.append(result)
                    predictions[json.dumps(config,sort_keys=True)]=p
    candidates.sort(key=lambda row:(row['metrics']['f1'],-row['config']['cat_weight'],
                                    -abs(row['threshold']-.5)),reverse=True)
    chosen=candidates[0]
    p=predictions[json.dumps(chosen['config'],sort_keys=True)]
    result={'selected':chosen,'baseline_v3':baseline_metrics,'candidates':candidates,
            'included_oof_components':['leaf']+(['cat'] if include_cat else [])+(['hash'] if include_hash else []),
            'protocol':'Coarse probability mixture grid on original development OOF only. V3 page support 3 and domain support 5 retained. Holdout, added 178 URLs and test labels excluded.',
            'external_test_labels_used':False,'new_astra_calls':0,'train_cache_sha256':digest}
    (out/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'development_oof.npz',indices=dev,labels=y,probabilities=p,
                        text=text,numeric=numeric,url=url,leaf=leaf['probabilities'],
                        cat=cat['probabilities'] if cat is not None else np.zeros(len(y)),
                        hashed=hashed['probabilities'] if hashed is not None else np.zeros(len(y)))
    print(json.dumps({'event':'ensemble_selected','selected':chosen,'baseline_v3':baseline_metrics}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-cat',action='store_true')
    parser.add_argument('--include-hash',action='store_true')
    args=parser.parse_args()
    main(args.include_cat,args.include_hash)
