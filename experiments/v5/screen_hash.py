"""Evaluate full-training hashed text on the protected inner split."""

import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from phishing_url.content_checkpoints import data_digest,guard,save_component
from phishing_url.content_validation import aligned_splits,records_at
from phishing_url.evaluation import best_threshold,measure
from phishing_url.hashed_text import HashedTextModel


def main():
    started=time.perf_counter()
    out=ROOT/'experiments/v5'
    cache=joblib.load(ROOT/'data/content_train.joblib')
    splits=aligned_splits(ROOT,cache)
    tr,va=splits['inner_train'],splits['inner_validation']
    y=cache['labels'][tr]
    y_valid=cache['labels'][va]
    records,valid=records_at(cache,tr),records_at(cache,va)
    config=json.loads((ROOT/'experiments/html_astra/selection.json').read_text())['model_parameters']
    scope=guard(cache,tr,{'kind':'hashed_text','base_config':config},data_digest(ROOT))
    del cache
    gc.collect()
    model=HashedTextModel(**config)
    x=model._matrix(records,fit=True)
    xv=model._matrix(valid)
    prior=np.load(ROOT/'experiments/html_astra/inner_predictions.npz')
    np.testing.assert_array_equal(prior['indices'],va)
    cat=np.load(out/'cat_inner_1.npz')
    np.testing.assert_array_equal(cat['indices'],va)
    candidates=[]
    classifiers={}
    probabilities={}
    for c in (4.,16.,64.):
        classifier=LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=20261006)
        classifier.fit(x,y)
        p=classifier.predict_proba(xv)[:,1]
        cut,_=best_threshold(y_valid,p)
        mixes=[]
        for old_weight in (0.,.25,.5):
            text=(1-old_weight)*p+old_weight*prior['probabilities']
            proba=.5*text+.5*cat['probabilities']
            threshold,_=best_threshold(y_valid,proba)
            mixes.append({'old_text_weight':old_weight,**measure(y_valid,proba,threshold)})
        result={'c':c,'metrics':measure(y_valid,p,cut),'best_cat_blend':max(mixes,key=lambda row:row['f1']),
                'cat_blends':mixes}
        candidates.append(result)
        classifiers[c]=classifier
        probabilities[c]=p
        print(json.dumps({'event':'hashed_inner_candidate',**result}),flush=True)
    selected=max(candidates,key=lambda row:row['best_cat_blend']['f1'])
    model.c=selected['c']
    model.classifier_=classifiers[model.c]
    model.ratio_=None
    model.classes_=model.classifier_.classes_
    model.n_features_=x.shape[1]
    config={**config,'c':model.c}
    save_component(out/'hash_inner.joblib',model,scope)
    np.savez_compressed(out/'hash_inner.npz',indices=va,labels=y_valid,probabilities=probabilities[model.c])
    result={'selected':selected,'candidates':candidates,'config':config,
            'protocol':'Hashed counts and IDF fitted on inner training only; C and blends selected on inner validation only.',
            'seconds':time.perf_counter()-started}
    (out/'hash_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'event':'hash_screen_complete',**result}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
