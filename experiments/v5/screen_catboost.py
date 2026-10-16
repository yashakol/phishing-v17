"""Test larger tree models on the pre-existing inner development split only."""

import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit,load_frame
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import best_threshold,measure


def main():
    out=ROOT/'experiments/v5'
    cache=joblib.load(ROOT/'data/content_train.joblib')
    splits=aligned_splits(ROOT,cache)
    frame=load_frame(cache)
    y=cache['labels']
    del cache
    gc.collect()
    tr,va=splits['inner_train'],splits['inner_validation']
    candidates=[dict(iterations=4500,depth=8,learning_rate=.04,l2_leaf_reg=5),
                dict(iterations=3500,depth=9,learning_rate=.05,l2_leaf_reg=5)]
    results=[]
    for n,config in enumerate(candidates):
        path=out/f'cat_inner_{n}.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        print(json.dumps({'event':'cat_inner_started','candidate':n,'config':config}),flush=True)
        started=time.perf_counter()
        model=fit(frame.iloc[tr],y[tr],config,(frame.iloc[va],y[va]),threads=4)
        p=model.predict_proba(frame.iloc[va])[:,1]
        threshold,_=best_threshold(y[va],p)
        result={'candidate':n,'config':{**config,'iterations':int(model.tree_count_)},
                'metrics':measure(y[va],p,threshold),'seconds':time.perf_counter()-started,
                'scope':'Inner development validation only; no holdout or external test labels.'}
        np.savez_compressed(out/f'cat_inner_{n}.npz',indices=va,labels=y[va],probabilities=p)
        joblib.dump(model,out/f'cat_inner_{n}.joblib',compress=3)
        path.write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        print(json.dumps({'event':'cat_inner_complete',**result}),flush=True)
        del model
        gc.collect()
    selected=max(results,key=lambda row:row['metrics']['f1'])
    (out/'cat_selection.json').write_text(json.dumps({'selected':selected,'candidates':results},indent=2)+'\n')
    print(json.dumps({'event':'cat_selection_complete','selected':selected}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
