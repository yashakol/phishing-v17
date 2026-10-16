"""Compare complementary tree models within the protected development partition."""

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

from experiments.content.boost import load_frame
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import best_threshold,measure
from phishing_url.leaf_model import LeafModel


def main():
    out=ROOT/'experiments/v5'
    cache=joblib.load(ROOT/'data/content_train.joblib')
    splits=aligned_splits(ROOT,cache)
    frame=load_frame(cache)
    y=cache['labels']
    del cache
    gc.collect()
    tr,va=splits['inner_train'],splits['inner_validation']
    text=np.load(ROOT/'experiments/html_astra/inner_predictions.npz')
    np.testing.assert_array_equal(text['indices'],va)
    p_text=text['probabilities']
    common=dict(n_estimators=4500,learning_rate=.035,colsample_bytree=.9,subsample=.85,
                subsample_freq=1,reg_lambda=5,min_data_per_group=10,cat_smooth=10)
    candidates=[{**common,'num_leaves':31,'min_child_samples':20},
                {**common,'num_leaves':63,'min_child_samples':15},
                {**common,'num_leaves':127,'min_child_samples':25}]
    results=[]
    for n,config in enumerate(candidates):
        path=out/f'leaf_inner_{n}.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        print(json.dumps({'event':'leaf_inner_started','candidate':n,'config':config}),flush=True)
        started=time.perf_counter()
        model=LeafModel(config,threads=2).fit(frame.iloc[tr],y[tr],(frame.iloc[va],y[va]))
        p=model.predict_proba(frame.iloc[va])
        threshold,_=best_threshold(y[va],p)
        blends=[]
        for weight in (0.,.25,.5,.75):
            mixture=weight*p_text+(1-weight)*p
            cut,_=best_threshold(y[va],mixture)
            blends.append({'text_weight':weight,**measure(y[va],mixture,cut)})
        result={'candidate':n,'config':{**config,'n_estimators':model.iterations_},
                'metrics':measure(y[va],p,threshold),'best_text_blend':max(blends,key=lambda row:row['f1']),
                'blends':blends,'seconds':time.perf_counter()-started,
                'scope':'Inner development only; no holdout or external/test labels.'}
        np.savez_compressed(out/f'leaf_inner_{n}.npz',indices=va,labels=y[va],probabilities=p)
        joblib.dump(model,out/f'leaf_inner_{n}.joblib',compress=3)
        path.write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        print(json.dumps({'event':'leaf_inner_complete',**result}),flush=True)
        del model
        gc.collect()
    selected=max(results,key=lambda row:row['best_text_blend']['f1'])
    (out/'leaf_selection.json').write_text(json.dumps({'selected':selected,'candidates':results},indent=2)+'\n')
    print(json.dumps({'event':'leaf_selection_complete','selected':selected}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
