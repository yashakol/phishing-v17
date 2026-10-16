"""Train the inner-selected tree candidate on the original three OOF folds."""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit,load_frame
from phishing_url.content_checkpoints import data_digest,guard,load_component,save_component
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import best_threshold,measure


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def model_parameters(kind):
    selected=json.loads((ROOT/f'experiments/v5/{kind}_selection.json').read_text())['selected']
    if kind=='hash':
        return json.loads((ROOT/'experiments/v5/hash_selection.json').read_text())['config']
    return selected['config']


def scope_parameters(kind,config):
    source=ROOT/({'leaf':'phishing_url/leaf_model.py','cat':'experiments/content/boost.py',
                  'hash':'phishing_url/hashed_text.py'}[kind])
    result={'kind':kind,'model_config':config,'source_sha256':sha(source),
            'url_features_sha256':sha(ROOT/'data/content_url_train.joblib'),
            'seed':17021 if kind=='leaf' else 20261006 if kind=='hash' else 2026}
    if kind=='hash':
        result['text_base_source_sha256']=sha(ROOT/'phishing_url/html_text.py')
    return result


def main(kind):
    config=model_parameters(kind)
    parameters=scope_parameters(kind,config)
    out=ROOT/'experiments/v5'
    cache=joblib.load(ROOT/'data/content_train.joblib')
    frame=load_frame(cache)
    dev=aligned_splits(ROOT,cache)['development']
    labels=cache['labels'][dev]
    records=cache['records'] if kind=='hash' else None
    del cache['records'],cache['hashes'],cache['features']
    gc.collect()
    digest=data_digest(ROOT)
    probabilities=np.full(len(dev),np.nan)
    folds=np.full(len(dev),-1,dtype=np.int8)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,labels)):
        checkpoint=out/f'{kind}_oof/fold_{fold}.joblib'
        prediction=checkpoint.with_suffix('.npz')
        scope=guard(cache,dev[tr],parameters,digest)
        scope_json=json.dumps(scope,sort_keys=True)
        started=time.perf_counter()
        if prediction.exists():
            saved=np.load(prediction)
            np.testing.assert_array_equal(saved['indices'],dev[va])
            np.testing.assert_array_equal(saved['labels'],labels[va])
            if str(saved['scope_json'])!=scope_json:
                raise ValueError('OOF prediction scope differs')
            p=saved['probabilities']
            print(json.dumps({'event':'oof_fold_reused','kind':kind,'fold':fold}),flush=True)
        else:
            print(json.dumps({'event':'oof_fold_started','kind':kind,'fold':fold,'config':config}),flush=True)
            if checkpoint.exists():
                model=load_component(checkpoint,scope)
            elif kind=='leaf':
                from phishing_url.leaf_model import LeafModel
                model=LeafModel(config,threads=2).fit(frame.iloc[dev[tr]],labels[tr])
                save_component(checkpoint,model,scope)
            elif kind=='hash':
                from phishing_url.hashed_text import HashedTextModel
                model=HashedTextModel(**config).fit([records[int(i)] for i in dev[tr]],labels[tr])
                save_component(checkpoint,model,scope)
            else:
                model=fit(frame.iloc[dev[tr]],labels[tr],config,threads=4)
                save_component(checkpoint,model,scope)
            p=model.predict_proba([records[int(i)] for i in dev[va]] if kind=='hash' else frame.iloc[dev[va]])
            if kind=='cat':
                p=p[:,1]
            np.savez_compressed(prediction,indices=dev[va],labels=labels[va],probabilities=p,scope_json=scope_json)
            print(json.dumps({'event':'oof_fold_complete','kind':kind,'fold':fold,
                              'metrics_at_half':measure(labels[va],p,.5),
                              'seconds':time.perf_counter()-started}),flush=True)
            del model
            gc.collect()
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Invalid OOF probabilities')
        probabilities[va]=p
        folds[va]=fold
    assert np.isfinite(probabilities).all() and (folds>=0).all()
    threshold,_=best_threshold(labels,probabilities)
    metrics=measure(labels,probabilities,threshold)
    np.savez_compressed(out/f'{kind}_development_oof.npz',indices=dev,labels=labels,
                        probabilities=probabilities,folds=folds)
    result={'parameters':parameters,'metrics':metrics,
            'protocol':'Inner-selected parameters; original development-only three folds; no test/holdout labels.'}
    (out/f'{kind}_oof_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'event':'oof_complete','kind':kind,**result}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['leaf','cat','hash'],required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.kind)
