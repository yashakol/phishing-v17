"""Evaluate the frozen OOF mixture and fit the complementary models on all train rows."""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/astra'))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit
from experiments.v5.oof import model_parameters,scope_parameters
from phishing_url.content_bundle import prediction_frame
from phishing_url.content_checkpoints import data_digest,guard,load_component,save_component
from phishing_url.content_memory import fit_memory,text_fingerprint
from phishing_url.content_validation import aligned_splits,records_at
from phishing_url.data import write_submission
from phishing_url.evaluation import measure
from phishing_url.rich_text import URLModel
from phishing_url.v5_bundle import predict_v5
from phishing_url.soft_context import context_keys,fit_context

REFIT_NEW_MODELS=False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def subset(cache,indices):
    return {'urls':[cache['urls'][int(i)] for i in indices],
            'records':records_at(cache,indices),'hashes':[cache['hashes'][int(i)] for i in indices],
            'features':cache['features'].iloc[indices].reset_index(drop=True)}


def complementary(cache,frame,indices,kind,stage,digest):
    config=model_parameters(kind)
    scope=guard(cache,indices,scope_parameters(kind,config),digest)
    path=ROOT/f'artifacts/v5/component_cache/{stage}_{kind}.joblib'
    if path.exists() and not REFIT_NEW_MODELS:
        return load_component(path,scope)
    print(json.dumps({'event':'model_fit_started','kind':kind,'stage':stage,'rows':len(indices)}),flush=True)
    started=time.perf_counter()
    if kind=='leaf':
        from phishing_url.leaf_model import LeafModel
        model=LeafModel(config,threads=2).fit(frame.iloc[indices],cache['labels'][indices])
    elif kind=='hash':
        from phishing_url.hashed_text import HashedTextModel
        model=HashedTextModel(**config).fit(records_at(cache,indices),cache['labels'][indices])
    else:
        model=fit(frame.iloc[indices],cache['labels'][indices],config,threads=4)
    save_component(path,model,scope)
    print(json.dumps({'event':'model_fit_complete','kind':kind,'stage':stage,
                      'seconds':time.perf_counter()-started}),flush=True)
    return model


def development_base(cache,frame,splits,digest):
    tr=splits['development']
    config=json.loads((ROOT/'artifacts/v3/config.json').read_text())
    text=load_component(ROOT/'experiments/html_astra/component_cache/holdout_text.joblib',
                         guard(cache,tr,config['text_parameters'],digest))
    boost=load_component(ROOT/'artifacts/v3/component_cache/holdout_boost.joblib',
                          guard(cache,tr,config['numeric_parameters'],digest))
    meta=json.loads((ROOT/'experiments/astra/oof_results.json').read_text())
    assert meta['data_sha256']==sha(ROOT/'data/train.csv')
    assert meta['split_sha256']==sha(ROOT/'reports/split_indices.npz')
    assert (ROOT/'experiments/astra/url_model.py').read_bytes()==(ROOT/'phishing_url/rich_text.py').read_bytes()
    rich_path=ROOT/'experiments/astra/development_model.joblib'
    if rich_path.exists():
        rich=joblib.load(rich_path)
        # The source is byte-identical; store the portable package class.
        rich.__class__=URLModel
    else:
        rich=URLModel(representation='rich',c=16,nb=False).fit(
            [cache['urls'][int(i)] for i in tr],cache['labels'][tr])
    aux_config={'iterations':1000,'depth':7,'learning_rate':.055,'l2_leaf_reg':5,'random_seed':2026,'threads':4}
    path=ROOT/'artifacts/v4/component_cache/development_url_boost.joblib'
    scope=guard(cache,tr,aux_config,digest)
    if path.exists():
        auxiliary=load_component(path,scope)
    else:
        from phishing_url.models import fit_boost
        url_columns=[name for name in frame if name not in set(cache['features'].columns)]
        auxiliary=fit_boost(frame[url_columns].iloc[tr],cache['labels'][tr],threads=4)
        save_component(path,auxiliary,scope)
    return {'text':text,'boost':boost,'aux_url_text':rich,'aux_url_boost':auxiliary,
            'text_weight':.5,'aux_url_weight':.2,'extractor_version':1,
            'feature_columns':list(frame.columns),'known_urls':dict(zip(
                [cache['urls'][int(i)] for i in tr],cache['labels'][tr].tolist(),strict=True)),
            'memory':{},'memory_minimum_support':0,'domain_minimum_support':0,'training_rows':len(tr)}


def main():
    started=time.perf_counter()
    out=ROOT/'reports/v5'
    artifacts=ROOT/'artifacts/v5'
    artifacts.mkdir(parents=True,exist_ok=True)
    selection=json.loads((out/'selection.json').read_text())
    chosen=selection['selected']
    config=chosen['config']
    cache=joblib.load(ROOT/'data/content_train.joblib')
    urls=joblib.load(ROOT/'data/content_url_train.joblib')
    assert cache['urls']==urls['urls']
    frame=prediction_frame(cache,urls['features'])
    splits=aligned_splits(ROOT,cache)
    tr,va=splits['development'],splits['holdout']
    assert not set(tr)&set(va) and len(splits['added'])==178
    digest=data_digest(ROOT)
    assert selection['train_cache_sha256']==digest
    base=development_base(cache,frame,splits,digest)
    leaf=complementary(cache,frame,tr,'leaf','holdout',digest) if config['leaf_weight'] else None
    cat=complementary(cache,frame,tr,'cat','holdout',digest) if config['cat_weight'] else None
    hashed=complementary(cache,frame,tr,'hash','holdout',digest) if config.get('hash_weight',0.) else None
    memory=fit_memory(records_at(cache,tr),[cache['hashes'][int(i)] for i in tr],cache['labels'][tr],3,5)
    bundle={'base':base,'leaf':leaf,'cat':cat,'hashed':hashed,'config':config,'memory':memory,
            'threshold':chosen['threshold'],'training_rows':len(tr)}
    if 'context_config' in chosen:
        bundle.update(context_config=chosen['context_config'],
                      context=fit_context(context_keys(records_at(cache,tr)),cache['labels'][tr]))
    val=subset(cache,va)
    p,hits=predict_v5(bundle,val,urls['features'].iloc[va])
    metrics=measure(cache['labels'][va],p,chosen['threshold'])
    previous=pd.read_csv(ROOT/'reports/v3/holdout_predictions.csv')
    np.testing.assert_array_equal(previous.cache_index.to_numpy(),va)
    np.testing.assert_array_equal(previous.label.to_numpy(),cache['labels'][va])
    assert previous.url.tolist()==val['urls']
    pred=p>=chosen['threshold']
    old=previous.prediction.to_numpy()
    y=cache['labels'][va]
    raw_seen=set(cache['hashes'][int(i)] for i in tr)
    compact_seen={text_fingerprint(record) for record in records_at(cache,tr)}
    novel=np.array([raw not in raw_seen and text_fingerprint(record) not in compact_seen
                    for record,raw in zip(val['records'],val['hashes'],strict=True)])
    results={'selection':chosen,'holdout':metrics,
             'holdout_fixed_errors':int(np.sum((old!=y)&(pred==y))),
             'holdout_new_errors':int(np.sum((old==y)&(pred!=y))),
             'holdout_note':'Previously used holdout, a comparison diagnostic. Configuration and threshold frozen on original development OOF.',
             'unseen_content_rows':int(novel.sum()),
             'unseen_content_metrics':measure(y[novel],p[novel],chosen['threshold']),
             'external_test_labels_used':False,'new_astra_calls':0,'training_rows':len(cache['urls'])}
    pd.DataFrame({'cache_index':va,'url':val['urls'],'label':y,'probability':p,'prediction':pred,
                  'previous_prediction':old,'training_rule_match':hits['training_match'],
                  'unseen_content':novel}).to_csv(out/'holdout_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'event':'holdout_complete',**results}),flush=True)
    del base,bundle,leaf,cat,hashed,memory,val,raw_seen,compact_seen
    gc.collect()
    indices=np.arange(len(cache['urls']))
    full=joblib.load(ROOT/'artifacts/v3/content.joblib')
    assert full['known_urls']==dict(zip(cache['urls'],cache['labels'].tolist(),strict=True))
    assert full['training_rows']==len(cache['urls'])==63900 and 'external_reference' not in full
    leaf=complementary(cache,frame,indices,'leaf','full',digest) if config['leaf_weight'] else None
    cat=complementary(cache,frame,indices,'cat','full',digest) if config['cat_weight'] else None
    hashed=complementary(cache,frame,indices,'hash','full',digest) if config.get('hash_weight',0.) else None
    bundle={'base':full,'leaf':leaf,'cat':cat,'hashed':hashed,'config':config,
            'memory':fit_memory(cache['records'],cache['hashes'],cache['labels'],3,5),
            'threshold':chosen['threshold'],'training_rows':len(indices),
            'train_cache_sha256':digest,'validation':results,'source_metadata':cache['source']}
    if 'context_config' in chosen:
        bundle.update(context_config=chosen['context_config'],
                      context=fit_context(context_keys(cache['records']),cache['labels']),
                      context_source_sha256=sha(ROOT/'phishing_url/soft_context.py'))
    joblib.dump(bundle,artifacts/'ensemble.joblib',compress=3)
    (artifacts/'config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    del cache,frame,urls,full,leaf,cat,hashed
    gc.collect()
    test=joblib.load(ROOT/'data/content_test.joblib')
    test_url=joblib.load(ROOT/'data/content_url_test.joblib')
    assert test['urls']==test_url['urls'] and test['labels'] is None
    p,hits=predict_v5(bundle,test,test_url['features'])
    info=write_submission(ROOT/'data/sample.csv',ROOT/'submissions/submission_content_v5.csv',
                          (p>=chosen['threshold']).astype(np.int8))
    v3=pd.read_csv(ROOT/'submissions/submission_content_v3.csv').Predicted.to_numpy()
    info['prediction_changes_vs_v3']=int(np.sum(v3!=(p>=chosen['threshold'])))
    info['known_training_urls']=int(hits['known_url'].sum())
    info['training_rule_matches']=int(hits['training_match'].sum())
    results.update(submission=info,seconds=time.perf_counter()-started)
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    pd.DataFrame({'Id':range(len(p)),'url':test['urls'],'probability':p,
                  'training_rule_match':hits['training_match'],'known_training_url':hits['known_url']}).to_csv(
                      out/'test_prediction_provenance.csv',index=False)
    print(json.dumps({'event':'v5_complete',**results}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--refit-new-models',action='store_true',
                        help='Refit the selected added models instead of reusing their checked checkpoints.')
    REFIT_NEW_MODELS=parser.parse_args().refit_new_models
    with threadpool_limits(limits=2):
        main()
