"""Check fold scopes, replay OOF memory and repeat saved-model inference offline."""

import gc
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.v5.oof import model_parameters,scope_parameters
from phishing_url.content_bundle import prediction_frame
from phishing_url.content_checkpoints import data_digest,guard,load_component
from phishing_url.content_memory import apply_memory,fit_memory,text_fingerprint
from phishing_url.content_validation import aligned_splits,records_at
from phishing_url.data import load_test,write_submission
from phishing_url.evaluation import measure
from phishing_url.v5_bundle import mix_probabilities,predict_v5
from phishing_url.soft_context import apply_context,context_keys,fit_context


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    out=ROOT/'reports/v5'
    selection=json.loads((out/'selection.json').read_text())
    selected=selection['selected']
    config=selected['config']
    assert selected['metrics']['f1']>selection['baseline_v3']['f1'], 'No development improvement'
    cache=joblib.load(ROOT/'data/content_train.joblib')
    urls=joblib.load(ROOT/'data/content_url_train.joblib')
    assert cache['urls']==urls['urls']
    frame=prediction_frame(cache,urls['features'])
    splits=aligned_splits(ROOT,cache)
    dev=splits['development']
    table=np.load(out/'development_oof.npz')
    np.testing.assert_array_equal(table['indices'],dev)
    np.testing.assert_array_equal(table['labels'],cache['labels'][dev])
    y=table['labels']
    base=mix_probabilities(table['text'],table['numeric'],table['url'],table['leaf'],table['cat'],
                           config,table['hashed'])
    p=np.zeros(len(y))
    fingerprints=[text_fingerprint(record) for record in records_at(cache,dev)]
    digest=data_digest(ROOT)
    assert digest==selection['train_cache_sha256']
    model_checks=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,y)):
        memory=fit_memory(records_at(cache,dev[tr]),[cache['hashes'][int(i)] for i in dev[tr]],y[tr],3,5,
                          [fingerprints[int(i)] for i in tr])
        hard,hits=apply_memory(memory,records_at(cache,dev[va]),[cache['hashes'][int(i)] for i in dev[va]],
                               base[va],[fingerprints[int(i)] for i in va])
        if 'context_config' in selected:
            context=fit_context(context_keys(records_at(cache,dev[tr])),y[tr])
            p[va]=apply_context(context,records_at(cache,dev[va]),base[va],selected['context_config'])
            p[va[hits]]=hard[hits]
            model_checks.append({'kind':'smoothed_context','fold':fold,'replayed_rows':len(va),
                                 'training_rows':len(tr),'training_scope':'fold training only'})
        else:
            p[va]=hard
        for kind,weight in [('leaf',config['leaf_weight']),('cat',config['cat_weight']),
                            ('hash',config.get('hash_weight',0.))]:
            if not weight:
                continue
            saved=np.load(ROOT/f'experiments/v5/{kind}_oof/fold_{fold}.npz')
            np.testing.assert_array_equal(saved['indices'],dev[va])
            np.testing.assert_array_equal(saved['labels'],y[va])
            expected=guard(cache,dev[tr],scope_parameters(kind,model_parameters(kind)),digest)
            assert str(saved['scope_json'])==json.dumps(expected,sort_keys=True)
            model=load_component(ROOT/f'experiments/v5/{kind}_oof/fold_{fold}.joblib',expected)
            positions=va[:32] if kind=='hash' else va
            actual=model.predict_proba(records_at(cache,dev[positions]) if kind=='hash' else frame.iloc[dev[positions]])
            if kind=='cat':
                actual=actual[:,1]
            field='hashed' if kind=='hash' else kind
            np.testing.assert_allclose(actual,table[field][positions],rtol=0,atol=1e-14)
            model_checks.append({'kind':kind,'fold':fold,'replayed_rows':len(positions),
                                 'training_scope_guard':'passed'})
            del model
    np.testing.assert_allclose(p,table['probabilities'],rtol=0,atol=1e-14)
    recovered=measure(y,p,selected['threshold'])
    for field in ('f1','fp','fn','tp','tn','rows'):
        assert recovered[field]==selected['metrics'][field]
    train_labels=dict(zip(cache['urls'],cache['labels'].tolist(),strict=True))
    full_context=fit_context(context_keys(cache['records']),cache['labels']) if 'context_config' in selected else None
    del cache,urls,frame,table,p,base,fingerprints
    gc.collect()
    hold=pd.read_csv(out/'holdout_predictions.csv')
    metrics=json.loads((out/'metrics.json').read_text())
    recovered_hold=measure(hold.label.to_numpy(),hold.probability.to_numpy(),selected['threshold'])
    for field in ('f1','fp','fn','tp','tn','rows'):
        assert recovered_hold[field]==metrics['holdout'][field]
    np.testing.assert_array_equal(hold.cache_index.to_numpy(),splits['holdout'])
    np.testing.assert_array_equal(hold.prediction.to_numpy(),hold.probability.to_numpy()>=selected['threshold'])
    cache=joblib.load(ROOT/'data/content_test.joblib')
    urls=joblib.load(ROOT/'data/content_url_test.joblib')
    assert cache['labels'] is None
    assert cache['urls']==urls['urls']==load_test(ROOT/'data/test.csv')
    path=ROOT/'artifacts/v5/ensemble.joblib'
    bundle=joblib.load(path)
    assert bundle['config']==config and bundle['threshold']==selected['threshold']
    assert bundle['training_rows']==bundle['base']['training_rows']==63900
    assert bundle['base']['known_urls']==train_labels
    assert bundle['train_cache_sha256']==digest
    assert 'external_reference' not in bundle and 'external_reference' not in bundle['base']
    if 'context_config' in selected:
        assert bundle['context_config']==selected['context_config']
        assert bundle['context_source_sha256']==sha(ROOT/'phishing_url/soft_context.py')
        assert bundle['context']==full_context
    with (patch('socket.create_connection',side_effect=AssertionError('Offline verification')),
          patch('socket.getaddrinfo',side_effect=AssertionError('Offline verification'))):
        p,hits=predict_v5(bundle,cache,urls['features'])
    replay=out/'reproduced_submission.csv'
    info=write_submission(ROOT/'data/sample.csv',replay,(p>=bundle['threshold']).astype(np.int8))
    assert replay.read_bytes()==(ROOT/'submissions/submission_content_v5.csv').read_bytes()
    result={'oof_memory_direct_replay':'passed','oof_rows':len(y),'recovered_oof':recovered,
            'oof_model_checks':model_checks,'recovered_holdout':recovered_hold,
            'offline_full_inference_byte_identical':True,'test_rows_in_original_order':len(p),
            'external_test_labels_used':False,'new_astra_calls':0,'model_sha256':sha(path),
            'submission':info,'training_rule_matches':int(hits['training_match'].sum()),
            'known_training_urls':int(hits['known_url'].sum())}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
