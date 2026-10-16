"""Recheck second-candidate guards, parameters, split alignment, and metrics."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.content_validation import aligned_splits
from phishing_url.content_checkpoints import data_digest,guard,load_component
from phishing_url.evaluation import measure
from experiments.v7_astra.expanded import core,CONFIG,sources


def main():
    sys.addaudithook(core.audit)
    started=time.perf_counter()
    out=core.OUT
    selected=json.loads((out/'selection.json').read_text())
    assert selected['parameters']==CONFIG and selected['sources']==sources()
    digest=data_digest(ROOT)
    assert selected['data_sha256']==digest
    cache=joblib.load(ROOT/'data/content_train.joblib')
    splits=aligned_splits(ROOT,cache)
    va=splits['inner_validation']
    y=cache['labels'][va]
    old=core.aligned_prediction('experiments/html_astra/inner_predictions.npz',va,y)
    cat=core.aligned_prediction('experiments/v5/cat_inner_1.npz',va,y)
    new=core.aligned_prediction('experiments/v7_astra/expanded/inner_full_vocabulary.npz',va,y)
    result=json.loads((out/'inner_results.json').read_text())
    assert core.metrics(y,.5*old+.5*cat)==result['baseline']
    assert core.metrics(y,new)==result['new_text']
    assert core.metrics(y,.5*new+.5*cat)==result['replacement']
    checks=[('inner_full_vocabulary',splits['inner_train'],va)]
    if (out/'development_oof.npz').exists():
        indices=splits['development']
        labels=cache['labels'][indices]
        fold_ids=np.full(len(indices),-1,dtype=np.int8)
        counts=np.zeros(len(indices),dtype=np.int8)
        for f,(tr,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(indices,labels)):
            fold_ids[valid]=f
            counts[valid]+=1
            checks.append((f'fold_{f}',indices[tr],indices[valid]))
        assert np.all(counts==1)
        with np.load(out/'development_oof.npz') as a:
            np.testing.assert_array_equal(a['indices'],indices)
            np.testing.assert_array_equal(a['labels'],labels)
            np.testing.assert_array_equal(a['folds'],fold_ids)
            report=json.loads((out/'oof_results.json').read_text())
            assert measure(labels,a['replacement'],selected['threshold'])==report['replacement']
            assert measure(labels,a['probabilities'],selected['standalone_threshold'])==report['standalone']
    for name,tr,valid in checks:
        scope=json.loads((out/f'guards/{name}.json').read_text())
        fresh=guard(cache,tr,dict(parameters=CONFIG,sources=sources()),digest)
        for k in fresh:
            assert scope[k]==fresh[k]
        with np.load(out/f'{name}.npz') as a:
            np.testing.assert_array_equal(a['train_indices'],tr)
            np.testing.assert_array_equal(a['indices'],valid)
            np.testing.assert_array_equal(a['labels'],cache['labels'][valid])
        model=load_component(out/f'models/{name}.joblib',scope)
        assert model.get_params()==CONFIG
        assert model.vocabulary_fit_rows_==model.training_rows_==len(tr)
        fit=json.loads((out/f'{name}_fit.json').read_text())
        assert fit['model_sha256']==core.sha(out/f'models/{name}.joblib')
        assert fit['serialization_verified_real_validation_rows']>=32
        assert fit['serialization_max_abs_error']==0
        del model
    verified=dict(models_verified=len(checks),source_and_data_guards_match=True,
                  parameters_match=True,full_training_vocabulary_rows_verified=True,
                  metrics_recomputed=True,indices_labels_match=True,
                  oof_verified=len(checks)==4,seconds=time.perf_counter()-started)
    core.write_json('verification.json',verified)
    print(json.dumps(verified))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
