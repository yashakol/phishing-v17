"""Independent OOF assembly, scope, coefficient and prediction audit; no fits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import hashlib,json,time
import joblib,numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.content_validation import aligned_splits
# Reuse the prior verifier, not the training helper: independently reconstructs
# guards, compares every vocabulary/IDF and computes sigmoid(X @ coef + bias).
from experiments.v8_astra_regularization.verify import audit,sha,check_model

def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def emit(event,**values):print(json.dumps(dict(event=event,**values)),flush=True)

def main():
    sys.addaudithook(audit)
    start=time.perf_counter();s=read('exploratory_oof_selection.json');p=read('exploratory_oof_protocol.json')
    old=read('selection.json');execution=read('exploratory_oof_execution.json')
    assert old['gate_passed'] is False and s['original_gate_passed'] is False and p['original_gate_passed'] is False
    assert s['chosen_c']==32 and s['parameters']==old['parameters'] and s['classifier_parameters']==old['classifier_parameters']
    assert s['threshold'] is None and s['mixture_weights'] is None
    assert s['fit_budget']==dict(additional=3,total_branch=6)
    assert sha(OUT/'selection.json')==s['original_selection_sha256']==p['original_selection_sha256']
    assert sha(OUT/'exploratory_oof_protocol.json')==s['protocol_sha256']
    assert sha(OUT/'exploratory_oof_selection.json')==execution['selection_sha256']
    assert sha(OUT/'development_oof.npz')==execution['oof_sha256']
    for name,h in p['old_artifacts_sha256'].items():assert sha(OUT/name)==h,name
    for name,h in s['sources'].items():assert sha(ROOT/name)==h,name
    digest=sha(ROOT/'data/content_train.joblib');assert digest==s['data_sha256']
    cache=joblib.load(ROOT/'data/content_train.joblib');splits=aligned_splits(ROOT,cache)
    dev=splits['development'];y=cache['labels'][dev];assert len(dev)==50977 and len(set(dev))==50977
    assert not set(dev)&set(splits['holdout']) and not set(dev)&set(splits['added'])
    assert ah(dev)==s['development_indices_sha256']
    folds=np.full(len(dev),-1,dtype=np.int8);coverage=np.zeros(len(dev),dtype=np.int8);assembled=np.full(len(dev),np.nan);checks=[]
    for f,(ti,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,y)):
        tr,va=dev[ti],dev[vi];source=s['fold_sources'][f]
        assert source['fold']==f and source['training_indices_sha256']==ah(tr) and source['validation_indices_sha256']==ah(va)
        assert len(tr)==source['train_rows'] and len(va)==source['validation_rows']
        assert not set(tr)&set(va) and set(tr)|set(va)==set(dev)
        emit('independent_fold_check_start',fold=f)
        result,proba=check_model(f'exploratory_fold_{f}',tr,va,32,cache,digest,s)
        g=read(f'guards/exploratory_fold_{f}.json')
        for key in ('model','model_sha256','guard','guard_sha256','guard_content'):assert g['config']['representation_origin'][key]==source[key]
        assert g['config']['parameters']==s['parameters'] and g['config']['classifier_parameters']==s['classifier_parameters']
        assert g['config']['sources']=={**s['sources'],'experiments/v8_astra_regularization/exploratory_oof_selection.json':sha(OUT/'exploratory_oof_selection.json')}
        rep=read(f'exploratory_fold_{f}_representation.json')
        assert rep['full_training_transforms']==rep['full_validation_transforms']==1
        assert rep['base_unchanged'] and rep['original_base_predictions_unchanged_on_32_real_rows'] and not rep['raw_sparse_matrices_saved']
        with np.load(OUT/f'exploratory_fold_{f}.npz',allow_pickle=False) as a:
            assert set(a.files)=={'train_indices','indices','labels','probabilities'}
            assert all(a[k].ndim==1 for k in a.files)
        checks.append(result);coverage[vi]+=1;folds[vi]=f;assembled[vi]=proba
        emit('independent_fold_check_complete',fold=f,manual_prediction_error=result['manual_logistic_max_abs_error'])
    assert np.all(coverage==1) and ah(folds)==s['folds_sha256']
    assert np.isfinite(assembled).all() and np.all((assembled>=0)&(assembled<=1))
    with np.load(OUT/'development_oof.npz',allow_pickle=False) as a:
        assert set(a.files)=={'indices','labels','probabilities','folds'}
        for key,val in [('indices',dev),('labels',y),('probabilities',assembled),('folds',folds)]:np.testing.assert_array_equal(a[key],val)
    # Exactly six starts and six successful fits across the whole branch.
    events=[]
    for filename in ('run.log','resume.log','exploratory_oof.log'):
        for line in (OUT/filename).read_text().splitlines():
            if line.startswith('{'):events.append(json.loads(line))
    starts=[e['name'] for e in events if e.get('event')=='fit_start']
    ends=[e['name'] for e in events if e.get('event')=='fit_done']
    expected=['inner_c4','inner_c8','inner_c32','exploratory_fold_0','exploratory_fold_1','exploratory_fold_2']
    assert starts==ends==expected,(starts,ends)
    result=dict(independent_implementation=True,no_additional_fits=True,source_and_data_hashes_match=True,
        original_gate_still_false=True,original_artifacts_unchanged=True,development_rows=len(dev),
        original_threefold_alignment_exact=True,coverage_exactly_once=True,all_oof_probabilities_equal_fold_predictions=True,
        pure_text_oof=True,forbidden_rows_excluded_from_fit_and_prediction=True,
        coefficients_and_model_predictions_checked_on_real_validation_rows_per_fold=32,
        full_probability_retransform=False,probability_verification='All 50977 OOF entries match independently aligned fold archives; manual logistic formula and saved estimator replay checked on 32 real validation rows per fold.',
        classifiers_total=6,classifiers_this_followup=3,threshold_or_weight_tuning=False,models=checks,
        oof_sha256=sha(OUT/'development_oof.npz'),selection_sha256=sha(OUT/'exploratory_oof_selection.json'),
        verifier_sha256=sha(__file__),independent_helper_sha256=sha(OUT/'verify.py'),seconds=time.perf_counter()-start)
    (OUT/'exploratory_oof_verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    emit('independent_oof_verified',rows=len(dev),total_fits=6)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
