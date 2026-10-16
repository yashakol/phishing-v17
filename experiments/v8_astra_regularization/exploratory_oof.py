"""Explicit follow-up after failed original gate: exactly three C32 fold fits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'): os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import copy,gc,hashlib,json,time
import joblib,numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits,threadpool_info
from phishing_url.content_validation import aligned_splits,records_at
from phishing_url.content_checkpoints import data_digest
from experiments.v8_astra_regularization.resume import (audit,READS,sha,write,emit,scope,checked_base,refit,semantic_hash)

def array_sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def main():
    sys.addaudithook(audit)
    assert not (OUT/'exploratory_oof_protocol.json').exists(),'No silent refit/restart'
    assert not (OUT/'development_oof.npz').exists()
    old=json.loads((OUT/'selection.json').read_text())
    assert old['gate_passed'] is False and old['chosen_c']==32
    assert old['winner']['standalone']['errors']==66 and old['baseline']['standalone']['errors']==68
    assert old['winner']['mix']['errors']==old['baseline']['mix']['errors']==51
    original_files=json.loads((OUT/'manifest.json').read_text())['files_sha256']
    for p,h in original_files.items():assert sha(OUT/p)==h,p
    original_files['manifest.json']=sha(OUT/'manifest.json')
    protocol=dict(reason='standalone two-error gain despite original half-mixture gate failed',
        authorization='Explicit user follow-up to explore C32 text in full v7 integration; do not relabel original gate.',
        original_gate_passed=False,original_selection_sha256=sha(OUT/'selection.json'),chosen_c=32,
        parameters=old['parameters'],classifier_parameters=old['classifier_parameters'],
        additional_classifier_fits=3,total_branch_classifier_fits=6,other_fits=0,
        splitter=dict(n_splits=3,shuffle=True,random_state=2026),development_rows=50977,
        representations='Each original v7_astra fold train-only vocabulary/IDF is frozen and reused only after exact source/data/ordered-training guard checks.',
        matrices='Transform each fold TRAIN and VALIDATION once on a deep copy; keep full sparse matrices in memory only. Additional 32-row transforms are solely serialization/semantic checks.',
        training='Sequential LogisticRegression(C=32, solver=liblinear, max_iter=500, random_state=20261006).fit(X_train, y_train), threadpool_limits(2). Never fit on validation labels.',
        threshold_weight_policy='No thresholds, mixture weights or new hyperparameters selected here. Parent will select thresholds/weights in the full v7 mixture later.',
        numerical_limitation='Original fold training used float32 fit_transform; this experiment uses transform with frozen vocabulary/IDF. Scratch-fit numerical equivalence is not claimed.',
        evaluation_limitation='Post-inner exploratory follow-up; development includes rows involved in inner selection. Not nested model-selection evaluation.',
        old_artifacts_sha256=original_files,
        forbidden='External data, network, submissions, holdout/test prediction or label files, live URL visits. Only development rows enter fitting/scoring.')
    write('exploratory_oof_protocol.json',protocol)
    sources={**old['sources'],'experiments/v8_astra_regularization/exploratory_oof.py':sha(__file__),
        'experiments/v8_astra_regularization/exploratory_oof_protocol.json':sha(OUT/'exploratory_oof_protocol.json'),
        'experiments/v8_astra_regularization/selection.json':sha(OUT/'selection.json')}
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    digest=data_digest(ROOT);assert digest==old['data_sha256']
    cache=joblib.load(ROOT/'data/content_train.joblib');assert len(cache['records'])==63900
    splits=aligned_splits(ROOT,cache);dev=splits['development'];y=cache['labels'][dev]
    assert len(dev)==50977 and len(np.unique(dev))==50977
    assert not set(dev)&set(splits['holdout']) and not set(dev)&set(splits['added'])
    original_folds=list(StratifiedKFold(**protocol['splitter']).split(dev,y))
    folds=np.full(len(dev),-1,dtype=np.int8);origins=[]
    # Freeze all fold scopes and source provenance before ANY classifier fit.
    for f,(ti,vi) in enumerate(original_folds):
        tr,va=dev[ti],dev[vi];assert not set(tr)&set(va) and set(tr)|set(va)==set(dev)
        gp=ROOT/f'experiments/v7_astra/guards/fold_{f}.json';mp=ROOT/f'experiments/v7_astra/models/fold_{f}.joblib'
        g=json.loads(gp.read_text());assert g==scope(cache,tr,g['config'],digest)
        assert g['config']['parameters']==dict(old['parameters'],c=16.)
        for path,h in g['config']['sources'].items():assert sha(ROOT/path)==h,path
        origins.append(dict(fold=f,train_rows=len(tr),validation_rows=len(va),training_indices_sha256=array_sha(tr),validation_indices_sha256=array_sha(va),model=str(mp.relative_to(ROOT)),model_sha256=sha(mp),guard=str(gp.relative_to(ROOT)),guard_sha256=sha(gp),guard_content=g))
        folds[vi]=f
    selection=dict(exploratory=True,reason=protocol['reason'],original_gate_passed=False,
        original_selection_path='experiments/v8_astra_regularization/selection.json',original_selection_sha256=sha(OUT/'selection.json'),
        protocol_path='experiments/v8_astra_regularization/exploratory_oof_protocol.json',protocol_sha256=sha(OUT/'exploratory_oof_protocol.json'),
        parameters=old['parameters'],parameters_path='experiments/v8_astra_regularization/exploratory_oof_selection.json',parameters_json_pointer='/parameters',
        classifier_parameters=old['classifier_parameters'],chosen_c=32,data_sha256=digest,sources=sources,
        splitter=protocol['splitter'],development_indices_sha256=array_sha(dev),folds_sha256=array_sha(folds),fold_sources=origins,
        oof_path='experiments/v8_astra_regularization/development_oof.npz',oof_fields=['indices','labels','probabilities','folds'],
        fit_budget=dict(additional=3,total_branch=6),selection_scope='C fixed before OOF; no OOF model/threshold/weight search',
        threshold=None,mixture_weights=None,downstream_policy=protocol['threshold_weight_policy'])
    write('exploratory_oof_selection.json',selection)
    selection_hash=sha(OUT/'exploratory_oof_selection.json')
    fit_sources={**sources,'experiments/v8_astra_regularization/exploratory_oof_selection.json':selection_hash}
    probabilities=np.full(len(dev),np.nan);coverage=np.zeros(len(dev),dtype=np.int8)
    started=time.perf_counter()
    for f,(ti,vi) in enumerate(original_folds):
        tr,va=dev[ti],dev[vi]
        base,origin=checked_base(cache,tr,f'fold_{f}',digest)
        for k in ('model','model_sha256','guard','guard_sha256','guard_content'):assert origin[k]==origins[f][k]
        fingerprint=semantic_hash(base)
        emit('exploratory_fold_transform_start',fold=f,training_rows=len(tr),validation_rows=len(va))
        transformer=copy.deepcopy(base)
        X=transformer._matrix(records_at(cache,tr),fit=False)
        emit('exploratory_train_transform_done',fold=f,shape=list(X.shape),nnz=X.nnz)
        V=transformer._matrix(records_at(cache,va),fit=False)
        del transformer
        assert X.shape==(len(tr),base.n_features_) and V.shape==(len(va),base.n_features_)
        baseline32=base.classifier_.predict_proba(V[:32])[:,1].copy()
        emit('exploratory_validation_transform_done',fold=f,shape=list(V.shape),nnz=V.nnz)
        p=refit(base,origin,X,V,cache,tr,va,32,f'exploratory_fold_{f}',digest,fit_sources)
        assert semantic_hash(base)==fingerprint
        np.testing.assert_array_equal(baseline32,base.classifier_.predict_proba(V[:32])[:,1])
        np.testing.assert_array_equal(baseline32,base.predict_proba(records_at(cache,va[:32])))
        assert semantic_hash(base)==fingerprint
        assert sha(ROOT/origin['model'])==origin['model_sha256'] and sha(ROOT/origin['guard'])==origin['guard_sha256']
        write(f'exploratory_fold_{f}_representation.json',dict(fold=f,training_shape=list(X.shape),validation_shape=list(V.shape),training_nnz=X.nnz,validation_nnz=V.nnz,dtype=str(X.dtype),full_training_transforms=1,full_validation_transforms=1,base_semantics_sha=fingerprint,base_unchanged=True,original_base_predictions_unchanged_on_32_real_rows=True,only_semantic_hash_exclusion='sklearn vectorizer _stop_words_id process-local cache',raw_sparse_matrices_saved=False))
        probabilities[vi]=p;coverage[vi]+=1
        del X,V,base;gc.collect()
        emit('exploratory_fold_complete',fold=f,rows=len(vi))
    assert np.all(coverage==1) and np.all(folds>=0) and np.isfinite(probabilities).all()
    assert ((probabilities>=0)&(probabilities<=1)).all()
    assert sha(OUT/'exploratory_oof_selection.json')==selection_hash
    for p,h in fit_sources.items():assert sha(ROOT/p)==h,p
    assert data_digest(ROOT)==digest
    for p,h in original_files.items():assert sha(OUT/p)==h,p
    for origin in origins:
        assert sha(ROOT/origin['model'])==origin['model_sha256'] and sha(ROOT/origin['guard'])==origin['guard_sha256']
    np.savez_compressed(OUT/'development_oof.npz',indices=dev,labels=y,probabilities=probabilities,folds=folds)
    write('exploratory_oof_execution.json',dict(rows=len(dev),new_classifier_fits=3,total_branch_classifier_fits=6,threads=2,threadpools=threadpool_info(),coverage_exactly_once=True,original_artifacts_unchanged=True,original_gate_still_false=True,base_models_semantically_and_on_disk_unchanged=True,raw_sparse_matrices_saved=False,threshold_or_weight_tuning=False,selection_sha256=selection_hash,oof_sha256=sha(OUT/'development_oof.npz'),seconds=time.perf_counter()-started,project_files_read=sorted(READS)))
    emit('exploratory_oof_complete',rows=len(dev),oof='experiments/v8_astra_regularization/development_oof.npz',original_gate_passed=False)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
