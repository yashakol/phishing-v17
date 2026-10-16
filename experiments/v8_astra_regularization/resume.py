"""Bounded C-only refits on guarded frozen v7 representations; offline."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import copy, gc, hashlib, json, time, warnings
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.content_checkpoints import guard, load_component, save_component, data_digest
from phishing_url.evaluation import best_threshold, measure
from phishing_url.v7_text import FullVocabularyTextModel
READS=set()
CONFIG=FullVocabularyTextModel().get_params()

def audit(event,args):
    if event.startswith(('socket.connect','socket.getaddrinfo','socket.bind')):
        raise PermissionError('Offline experiment')
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p=Path(os.fsdecode(args[0])).resolve()
    mode,flags=args[1:3]
    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
    if writing and not p.is_relative_to(OUT): raise PermissionError(f'Write outside scope: {p}')
    if not p.is_relative_to(ROOT): return
    rel=p.relative_to(ROOT).as_posix()
    if rel.startswith('.venv/'): return
    if (rel.startswith(('external/','reports/v2/','submissions/')) or 'submission' in p.name.lower() or 'holdout' in rel.lower() or 'test' in p.name.lower() or (rel.startswith('reports/v7/') and rel!='reports/v7/selection.json')):
        raise PermissionError(f'Forbidden input: {rel}')
    if not writing: READS.add(rel)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(name,value):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def emit(event,**kw): print(json.dumps(dict(event=event,**kw)),flush=True)

def scope(cache,tr,config,digest):
    s=guard(cache,tr,config,digest)
    s['ordered_training_indices_sha256']=hashlib.sha256(tr.tobytes()).hexdigest()
    return s

def metric(y,p):
    cut,_=best_threshold(y,p)
    m=measure(y,p,cut); m['errors']=m['fp']+m['fn']
    return m

def prediction(path,va,y,tr=None):
    with np.load(ROOT/path,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['indices'],va)
        np.testing.assert_array_equal(a['labels'],y)
        if tr is not None: np.testing.assert_array_equal(a['train_indices'],tr)
        p=a['probabilities'].copy()
    assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p

def checked_base(cache,tr,name,digest):
    gp=ROOT/f'experiments/v7_astra/guards/{name}.json'
    mp=ROOT/f'experiments/v7_astra/models/{name}.joblib'
    g=json.loads(gp.read_text())
    assert g==scope(cache,tr,g['config'],digest)
    assert g['config']['parameters']==CONFIG
    for p,h in g['config']['sources'].items(): assert sha(ROOT/p)==h, p
    model=load_component(mp,g)
    assert model.get_params()==CONFIG and clone(model).get_params()==CONFIG
    assert model.training_rows_==model.vocabulary_fit_rows_==len(tr)
    assert model.ratio_ is None and not model.nb
    assert model.classifier_.C==16 and model.classifier_.solver=='liblinear'
    assert model.classifier_.max_iter==500 and model.classifier_.random_state==20261006
    provenance={'model':str(mp.relative_to(ROOT)),'model_sha256':sha(mp),'guard':str(gp.relative_to(ROOT)),'guard_sha256':sha(gp),'guard_content':g}
    return model,provenance

def refit(base,origin,X,V,cache,tr,va,c,name,digest,sources):
    start=time.perf_counter()
    model=copy.deepcopy(base)
    model.set_params(c=float(c))
    model.classifier_=LogisticRegression(C=float(c),solver='liblinear',max_iter=500,random_state=20261006)
    emit('fit_start',name=name,c=c,rows=len(tr))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=2): model.classifier_.fit(X,cache['labels'][tr])
    model.classes_=model.classifier_.classes_
    model.fit_seconds_=model.fit_seconds=time.perf_counter()-start
    model.regularization_refit_={'method':'frozen train-fitted v7 vectorizers; transform then classifier.fit','origin':origin,'c':c,'training_rows':len(tr)}
    assert model.c==model.classifier_.C==clone(model).c
    assert clone(model).get_params()==model.get_params()
    assert not hasattr(clone(model),'classifier_')
    assert np.isfinite(model.classifier_.coef_).all()
    assert np.isfinite(model.classifier_.intercept_).all()
    assert model.classifier_.coef_.shape==(1,X.shape[1])
    assert not np.array_equal(model.classifier_.coef_,base.classifier_.coef_)
    assert base.c==base.classifier_.C==16
    p=model.classifier_.predict_proba(V)[:,1]
    g=scope(cache,tr,dict(parameters=model.get_params(),classifier_parameters=model.classifier_.get_params(),sources=sources,representation_origin=origin),digest)
    path=OUT/f'models/{name}.joblib'
    save_component(path,model,g)
    restored=load_component(path,g)
    sample=records_at(cache,va[:32])
    before=model.predict_proba(sample); after=restored.predict_proba(sample)
    np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(p[:32],after)
    try: load_component(path,dict(g,training_rows=-1))
    except ValueError: pass
    else: raise AssertionError('Bad guard accepted')
    report=dict(c=c,fit_seconds=model.fit_seconds_,iterations=model.classifier_.n_iter_.tolist(),warnings=[str(w.message) for w in caught],model_sha256=sha(path),serialization_rows=32,serialization_max_abs_difference=0.,sparse_prediction_matches_estimator=True,coef_sha256=hashlib.sha256(model.classifier_.coef_.tobytes()).hexdigest(),clone_parameters_match=True,features=X.shape[1],training_rows=len(tr),validation_rows=len(va),threadpools=threadpool_info())
    assert all(t['num_threads']<=2 for t in report['threadpools'])
    write(f'guards/{name}.json',g); write(f'{name}_fit.json',report)
    np.savez_compressed(OUT/f'{name}.npz',train_indices=tr,indices=va,labels=cache['labels'][va],probabilities=p)
    emit('fit_done',name=name,**report)
    del model,restored; gc.collect()
    return p

def semantic_hash(model):
    # sklearn stores id(stop_words) as a process-local analyzer check cache.
    # Exclude only this runtime cache from semantic identity checks.
    clean=copy.deepcopy(model)
    for vectorizer in clean.vectorizers_.values():
        vectorizer.__dict__.pop('_stop_words_id',None)
    return joblib.hash(clean)


def main():
    sys.addaudithook(audit)
    assert (OUT/'preregistered.json').exists() and not (OUT/'selection.json').exists(), 'Resume incomplete run only'
    # Freeze all choices before reading predictions or fitting.
    protocol=dict(grid=[4,8,32],baseline_c=16,blend=dict(text=.5,cat=.5),gate='best candidate mix <=49 errors and at least 2 fewer than c16 mix',selection='min tuned mixture errors, then max tuned mixture F1, then lower C; exactly one winner before OOF',threshold='existing best_threshold: max F1; ties nearest 0.5',parameters=CONFIG,splitter=dict(n_splits=3,shuffle=True,random_state=2026),budget='3 sequential inner classifier fits; optional 3 sequential fold classifier fits',representation='reuse original guarded train-only vocabulary/IDF; transform inner training and validation once each; no vectorizer fits',numerical_limitation='Original baseline training used float32 fit_transform. C refits use float32 transform with frozen vocabulary/IDF. These paths can differ numerically; bitwise scratch-refit equivalence is not claimed, and no extra c16 fit is performed.',scratch_retraining='FullVocabularyTextModel(**selected parameters).fit learns fresh training-only vocabulary/IDF, uses fit_transform and may give numerically different coefficients from this frozen-representation ablation.')
    assert json.loads((OUT/'preregistered.json').read_text())==protocol
    recovery=dict(reason='Non-semantic sklearn _stop_words_id process cache changed during transform; original vocabulary and IDF stayed identical.',completed_fit_reused='inner_c4',extra_classifier_fits=0,full_training_transform_calls_total=2,full_validation_transform_calls_total=2,raw_matrices_lost_on_initial_assertion=True,initial_script_sha256=sha(OUT/'run.py'),initial_log_sha256=sha(OUT/'run.log'),diagnosis_sha256=sha(OUT/'recovery_diagnosis.json'))
    write('recovery.json',recovery)
    digest=data_digest(ROOT)
    cache=joblib.load(ROOT/'data/content_train.joblib')
    assert len(cache['records'])==63900
    splits=aligned_splits(ROOT,cache)
    assert {k:len(splits[k]) for k in ('development','inner_train','inner_validation','holdout','added')}==dict(development=50977,inner_train=34663,inner_validation=6118,holdout=12745,added=178)
    tr,va=splits['inner_train'],splits['inner_validation']; dev=splits['development']
    assert not set(tr)&set(va) and set(tr)|set(va)<=set(dev)
    # Only split metadata for excluded rows is touched; no excluded-row fitting/scoring.
    base,origin=checked_base(cache,tr,'inner_full_vocabulary',digest)
    sources={**origin['guard_content']['config']['sources'],str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/v8_astra_regularization/preregistered.json':sha(OUT/'preregistered.json'),'experiments/v8_astra_regularization/run.py':sha(OUT/'run.py'),'experiments/v8_astra_regularization/recovery.json':sha(OUT/'recovery.json')}
    inputs={p:sha(ROOT/p) for p in ('experiments/v7_astra/inner_full_vocabulary.npz','experiments/v5/cat_inner_1.npz')}
    y=cache['labels'][va]
    b=prediction('experiments/v7_astra/inner_full_vocabulary.npz',va,y,tr)
    cat=prediction('experiments/v5/cat_inner_1.npz',va,y)
    baseline=dict(standalone=metric(y,b),mix=metric(y,.5*b+.5*cat))
    assert baseline['mix']['errors']==51,baseline
    emit('baseline',**baseline)
    emit('transform_start',training_rows=len(tr),validation_rows=len(va))
    base_fingerprint=semantic_hash(base)
    transformer=copy.deepcopy(base)
    X=transformer._matrix(records_at(cache,tr),fit=False)
    V=transformer._matrix(records_at(cache,va),fit=False)
    del transformer
    np.testing.assert_array_equal(base.classifier_.predict_proba(V)[:,1],b)
    assert X.shape[0]==len(tr) and V.shape[0]==len(va) and X.shape[1]==V.shape[1]==base.n_features_
    write('representations.json',dict(train_shape=list(X.shape),validation_shape=list(V.shape),train_nnz=X.nnz,validation_nnz=V.nnz,dtype=str(X.dtype),base_model_hash=base_fingerprint,origin=origin,training_transform_calls=2,validation_transform_calls=2,recovery=recovery,baseline_probabilities_reproduced_exactly=True))
    emit('transform_done',train_shape=list(X.shape),validation_shape=list(V.shape))
    results=[]
    for c in protocol['grid']:
        if c==4:
            prior_scope=json.loads((OUT/'guards/inner_c4.json').read_text())
            assert prior_scope==scope(cache,tr,prior_scope['config'],digest)
            for path,h in prior_scope['config']['sources'].items(): assert sha(ROOT/path)==h
            prior=load_component(OUT/'models/inner_c4.joblib',prior_scope)
            assert prior.c==prior.classifier_.C==4
            assert sha(OUT/'models/inner_c4.joblib')==json.loads((OUT/'inner_c4_fit.json').read_text())['model_sha256']
            p=prediction('experiments/v8_astra_regularization/inner_c4.npz',va,y,tr)
            np.testing.assert_array_equal(prior.classifier_.predict_proba(V)[:,1],p)
            del prior
            emit('completed_fit_reused',c=4)
        else:
            p=refit(base,origin,X,V,cache,tr,va,c,f'inner_c{c}',digest,sources)
        result=dict(c=c,standalone=metric(y,p),mix=metric(y,.5*p+.5*cat))
        results.append(result); write('inner_results.json',dict(baseline=baseline,candidates=results))
        emit('inner_result',**result)
        assert semantic_hash(base)==base_fingerprint
    winner=min(results,key=lambda r:(r['mix']['errors'],-r['mix']['f1'],r['c']))
    passed=winner['mix']['errors']<=49 and baseline['mix']['errors']-winner['mix']['errors']>=2
    selection=dict(chosen_c=winner['c'],parameters=dict(CONFIG,c=float(winner['c'])),classifier_parameters=LogisticRegression(C=float(winner['c']),solver='liblinear',max_iter=500,random_state=20261006).get_params(),gate_passed=passed,baseline=baseline,winner=winner,threshold=winner['mix']['threshold'],standalone_threshold=winner['standalone']['threshold'],sources=sources,inputs=inputs,data_sha256=digest,representation_origin=origin,protocol=protocol,selection_scope='inner only; fixed before any OOF',recovery=recovery,oof_path='experiments/v8_astra_regularization/development_oof.npz' if passed else None,status='selected for OOF' if passed else 'negative inner result; stopped without OOF')
    write('selection.json',selection)
    emit('selection',chosen_c=winner['c'],gate_passed=passed,errors=winner['mix']['errors'])
    del X,V,base; gc.collect()
    if passed:
        labels=cache['labels'][dev]
        probabilities=np.full(len(dev),np.nan); folds=np.full(len(dev),-1,dtype=np.int8); coverage=np.zeros(len(dev),dtype=np.int8)
        for f,(ti,vi) in enumerate(StratifiedKFold(**protocol['splitter']).split(dev,labels)):
            tr,va=dev[ti],dev[vi]
            assert not set(tr)&set(va) and set(tr)|set(va)==set(dev)
            base,origin=checked_base(cache,tr,f'fold_{f}',digest)
            for p,h in origin['guard_content']['config']['sources'].items(): assert sha(ROOT/p)==h
            fingerprint=semantic_hash(base)
            emit('fold_transform_start',fold=f)
            transformer=copy.deepcopy(base)
            X=transformer._matrix(records_at(cache,tr),fit=False); V=transformer._matrix(records_at(cache,va),fit=False)
            del transformer
            p=refit(base,origin,X,V,cache,tr,va,winner['c'],f'fold_{f}',digest,sources)
            assert fingerprint==semantic_hash(base)
            probabilities[vi]=p; folds[vi]=f; coverage[vi]+=1
            del X,V,base; gc.collect()
        assert np.all(coverage==1) and np.isfinite(probabilities).all()
        np.savez_compressed(OUT/'development_oof.npz',indices=dev,labels=labels,probabilities=probabilities,folds=folds)
        write('oof_results.json',dict(rows=len(dev),standalone_at_frozen_inner_threshold=measure(labels,probabilities,selection['standalone_threshold']),standalone_at_05=measure(labels,probabilities),oof_sha256=sha(OUT/'development_oof.npz'),coverage_exactly_once=True,selection_sha256=sha(OUT/'selection.json'),limitation='Inner tuning overlaps development; OOF is not nested evaluation. Text-only probabilities; no OOF threshold or weight tuning.'))
    assert data_digest(ROOT)==digest
    for p,h in sources.items(): assert sha(ROOT/p)==h
    for p,h in inputs.items(): assert sha(ROOT/p)==h
    assert sha(ROOT/selection['representation_origin']['model'])==selection['representation_origin']['model_sha256']
    write('execution.json',dict(project_files_read=sorted(READS),threadpools=threadpool_info(),original_sources_and_inputs_unchanged=True,base_models_unmodified=True,raw_sparse_matrices_saved=False,gate_passed=passed))
    emit('complete',gate_passed=passed)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
