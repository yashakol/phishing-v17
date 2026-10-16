"""Independent artifact audit; no model fitting or selection changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'): os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import hashlib,json,gc
import joblib,numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.content_validation import aligned_splits,records_at

def audit(event,args):
    if event.startswith(('socket.connect','socket.getaddrinfo','socket.bind')): raise PermissionError('Offline')
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p=Path(os.fsdecode(args[0])).resolve(); mode,flags=args[1:3]
    write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
    if write and not p.is_relative_to(OUT): raise PermissionError(str(p))
    if not p.is_relative_to(ROOT): return
    rel=p.relative_to(ROOT).as_posix()
    if rel.startswith('.venv/'): return
    if rel.startswith(('external/','reports/v2/','submissions/')) or any(t in p.name.lower() for t in ('test','submission')) or 'holdout' in rel.lower(): raise PermissionError(rel)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(name):return json.loads((OUT/name).read_text())

def check_scope(g,tr,cache,digest):
    assert g['training_rows']==len(tr) and g['data_sha256']==digest
    assert g['ordered_training_indices_sha256']==hashlib.sha256(tr.tobytes()).hexdigest()
    h=hashlib.sha256()
    for i in tr:
        h.update(cache['urls'][int(i)].encode());h.update(b'\0');h.update(bytes([int(cache['labels'][int(i)])]))
    assert g['ordered_training_sha256']==h.hexdigest()
    for p,v in g['config']['sources'].items():assert sha(ROOT/p)==v,p

def optimal(y,p):
    # Independent grouped cumulative confusion counts, without project threshold helper.
    order=np.argsort(-p,kind='stable'); sortedp=p[order]; yy=y[order]
    ends=np.r_[np.flatnonzero(sortedp[:-1]!=sortedp[1:]),len(y)-1]
    tp=np.cumsum(yy)[ends]; fp=ends+1-tp; fn=y.sum()-tp
    scores=2*tp/(2*tp+fp+fn)
    tied=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))
    idx=tied[np.argmin(abs(sortedp[ends[tied]]-.5))]
    return dict(f1=float(scores[idx]),errors=int(fp[idx]+fn[idx]),threshold=float(sortedp[ends[idx]]))

def check_model(name,tr,va,c,cache,digest,selection):
    g=read(f'guards/{name}.json'); check_scope(g,tr,cache,digest)
    assert not set(tr)&set(va)
    packet=joblib.load(OUT/f'models/{name}.joblib'); assert packet['guard']==g
    model=packet['model']; origin=g['config']['representation_origin']
    assert sha(ROOT/origin['model'])==origin['model_sha256'] and sha(ROOT/origin['guard'])==origin['guard_sha256']
    oldguard=json.loads((ROOT/origin['guard']).read_text()); assert oldguard==origin['guard_content']
    check_scope(oldguard,tr,cache,digest)
    original=joblib.load(ROOT/origin['model']); assert original['guard']==oldguard
    base=original['model']
    for key, v in model.vectorizers_.items():
        original_v=base.vectorizers_[key]
        assert v.get_params()==original_v.get_params() and v.vocabulary_==original_v.vocabulary_
        np.testing.assert_array_equal(v.idf_,original_v.idf_)
        # Only analyzer's process-id cache may differ beyond fitted semantics.
        vd=dict(v.__dict__); bd=dict(original_v.__dict__)
        vd.pop('_stop_words_id',None); bd.pop('_stop_words_id',None)
        assert joblib.hash(vd)==joblib.hash(bd)
    assert model.training_rows_==model.vocabulary_fit_rows_==len(tr)
    assert model.c==model.classifier_.C==c and base.c==base.classifier_.C==16
    assert model.classifier_.get_params()==g['config']['classifier_parameters']
    assert model.get_params()==g['config']['parameters']==clone(model).get_params()
    assert not hasattr(clone(model),'classifier_')
    assert model.classifier_.solver=='liblinear' and model.classifier_.random_state==20261006 and model.classifier_.max_iter==500
    assert model.word_features==140000 and model.char_features==100000 and model.markup_features==70000 and model.url_features==90000
    assert model.ratio_ is None
    coef=model.classifier_.coef_; bias=model.classifier_.intercept_
    assert coef.shape==(1,sum(len(v.vocabulary_) for v in model.vectorizers_.values()))
    assert np.isfinite(coef).all() and np.isfinite(bias).all() and np.count_nonzero(coef)>0
    assert not np.array_equal(coef,base.classifier_.coef_)
    meta=read(f'{name}_fit.json')
    assert meta['model_sha256']==sha(OUT/f'models/{name}.joblib')
    assert meta['coef_sha256']==hashlib.sha256(coef.tobytes()).hexdigest()
    assert meta['serialization_rows']==32 and meta['serialization_max_abs_difference']==0
    with np.load(OUT/f'{name}.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['train_indices'],tr);np.testing.assert_array_equal(a['indices'],va)
        np.testing.assert_array_equal(a['labels'],cache['labels'][va]);p=a['probabilities'].copy()
    assert p.shape==(len(va),) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    records=records_at(cache,va[:32]); x=model._matrix(records,fit=False)
    manual=expit(np.asarray(x@coef.T).ravel()+bias[0])
    np.testing.assert_allclose(manual,p[:32],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(model.predict_proba(records),p[:32])
    result=dict(name=name,c=c,rows=len(tr),features=coef.shape[1],source_and_training_guards_match=True,vectorizers_identical_to_guarded_original=True,coefficients_changed_and_finite=True,clone_has_correct_c_and_no_fitted_state=True,real_validation_rows=32,manual_logistic_max_abs_error=float(np.max(abs(manual-p[:32]))))
    del model,base,packet,original,x;gc.collect()
    return result,p

def main():
    sys.addaudithook(audit)
    s=read('selection.json'); prereg=read('preregistered.json')
    assert s['protocol']==prereg and prereg['grid']==[4,8,32]
    digest=sha(ROOT/'data/content_train.joblib');assert digest==s['data_sha256']
    for p,h in s['sources'].items():assert sha(ROOT/p)==h
    for p,h in s['inputs'].items():assert sha(ROOT/p)==h
    cache=joblib.load(ROOT/'data/content_train.joblib');splits=aligned_splits(ROOT,cache)
    tr,va=splits['inner_train'],splits['inner_validation'];dev=splits['development'];y=cache['labels'][va]
    assert set(tr)|set(va)<=set(dev) and not set(dev)&set(splits['holdout']) and not set(dev)&set(splits['added'])
    with np.load(ROOT/'experiments/v7_astra/inner_full_vocabulary.npz') as a:
        np.testing.assert_array_equal(a['indices'],va);np.testing.assert_array_equal(a['labels'],y); b=a['probabilities'].copy()
    with np.load(ROOT/'experiments/v5/cat_inner_1.npz') as a:
        np.testing.assert_array_equal(a['indices'],va);np.testing.assert_array_equal(a['labels'],y); cat=a['probabilities'].copy()
    inner=read('inner_results.json');assert optimal(y,.5*b+.5*cat)['errors']==51
    checks=[]
    for row in inner['candidates']:
        c=row['c'];check,p=check_model(f'inner_c{c}',tr,va,c,cache,digest,s);checks.append(check)
        for key,proba in [('standalone',p),('mix',.5*p+.5*cat)]:
            independently=optimal(y,proba)
            for k,v in independently.items():assert np.isclose(row[key][k],v,rtol=0,atol=1e-14),(c,key,k)
    winner=min(inner['candidates'],key=lambda r:(r['mix']['errors'],-r['mix']['f1'],r['c']))
    assert winner==s['winner'] and winner['c']==s['chosen_c']
    assert s['gate_passed']==(winner['mix']['errors']<=49)
    result=dict(independent_implementation=True,no_additional_fits=True,source_and_data_hashes_match=True,inner_alignment_exact=True,inner_thresholds_independently_recomputed=True,selection_and_gate_reproduced=True,models=checks)
    if s['gate_passed']:
        labels=cache['labels'][dev];coverage=np.zeros(len(dev),dtype=int);folds=np.full(len(dev),-1,dtype=np.int8);proba=np.full(len(dev),np.nan)
        for f,(ti,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,labels)):
            check,p=check_model(f'fold_{f}',dev[ti],dev[vi],s['chosen_c'],cache,digest,s);checks.append(check)
            coverage[vi]+=1;folds[vi]=f;proba[vi]=p
        assert np.all(coverage==1)
        with np.load(OUT/'development_oof.npz') as a:
            assert set(a.files)=={'indices','labels','probabilities','folds'}
            for key,value in [('indices',dev),('labels',labels),('probabilities',proba),('folds',folds)]:np.testing.assert_array_equal(a[key],value)
        result['oof']=dict(rows=len(dev),original_stratified_threefold_exact=True,coverage_exactly_once=True,text_only=True)
    else:
        assert not (OUT/'development_oof.npz').exists()
        assert not list((OUT/'models').glob('fold_*.joblib'))
        result['oof']=dict(generated=False,reason='meaningful advance gate failed')
    # Recount word IDF on all INNER TRAIN rows, independently of recorded row counts.
    base=joblib.load(ROOT/'experiments/v7_astra/models/inner_full_vocabulary.joblib')['model']
    v=base.vectorizers_['word'];terms=sorted(v.vocabulary_);picked=[terms[i] for i in np.linspace(0,len(terms)-1,16,dtype=int)];wanted=set(picked);counts={t:0 for t in picked};analyze=v.build_analyzer()
    for start in range(0,len(tr),512):
        for doc in base._documents(records_at(cache,tr[start:start+512]))['word']:
            for t in wanted.intersection(analyze(doc)):counts[t]+=1
    expected=np.log((1+len(tr))/(1+np.array([counts[t] for t in picked])))+1
    actual=np.array([v.idf_[v.vocabulary_[t]] for t in picked]);np.testing.assert_allclose(expected,actual,rtol=1e-6,atol=1e-6)
    result['inner_idf']=dict(terms=16,training_documents=len(tr),max_abs_error=float(np.max(abs(expected-actual))))
    result['verifier_sha256']=sha(__file__)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
