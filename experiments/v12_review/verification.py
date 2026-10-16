"""Bounded offline review. No fit, threshold search, predictions, or large caches.
Run from repo: .venv/bin/python -B experiments/v12_review/verification.py
Only review output is written. Saved candidate thresholds are replayed verbatim.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = ROOT/'experiments/v12_expert'
ALLOWED = {
 'phishing_url/v12_expert.py','phishing_url/v11_meta.py','phishing_url/v11_bundle.py',
 'phishing_url/content_checkpoints.py','phishing_url/content_validation.py','phishing_url/evaluation.py',
 'experiments/v11_meta/selection.json','experiments/v11_meta/preregistered.json','experiments/v11_meta/training_frame.npz',
 'reports/v10/selection.json','experiments/v10_margin/selected_oof.npz',
 'reports/split_indices.npz','reports/v3/split_indices.npz',
}
def own_audit(event,args):
    if event.startswith(('socket.connect','socket.getaddrinfo','socket.bind')):
        raise PermissionError('Offline audit')
    if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
        return
    path=Path(os.fsdecode(args[0])).resolve()
    mode,flags=args[1:3]
    writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
    if writing and not path.is_relative_to(OUT):
        raise PermissionError('Review may only write its directory')
    if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT/'.venv') or path.is_relative_to(OUT):
        return
    rel=path.relative_to(ROOT).as_posix()
    if rel in ALLOWED:
        return
    if path.is_relative_to(EXP) and not any(t in rel.lower() for t in ('holdout','diagnostic','test','reference')):
        return
    raise PermissionError('Outside review allowlist: '+rel)
sys.addaudithook(own_audit)

def read_json(path):
    return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def functions(path,names,env):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

result={'checks':[],'pending':[],'limitations':[],'snapshots':{},'folds':[]}
def check(name,condition,detail=None):
    result['checks'].append({'name':name,'passed':bool(condition),**({'detail':detail} if detail is not None else {})})
def equal(a,b):
    return np.array_equal(a,b)
def independent_metrics(y,p,cut):
    pred=p>=cut
    tn=int(((y==0)&~pred).sum()); fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum()); tp=int(((y==1)&pred).sum())
    return {'tn':tn,'fp':fp,'fn':fn,'tp':tp,'f1':2*tp/(2*tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn),'rows':len(y),'threshold':cut,'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p))}
def metrics_equal(a,b):
    return a.keys()==b.keys() and all(np.isclose(a[k],b[k],rtol=0,atol=2e-15) for k in a)

with threadpool_limits(limits=2):
    expert=functions(ROOT/'phishing_url/v12_expert.py',{'eligibility','expert_frame','blend_expert'},{'np':np,'pd':pd,'logit':logit})
    search_source=(EXP/'search.py').read_text()
    protocol=read_json(EXP/'preregistered.json')
    result['snapshots']={p:sha(ROOT/p) for p in sorted(ALLOWED) if (ROOT/p).exists()}
    result['snapshots'].update({'experiments/v12_expert/search.py':sha(EXP/'search.py'),'experiments/v12_expert/preregistered.json':sha(EXP/'preregistered.json')})
    check('source_hash',protocol['source_sha256']==sha(EXP/'search.py'))
    check('expert_source_hash',protocol['expert_source_sha256']==sha(ROOT/'phishing_url/v12_expert.py'))
    check('bounded_frozen_parameters',protocol['classifier']=={'iterations':800,'learning_rate':.04,'depth':5,'l2_leaf_reg':30.,'loss_function':'Logloss','thread_count':2,'random_seed':20261010,'verbose':False,'allow_writing_files':False} and protocol['weights']==[.25,.5,.75] and protocol['classifier_fits']==3)
    for name,digest in protocol['input_sha256'].items():
        if name in ALLOWED:
            actual=sha(ROOT/name); result['snapshots'][name]=actual
            check('input_hash:'+name,actual==digest)
        else:
            result['limitations'].append('Not independently rehashed or opened by reviewer: '+name)
    v11=read_json(ROOT/'experiments/v11_meta/selection.json')
    old_protocol=read_json(ROOT/'experiments/v11_meta/preregistered.json')
    v10=read_json(ROOT/'reports/v10/selection.json')
    cols=v11['frame_columns']
    check('same_original_splitter',protocol['splitter']==old_protocol['splitter'])
    check('cache_digest_chain',protocol['input_sha256']['data/content_train.joblib']==v11['train_cache_sha256']==v10['train_cache_sha256'])
    check('limitation_explicit',all(t in protocol['limitation'].lower() for t in ['reused development','not fully nested','not independent']))
    with np.load(ROOT/'experiments/v11_meta/training_frame.npz',allow_pickle=False) as z:
        dev,y,meta=z['indices'],z['labels'],z['values']
    with np.load(ROOT/'experiments/v10_margin/selected_oof.npz',allow_pickle=False) as z:
        check('base_meta_row_label_alignment',equal(dev,z['indices']) and equal(y,z['labels']))
        baseline=z['probabilities']
    with np.load(ROOT/'reports/v3/split_indices.npz',allow_pickle=False) as z:
        check('exact_original_mapped_development',equal(dev,z['development']))
        check('excluded_original_holdout_and_added_indices',np.intersect1d(dev,z['holdout']).size==0 and np.intersect1d(dev,z['added']).size==0)
        with np.load(ROOT/'reports/split_indices.npz',allow_pickle=False) as original:
            check('original_mapping_chain',equal(z['old_to_cache'][original['development']],dev))
    check('unique_development_and_binary_labels',len(np.unique(dev))==len(dev) and equal(np.unique(y),[0,1]))
    check('meta_schema_unique_finite_no_label_field',meta.shape==(len(y),len(cols)) and len(cols)==len(set(cols)) and np.isfinite(meta).all() and not set(cols)&{'y','label','labels','target','result'})
    check('no_meta_column_equals_row_labels',not any(equal(meta[:,i],y) for i in range(meta.shape[1])))
    env=functions(ROOT/'phishing_url/v11_meta.py',{'COMPONENTS','STRUCTURAL','ensemble_frame'},{'np':np,'pd':pd,'logit':logit})
    expected_meta=env['ensemble_frame']({c:np.array([.2,.8]) for c in env['COMPONENTS']},pd.DataFrame({c:[1.,2.] for c in env['STRUCTURAL']}),np.array([False,True]),{'domain':np.array([[0,0],[1,0]]),'title':np.array([[0,1],[0,0]])},.4)
    check('meta_schema_matches_current_generator',list(expected_meta.columns)==cols)
    cut=v10['selected']['threshold']; before=baseline>=cut
    check('baseline_metrics_reproduced',metrics_equal(independent_metrics(y,baseline,cut),v10['selected']['metrics']))
    eligible=expert['eligibility'](baseline)
    check('eligibility_exact',equal(eligible,(baseline>=.05)&(baseline<=.95)))
    folds=list(StratifiedKFold(**protocol['splitter']).split(np.zeros(len(y)),y))
    foldids=np.full(len(y),-1,dtype=np.int8)
    for f,(tr,va) in enumerate(folds):
        fit=tr[eligible[tr]]; pred=va[eligible[va]]; foldids[va]=f
        check(f'fold_{f}_no_validation_in_fit',np.intersect1d(dev[fit],dev[va]).size==0)
        info={'fold':f,'fit_rows':len(fit),'eligible_validation_rows':len(pred),'validation_rows':len(va),'baseline_errors':int((before[va]!=y[va]).sum()),'ordered_training_indices_sha256':hashlib.sha256(dev[fit].tobytes()).hexdigest()}
        path=EXP/f'models/fold_{f}.joblib'
        if path.exists():
            saved=joblib.load(path); guard=saved['guard']; cfg=guard['config']
            check(f'fold_{f}_saved_scope',guard['training_rows']==len(fit) and guard['ordered_training_indices_sha256']==info['ordered_training_indices_sha256'])
            check(f'fold_{f}_guard_hashes',guard['data_sha256']==protocol['input_sha256']['data/content_train.joblib'] and cfg['source_sha256']==protocol['expert_source_sha256'] and cfg['protocol_sha256']==sha(EXP/'preregistered.json'))
            check(f'fold_{f}_model_parameters',saved['model'].get_params()==protocol['classifier']==cfg['parameters'])
            check(f'fold_{f}_model_feature_schema',list(saved['model'].feature_names_)==cfg['frame_columns'])
            info['saved_model_sha256']=sha(path)
            del saved
        else:
            result['pending'].append(f'fold_{f} saved model guard')
        result['folds'].append(info)
        if path.exists():
            info['feature_columns']=cfg['frame_columns']
    ptest=np.array([0.,.049,.05,.5,.95,.951,1.]); qtest=1-ptest
    for w in protocol['weights']:
        blended=expert['blend_expert'](ptest,qtest,w); mask=expert['eligibility'](ptest)
        check(f'endpoint_and_boundary_probe_{w}',equal(blended[~mask],ptest[~mask]) and equal(mask,[False,False,True,True,True,False,False]))
    probe=pd.DataFrame({'a':[1.,2.],'scheme':['https','http'],'b':[True,False]})
    try:
        frame=expert['expert_frame'](pd.DataFrame({'meta':[.1,.2]}),probe,np.array([.3,.4]))
        check('repaired_numeric_dtype_filter',all('scheme' not in c for c in frame.columns) and np.isfinite(frame.to_numpy(dtype=float)).all())
        result['numeric_probe_columns']=list(frame.columns)
    except (ValueError,TypeError) as exc:
        check('repaired_numeric_dtype_filter',False,str(exc))
    auditfn=functions(EXP/'search.py',{'audit'},{'os':os,'Path':Path,'ROOT':ROOT,'OUT':EXP})['audit']
    for forbidden in ['external/restricted','reports/v2/restricted','submissions/restricted','data/test.csv','models/holdout.bin','data/reference.sql']:
        try: auditfn('open',(str(ROOT/forbidden),'r',0)); blocked=False
        except PermissionError: blocked=True
        check('search_guard_blocks:'+forbidden,blocked)
    try: auditfn('socket.connect',()); blocked=False
    except PermissionError: blocked=True
    check('search_guard_blocks_network',blocked)
    result['limitations'] += ['Base OOF training for meta-training rows can contain meta-validation rows; specialist row exclusion does not make CV fully nested.', 'Train-only domain/title evidence is label-derived by design; current labels are not explicit feature columns. Upstream cache generation was not re-audited.', 'Ordered URL/label guard hash cannot be independently reconstructed without forbidden large cache reads.', 'Search audit is a path denylist installed after imports, not a complete process access log. It permits paths outside ROOT.', 'Saved thresholds only were evaluated; no new threshold or hyperparameter search.']
    complete=all((EXP/name).exists() for name in ['selection.json','development_expert.npz','selected_oof.npz','training_frame.npz'])
    if complete:
        report=read_json(EXP/'selection.json')
        with np.load(EXP/'development_expert.npz',allow_pickle=False) as z:
            p=z['probabilities']; check('expert_oof_alignment',equal(z['indices'],dev) and equal(z['labels'],y) and equal(z['folds'],foldids) and equal(z['eligibility'],eligible))
        check('expert_confident_probabilities_unchanged',equal(p[~eligible],baseline[~eligible]))
        check('expert_probabilities_valid',np.isfinite(p).all() and ((p>=0)&(p<=1)).all())
        check('report_protocol_exact',report['protocol']==protocol)
        check('candidate_weights_frozen',[c['weight'] for c in report['candidates']]==protocol['weights'])
        summaries=[]
        for c in report['candidates']:
            prob=expert['blend_expert'](baseline,p,c['weight']); after=prob>=c['threshold']; frozen=prob>=cut
            m=independent_metrics(y,prob,c['threshold']); fm=independent_metrics(y,prob,cut)
            net=int((before!=y).sum()-(after!=y).sum()); fnet=int((before!=y).sum()-(frozen!=y).sum())
            fc=[{'fold':f,'before_errors':int((before[va]!=y[va]).sum()),'after_errors':int((after[va]!=y[va]).sum())} for f,(_,va) in enumerate(folds)]
            gate=net>=8 and fnet>=8 and m['f1']>v10['selected']['metrics']['f1'] and sum(z['after_errors']<z['before_errors'] for z in fc)>=2 and all(z['after_errors']<=z['before_errors']+3 for z in fc)
            fixed=int(((before!=y)&(after==y)).sum()); new=int(((before==y)&(after!=y)).sum())
            check(f'candidate_{c["weight"]}_arithmetic',metrics_equal(m,c['metrics']) and metrics_equal(fm,c['metrics_at_baseline_threshold']) and net==c['net_errors']==fixed-new and fnet==c['frozen_net'] and fixed==c['fixed_errors'] and new==c['new_errors'] and fc==c['fold_comparison'] and gate==c['gate_passed'])
            check(f'candidate_{c["weight"]}_confident_outputs_unchanged',equal(prob[~eligible],baseline[~eligible]) and equal(after[~eligible],before[~eligible]))
            summaries.append({'weight':c['weight'],'threshold':c['threshold'],'errors':m['fp']+m['fn'],'net_errors':net,'frozen_net':fnet,'gate_passed':gate,'fold_comparison':fc})
        passed=[c for c in report['candidates'] if c['gate_passed']]
        chosen=max(passed or report['candidates'],key=lambda c:(c['metrics']['f1'],c['net_errors'],-c['weight']))
        check('selected_candidate_and_gate',chosen==report['selected'] and report['gate_passed']==bool(passed))
        with np.load(EXP/'selected_oof.npz',allow_pickle=False) as z:
            check('selected_oof_exact',equal(z['indices'],dev) and equal(z['labels'],y) and equal(z['folds'],foldids) and equal(z['baseline'],baseline) and equal(z['probabilities'],expert['blend_expert'](baseline,p,chosen['weight'])))
        with np.load(EXP/'training_frame.npz',allow_pickle=False) as z:
            values=z['values']; columns=z['columns'].tolist()
            check('training_frame_aligned',equal(z['indices'],dev) and equal(z['labels'],y) and equal(z['eligibility'],eligible))
            check('training_frame_numeric_finite',values.dtype.kind in 'fiu' and np.isfinite(values).all())
            check('training_frame_schema',columns==report['feature_columns'] and len(columns)==len(set(columns)) and values.shape==(len(y),len(columns)))
            check('all_saved_models_match_final_schema',all(f.get('feature_columns')==columns for f in result['folds']))
            check('training_meta_block_exact',columns[:len(cols)]==cols and equal(values[:,:len(cols)],meta))
            check('training_baseline_block_exact',columns[-2:]==['baseline_probability','baseline_logit'] and equal(values[:,-2],baseline) and equal(values[:,-1],logit(np.clip(baseline,1e-6,1-1e-6))))
            check('no_label_column_in_final_features',not any(c.removeprefix('numeric_') in {'y','label','labels','target','result'} for c in columns) and not any(equal(values[:,i],y) for i in range(values.shape[1])))
            numeric_cols=[c.removeprefix('numeric_') for c in columns if c.startswith('numeric_')]
            check('numeric_columns_declared_exact',numeric_cols==report['numeric_columns'])
            result['feature_count']=len(columns); result['numeric_feature_count']=len(numeric_cols)
        result['candidates']=summaries; result['gate_passed']=report['gate_passed']; result['selected_weight']=chosen['weight']
    else:
        result['pending'].append('Completed selection and development artifact set')
    recovery=EXP/'recovery.json'
    if recovery.exists():
        result['recovery']=read_json(recovery)
        initial=read_json(EXP/'initial_protocol.json')
        check('recovery_initial_source_hashes',initial['source_sha256']==sha(EXP/'initial_search_source.txt') and initial['expert_source_sha256']==sha(EXP/'initial_expert_source.txt'))
        frozen_fields=['classifier','weights','classifier_fits','splitter','eligibility','fit_scope','gate','limitation','input_sha256']
        check('recovery_protocol_changes_preprocessing_only',all(initial[k]==protocol[k] for k in frozen_fields))
        old_expert=(EXP/'initial_expert_source.txt').read_text()
        now_expert=(ROOT/'phishing_url/v12_expert.py').read_text()
        check('recovery_expert_only_dtype_filter_added',now_expert.replace('    numeric = numeric.select_dtypes(include=["number", "bool"])\n','')==old_expert)
        old_search=(EXP/'initial_search_source.txt').read_text()
        stripped='\n'.join(line for line in search_source.split('\n') if '"numeric_schema":' not in line)
        check('recovery_search_only_schema_documentation_added',stripped==old_search)
        initial_log=(EXP/'initial_run.log').read_text()
        check('initial_failure_before_fit_supported_by_log',"could not convert string to float: 'https'" in initial_log and 'expert_fold_started' not in initial_log)
        logs=[json.loads(line) for line in (EXP/'run.log').read_text().splitlines() if line.startswith('{')]
        check('three_completed_folds_logged',[v['fold'] for v in logs if v.get('event')=='expert_fold_complete']==[0,1,2])
        for name in ['initial_protocol.json','initial_search_source.txt','initial_expert_source.txt','initial_run.log','recovery.json','run.log','selection.json','development_expert.npz','selected_oof.npz','training_frame.npz']:
            if (EXP/name).exists(): result['snapshots']['experiments/v12_expert/'+name]=sha(EXP/name)
        for f in result['folds']:
            f.pop('feature_columns',None)
    else:
        result['pending'].append('recovery.json')
    result['development_rows']=len(y); result['eligible_rows']=int(eligible.sum()); result['confident_rows']=int((~eligible).sum()); result['baseline_errors']=int((before!=y).sum())
    result['passed']=all(c['passed'] for c in result['checks'])
    result['check_count']=len(result['checks'])
    result['complete']=complete and not result['pending']
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','complete','check_count','pending','development_rows','eligible_rows','baseline_errors']},indent=2))
    print('Failed checks:',[c for c in result['checks'] if not c['passed']])
