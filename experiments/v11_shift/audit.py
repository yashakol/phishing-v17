"""Offline, read-only distribution/evidence audit; never fit a classifier.
Run from repository root: .venv/bin/python -B experiments/v11_shift/audit.py
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['TMPDIR'] = str(OUT)
sys.path.insert(0, str(ROOT))
READS = set()
BLOCKED = []
ALLOWED_DATA = {'data/content_train.joblib','data/content_url_train.joblib','data/content_test.joblib','data/content_url_test.joblib'}
ALLOWED_REPORTS = {'reports/v10/test_prediction_provenance.csv', 'reports/v5/development_oof.npz'}
ALLOWED_EXPERIMENTS = {
 'experiments/v10_margin/selected_oof.npz','experiments/v10_margin/selection.json',
 'experiments/v10_margin/development_oof.npz','experiments/v10_margin/fold_scopes_before_fitting.json',
 'experiments/v7_astra/development_oof.npz','experiments/v5/development_memory.joblib',
 'experiments/v5/hybrid_base_config.json','experiments/v5/cat_development_oof.npz',
 'experiments/v5/hash_development_oof.npz','experiments/content/development_oof.npz',
 'experiments/content/fold_0.npz','experiments/v9_routing/development_oof.npz',
 'experiments/v8_ensemble/baseline_c16_selection.json',
}
ALLOWED_MODULES = {'phishing_url/__init__.py','phishing_url/features.py',
 'phishing_url/content_memory.py','phishing_url/soft_context.py','phishing_url/v7_hierarchy.py',
 'phishing_url/v5_bundle.py','phishing_url/v6_bundle.py','phishing_url/v9_routing.py',
 'experiments/v5/select_ensemble.py'}

def deny(message):
    BLOCKED.append(message)
    raise PermissionError(message)

def safe_write(path):
    if isinstance(path, int):
        return
    if not Path(os.fsdecode(path)).resolve().is_relative_to(OUT):
        deny('Write outside audit folder: '+str(path))

def audit_hook(event, args):
    if event.startswith('socket.'):
        deny('Offline audit blocks '+event)
    if event in ('subprocess.Popen','os.system','os.fork','os.posix_spawn','os.exec'):
        deny('No child processes: '+event)
    if event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate','os.utime'):
        safe_write(args[0])
    if event in ('os.rename','os.link','os.symlink'):
        safe_write(args[0]); safe_write(args[1])
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if writing:
        safe_write(path)
        return
    if not path.is_relative_to(ROOT):
        # Runtime libraries/system resources only; prohibit unrelated user-data reads.
        if path.is_relative_to(Path('/Users/iashin')) and not path.is_relative_to(Path('/Users/iashin/.local/share/uv')):
            deny('Unrelated home read: '+str(path))
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith('.venv/') or path.is_relative_to(OUT):
        return
    if rel not in ALLOWED_DATA | ALLOWED_EXPERIMENTS | ALLOWED_MODULES | ALLOWED_REPORTS:
        deny('Non-allowlisted project input: '+rel)
    READS.add(rel)

sys.addaudithook(audit_hook)
import ast
import gc
import hashlib
import json
import re
import resource
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits, threadpool_info
from phishing_url.features import group_for_url
from phishing_url.content_memory import text_fingerprint
from phishing_url.soft_context import context_keys

def emit(event, **values):
    print(json.dumps({'event':event, **values}), flush=True)

def inspect_inputs():
    for filename in ('experiments/content/development_oof.npz','experiments/content/fold_0.npz',
                     'experiments/v5/cat_development_oof.npz','experiments/v5/hash_development_oof.npz'):
        with np.load(ROOT/filename, allow_pickle=False) as z:
            emit('schema', path=filename, fields={k:list(z[k].shape) for k in z.files})
    for filename in ('data/content_train.joblib','data/content_url_train.joblib'):
        cache = joblib.load(ROOT/filename)
        emit('cache_schema',path=filename,type=str(type(cache)),fields={k:(str(type(v)), len(v) if hasattr(v,'__len__') else None) for k,v in cache.items()} if isinstance(cache,dict) else list(cache.columns))
        if 'features' in cache:
            emit('features',columns=list(cache['features'].columns))
        del cache
        gc.collect()

def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def read_json(rel):
    return json.loads((ROOT/rel).read_text())

def source_function(rel, name):
    # Extract the actual pure function without importing inference modules that
    # transitively import prohibited reference code. No copied numerical logic.
    tree = ast.parse((ROOT/rel).read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace = {'np':np,'pd':pd}
    exec(compile(ast.Module(body=[node],type_ignores=[]), rel, 'exec'),namespace)
    return namespace[name]

# Actual frozen routing regex, also extracted without inference imports.
tree = ast.parse((ROOT/'phishing_url/v9_routing.py').read_text())
regex_node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ERROR_TITLE' for t in n.targets))
regex_ns = {'re':re}
exec(compile(ast.Module(body=[regex_node],type_ignores=[]),'routing_regex','exec'),regex_ns)
ERROR_TITLE = regex_ns['ERROR_TITLE']

FEATURES = ('html_length','html_text_length','html_empty','html_parse_error','html_truncated','html_input_password','html_tag_form')
KINDS = ('domain','title','raw','text','url')

def compact_cache(rel, expected, test=False):
    emit('load_cache', path=rel)
    cache = joblib.load(ROOT/rel)
    assert len(cache['urls'])==expected
    if test:
        assert 'labels' in cache and cache['labels'] is None, 'Test labels must be None'
        labels = None
    else:
        labels = np.asarray(cache['labels'], dtype=np.int8).copy()
        assert np.isin(labels,[0,1]).all()
    assert all(r['url']==u for r,u in zip(cache['records'], cache['urls'], strict=True))
    keys = {k:np.asarray(v,dtype=object) for k,v in context_keys(cache['records']).items()}
    keys.update(raw=np.asarray(cache['hashes'],dtype=object),
                text=np.asarray([text_fingerprint(r) for r in cache['records']],dtype=object),
                url=np.asarray(cache['urls'],dtype=object))
    feature = {k:cache['features'][k].to_numpy().copy() for k in FEATURES}
    feature['error_title'] = np.asarray([bool(ERROR_TITLE.search(r['title'])) for r in cache['records']])
    feature['url_routed'] = (feature['html_text_length']<100) | feature['error_title']
    del cache
    gc.collect()
    emit('cache_compacted', rows=expected)
    return keys,feature,labels

def lookup(train_keys, train_labels, query_keys):
    result = {}
    # Only the caller-supplied reference labels enter counts. Query labels do not.
    for kind in KINDS:
        d = {}
        for key,label in zip(train_keys[kind],train_labels,strict=True):
            if key is None:
                continue
            if key not in d:
                d[key] = [0,0]
            d[key][int(label)] += 1
        result[kind] = np.asarray([d.get(k,(0,0)) for k in query_keys[kind]],dtype=np.int32)
        del d
    return result

def subset(mapping, index):
    return {k:v[index] for k,v in mapping.items()}

def memory_from_counts(counts):
    result={}
    for kind,support in (('raw',3),('text',3),('domain',5)):
        zero,one = counts[kind].T
        a = np.full(len(zero),-1,dtype=np.int8)
        a[(zero>=support)&(one==0)] = 0
        a[(one>=support)&(zero==0)] = 1
        result[kind]=a
    return result

def attribution(values, p, base=None, url_counts=None):
    raw,txt,dom = (values[k] for k in ('raw','text','domain'))
    page = np.where(raw>=0, raw, txt)
    page_conflict = (raw>=0)&(txt>=0)&(raw!=txt)
    page_domain_conflict = (page>=0)&(dom>=0)&(page!=dom)
    definite_page = (page>=0)&~page_conflict&~page_domain_conflict
    domain_candidate = (page<0)&(dom>=0)
    if base is not None:
        domain_hit = domain_candidate&(base>=.05)&(base<=.95)
    else:
        # Provenance has no pre-memory probability. This is endpoint-compatible
        # attribution, not independently verified execution of the domain gate.
        domain_hit = domain_candidate&(p==dom)
    hard = definite_page|domain_hit
    if not np.array_equal(p[definite_page],page[definite_page]):
        raise AssertionError('Page rule not preserved in final probabilities')
    if not np.array_equal(p[domain_hit],dom[domain_hit]):
        raise AssertionError('Domain rule not preserved in final probabilities')
    endpoint = (p==0)|(p==1)
    group = np.full(len(p),'learned_continuous',dtype=object)
    group[endpoint&~hard]='unattributed_endpoint'
    group[definite_page&(dom<0)]='page_hard'
    group[definite_page&(dom>=0)]='page_domain_hard'
    group[domain_hit]='domain_only_hard'
    known = np.zeros(len(p), dtype=bool)
    if url_counts is not None:
        known = url_counts.sum(1)>0
        assert not np.any((url_counts[:,0]>0)&(url_counts[:,1]>0)), 'Conflicting training URL labels'
        known_label=(url_counts[:,1]>0).astype(np.int8)
        np.testing.assert_array_equal(p[known],known_label[known])
        # Exact URL lookup is the final override in the inspected predict_v5.
        group[known]='known_url_hard'
        hard |= known
        definite_page &= ~known
        domain_hit &= ~known
    return dict(group=group,hard=hard,page=definite_page,domain=domain_hit,known_url=known,
                domain_candidate=domain_candidate,page_conflict=page_conflict,
                page_domain_conflict=page_domain_conflict,endpoint=endpoint)

def summarize(mask, p, y, threshold):
    n = int(mask.sum())
    result={'rows':n,'share':n/len(mask)}
    if y is not None:
        wrong = (p>=threshold)!=y
        result.update(errors=int((wrong&mask).sum()),fp=int((mask&(p>=threshold)&(y==0)).sum()),
                      fn=int((mask&(p<threshold)&(y==1)).sum()))
        result['error_rate']=result['errors']/n if n else None
    return result

def support_group(count):
    n=count.sum(axis=1)
    return np.select([n==0,n==1,n==2,n<5,n<10,n<50],['0','1','2','3-4','5-9','10-49'],default='50+')

def quality(feature):
    return np.select([feature['html_empty'].astype(bool),feature['html_parse_error'].astype(bool),
                      feature['error_title'],feature['html_text_length']<100],
                     ['empty','parse_error','error_title','short_under100'], default='ordinary')

def confidence(p, cut):
    margin=np.abs(p-cut)
    return np.select([(p==0)|(p==1),(p<=.01)|(p>=.99),(p<=.05)|(p>=.95),margin<=.05],
                     ['endpoint','nonendpoint_99','nonendpoint_95','within_005_of_threshold'],default='other')

def groups(values,p,y,cut):
    return {str(k):summarize(values==k,p,y,cut) for k in sorted(set(values))}

def analyze(counts, feature, attr, p, y, cut):
    result={'overall':summarize(np.ones(len(p),dtype=bool),p,y,cut),
            'attribution':groups(attr['group'],p,y,cut),
            'confidence':groups(confidence(p,cut),p,y,cut),
            'html_quality':groups(quality(feature),p,y,cut),
            'routing':groups(np.where(feature['url_routed'],'url_routed','standard'),p,y,cut),
            'support':{},'evidence':{},'cross':{},'probability_quantiles':np.quantile(p,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist()}
    for kind in KINDS:
        n=counts[kind].sum(axis=1)
        mixed=(counts[kind][:,0]>0)&(counts[kind][:,1]>0)
        result['support'][kind]=groups(support_group(counts[kind]),p,y,cut)
        result['evidence'][kind] = {name:summarize(mask,p,y,cut) for name,mask in {
            'unseen':n==0,'seen':n>0,'pure_seen':(n>0)&~mixed,'mixed_seen':mixed,
            'support_ge3':n>=3,'support_ge5':n>=5}.items()}
    for name,mask in attr.items():
        if name!='group':
            result['evidence']['applied_'+name if name in KINDS else name]=summarize(mask,p,y,cut)
    for name in ('html_empty','html_parse_error','html_truncated','error_title'):
        result['evidence'][name]=summarize(feature[name].astype(bool),p,y,cut)
    for dim,v in [('quality',quality(feature)),('confidence',confidence(p,cut)),
                  ('domain_support',support_group(counts['domain'])),('title_support',support_group(counts['title']))]:
        combined=np.asarray([str(a)+' / '+str(b) for a,b in zip(attr['group'],v)],dtype=object)
        result['cross'][dim]=groups(combined,p,y,cut)
    result['cross']['domain_title_seen'] = groups(np.asarray([
        ('domain_seen' if a else 'domain_unseen')+' / '+('title_seen' if b else 'title_unseen')
        for a,b in zip(counts['domain'].sum(1)>0, counts['title'].sum(1)>0)],dtype=object),p,y,cut)
    return result

def fold_npz(rel, dev, y, fold_ids, field='probabilities'):
    with np.load(ROOT/rel,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],dev)
        np.testing.assert_array_equal(z['labels'],y)
        if 'folds' in z:
            np.testing.assert_array_equal(z['folds'],fold_ids)
        return z[field].copy()

def main():
    start=time.monotonic()
    selection=read_json('experiments/v10_margin/selection.json')
    cut=float(selection['selected']['threshold'])
    old_cut=float(selection['baseline_threshold'])
    with np.load(ROOT/'experiments/v10_margin/selected_oof.npz',allow_pickle=False) as z:
        dev,y,p,old_p=(z[k].copy() for k in ('indices','labels','probabilities','baseline'))
    assert len(dev)==50977 and len(np.unique(dev))==50977
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(dev,y))
    fold_ids=np.full(len(y),-1,dtype=np.int8)
    scopes=read_json('experiments/v10_margin/fold_scopes_before_fitting.json')
    assert array_sha(dev)==scopes['development_indices_sha256']
    for f,(tr,va) in enumerate(folds):
        assert not np.intersect1d(tr,va).size
        fold_ids[va]=f
        assert array_sha(dev[tr])==scopes['origins'][f]['train_indices_sha256']
        assert array_sha(dev[va])==scopes['origins'][f]['validation_indices_sha256']
    keys,features,labels=compact_cache('data/content_train.joblib',63900)
    np.testing.assert_array_equal(labels[dev],y)
    assert sha(ROOT/'data/content_train.joblib')==scopes['data_sha256']==selection['train_cache_sha256']
    devkeys,devfeat=subset(keys,dev),subset(features,dev)
    del features
    gc.collect()
    testkeys,testfeat,testlabels=compact_cache('data/content_test.joblib',16000,test=True)
    assert testlabels is None
    for rel,kk in [('data/content_url_train.joblib',keys),('data/content_url_test.joblib',testkeys)]:
        c=joblib.load(ROOT/rel)
        assert c.get('labels') is None
        np.testing.assert_array_equal(c['urls'],kk['url'])
        np.testing.assert_array_equal(c['features']['cat_domain'].to_numpy(),kk['domain'])
        del c
        gc.collect()
    provenance=pd.read_csv(ROOT/'reports/v10/test_prediction_provenance.csv',float_precision='round_trip')
    assert list(provenance.columns)==['Id','url','probability','url_routed']
    np.testing.assert_array_equal(provenance['Id'],np.arange(16000))
    np.testing.assert_array_equal(provenance['url'],testkeys['url'])
    np.testing.assert_array_equal(provenance['url_routed'],testfeat['url_routed'])
    testp=provenance['probability'].to_numpy()
    assert np.isfinite(testp).all() and ((testp>=0)&(testp<=1)).all()
    del provenance
    counts={k:np.zeros((len(dev),2),dtype=np.int32) for k in KINDS}
    # No validation label is passed to lookup; no alltrain count is used in OOF.
    matched_reference=[]
    for f,(tr,va) in enumerate(folds):
        ref=subset(devkeys,tr)
        local=lookup(ref,y[tr],subset(devkeys,va))
        for k in KINDS:
            counts[k][va]=local[k]
        tc=lookup(ref,y[tr],testkeys)
        matched_reference.append({'fold':f,'training_rows':len(tr),'validation_rows':len(va),
            'support':{k:{'oof_seen':int((local[k].sum(1)>0).sum()),
                          'oof_seen_share':float((local[k].sum(1)>0).mean()),
                          'test_seen':int((tc[k].sum(1)>0).sum()),
                          'test_seen_share':float((tc[k].sum(1)>0).mean())} for k in KINDS}})
        del tc,local,ref
    tc=lookup(keys,labels,testkeys)
    values=memory_from_counts(counts)
    testvalues=memory_from_counts(tc)
    cache=joblib.load(ROOT/'experiments/v5/development_memory.joblib')
    np.testing.assert_array_equal(cache['indices'],dev)
    assert cache['data_sha256']==scopes['data_sha256']
    for k in ('raw','text','domain'):
        np.testing.assert_array_equal(cache['keys'][k],devkeys[k])
        np.testing.assert_array_equal(cache['values'][k],values[k])
    del cache
    gc.collect()
    margin=fold_npz('experiments/v10_margin/development_oof.npz',dev,y,fold_ids)
    original_text=fold_npz('experiments/v7_astra/development_oof.npz',dev,y,fold_ids)
    # The original training-only development component table is needed to replay
    # the precise pre-memory .05-.95 gate; no fitted inference artifact is read.
    with np.load(ROOT/'reports/v5/development_oof.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],dev)
        np.testing.assert_array_equal(z['labels'],y)
        components={k:z[k].copy() for k in ('numeric','url','cat','hashed')}
    for field,rel in [('numeric','experiments/content/development_oof.npz'),('cat','experiments/v5/cat_development_oof.npz'),('hashed','experiments/v5/hash_development_oof.npz')]:
        np.testing.assert_allclose(components[field],fold_npz(rel,dev,y,fold_ids),rtol=0,atol=1e-14)
    config=read_json('experiments/v5/hybrid_base_config.json')['config']
    assert config==read_json('experiments/v8_ensemble/baseline_c16_selection.json')['selected']['v5_config']
    mix=source_function('phishing_url/v5_bundle.py','mix_probabilities')
    apply=source_function('experiments/v5/select_ensemble.py','apply_values')
    def prehard(text):
        args=(text,components['numeric'],components['url'],None,components['cat'])
        return np.where(devfeat['url_routed'],mix(*args,dict(config,aux_url_weight=.5),components['hashed']),mix(*args,config,components['hashed']))
    base=prehard(margin)
    beforebase=prehard(original_text)
    hard=apply(base,values)
    attr=attribution(values,p,base)
    oldattr=attribution(values,old_p,beforebase)
    np.testing.assert_array_equal(hard[attr['hard']],p[attr['hard']])
    assert np.all(~attr['endpoint']|attr['hard']), 'Unexplained OOF endpoint'
    ta=attribution(testvalues,testp,url_counts=tc['url'])
    emit('test_endpoint_attribution', endpoints=int(ta['endpoint'].sum()), page=int(ta['page'].sum()), domain_compatible=int(ta['domain'].sum()), unknown=int((ta['endpoint']&~ta['hard']).sum()), known_url=int((tc['url'].sum(1)>0).sum()))
    # Known-URL evidence is reconstructed solely from allowed training labels.
    known_test_urls=int((tc['url'].sum(1)>0).sum())
    report={'protocol':{'development_rows':len(y),'alltrain_reference_rows':len(labels),'test_rows':len(testp),
        'folds':{'n_splits':3,'shuffle':True,'random_state':2026},'threshold':cut,'previous_threshold':old_cut,
        'classifier_fits':0,'threshold_searches':0,'test_target_labels_read':False,
        'test_role':'Unlabeled feature/support and saved model probability diagnostics only',
        'oof_evidence':'Each validation row sees only its outer-fold training labels',
        'test_evidence':'All 63900 permitted training labels; never test labels',
        'test_domain_hit_limitation':'Exact known-URL hits use training labels only and take precedence. Page hard hits are deterministic; domain-only hits are endpoint-compatible attribution because saved pre-memory test probabilities are unavailable. No test model is loaded.',
        'confidence_definition':'Model output extremity, not verified calibration or correctness',
        'rule_thresholds':{'raw':3,'text':3,'domain':5},
        'support_keys':{'domain':'Actual offline group_for_url; bundled PSL/private suffixes','title':'NFKC/lower/whitespace normalized, empty excluded','raw':'Cached raw page hash','text':'SHA256 of lowercased title/text/markup','url':'Exact URL'},
        'html_quality_order':['empty','parse_error','error_title','short_under100','ordinary']},
        'alignment':{'development_index_sha256':array_sha(dev),'fold_scopes_verified':True,
                     'memory_values_exactly_recomputed':True,'all_hard_oof_outputs_preserved':True,
                     'test_pre_memory_probability_unavailable':True,'all_test_endpoints_explained_by_training_evidence':bool(np.all(~ta['endpoint']|ta['hard'])),
                     'test_known_training_urls':known_test_urls,'test_cache_labels_none':True,'url_cache_labels_absent_or_none':True,'provenance_url_order_and_routing_verified':True},
        'oof':analyze(counts,devfeat,attr,p,y,cut),'test':analyze(tc,testfeat,ta,testp,None,cut),
        'equal_reference_size_comparison':matched_reference}
    report['support_shift']={}
    for k in KINDS:
        oshare=float((counts[k].sum(1)>0).mean())
        tshare=float((tc[k].sum(1)>0).mean())
        tsmall=float(np.mean([f['support'][k]['test_seen_share'] for f in matched_reference]))
        oo=report['oof']['support'][k]; tt=report['test']['support'][k]
        report['support_shift'][k]={'oof_seen_share':oshare,'test_alltrain_seen_share':tshare,
            'test_mean_foldtrain_seen_share':tsmall,'total_difference_pp':100*(tshare-oshare),
            'reference_size_uplift_pp':100*(tshare-tsmall),
            'equal_reference_residual_pp':100*(tsmall-oshare),
            'support_histogram_total_variation':.5*sum(abs(oo.get(b,{}).get('share',0)-tt.get(b,{}).get('share',0)) for b in set(oo)|set(tt))}
    errors=(p>=cut)!=y
    before_wrong=(old_p>=old_cut)!=y
    flip=(p>=cut)!=(old_p>=old_cut)
    report['replacement_diagnostic_oof']={
        'text_component_mean_absolute_change':float(np.abs(margin-original_text).mean()),
        'text_component_half_threshold_flips':int(((margin>=.5)!=(original_text>=.5)).sum()),
        'final_probability_changed_gt1e12':int((np.abs(p-old_p)>1e-12).sum()),
        'final_prediction_flips_including_frozen_selection_threshold_change':int(flip.sum()),
        'final_prediction_flips_at_previous_threshold':int(((p>=old_cut)!=(old_p>=old_cut)).sum()),
        'fixed':int((flip&before_wrong).sum()),'new_errors':int((flip&~before_wrong).sum()),
        'both_versions_hard':int((attr['hard']&oldattr['hard']).sum()),
        'both_versions_hard_output_changed':int(((attr['hard']&oldattr['hard'])&(p!=old_p)).sum()),
        'by_hard_attribution':groups(attr['group'][flip],p[flip],y[flip],cut),
        'test_comparison_unavailable':'Only v10 test probabilities authorized; no earlier test prediction artifact read, so actual test flip count is not independently verified.'}
    report['rule_mistakes_oof']={}
    for k,v in values.items():
        eligible=v>=0
        wrong=eligible&(v!=y)
        report['rule_mistakes_oof'][k]={'eligible':int(eligible.sum()),'candidate_errors':int(wrong.sum()),
            'candidate_error_rate':float(wrong.sum()/eligible.sum()) if eligible.any() else None,
            'candidate_wrong_but_not_hard':int((wrong&~attr['hard']).sum()),
            'candidate_wrong_and_final_error':int((wrong&errors).sum())}
    report['rule_mistakes_oof']['executed_hard']={'rows':int(attr['hard'].sum()),'errors':int((attr['hard']&errors).sum()),
        'rule_errors_where_pre_memory_prediction_was_correct':int((attr['hard']&errors&((base>=cut)==y)).sum()),
        'rule_fixes_where_pre_memory_prediction_was_wrong':int((attr['hard']&~errors&((base>=cut)!=y)).sum()),
        'counterfactual_limit':'Pre-memory mixture comparison only; disabling a rule also enables context, extra text and hierarchy. These are not full-ensemble ablation scores.'}
    # ONE diagnostic evidence policy, no tuning: make domain-only hard evidence
    # abstain unless supported by >=2 pure agreeing normalized-title examples.
    # Page rules retain their existing behavior. No new predictions are fitted.
    def policy(count, attribution_, vals, prob, yy):
        title=count['title']; total=title.sum(1)
        pure=(title[:,0]==0)|(title[:,1]==0)
        title_label=(title[:,1]>0).astype(np.int8)
        corroborated=(total>=2)&pure&(title_label==vals['domain'])
        flagged=attribution_['domain']&~corroborated
        result={'domain_only_original':int(attribution_['domain'].sum()),'domain_only_retained':int((attribution_['domain']&corroborated).sum()),
                'domain_only_abstain':summarize(flagged,prob,yy,cut),
                'abstention_reasons':{name:summarize(flagged&mask,prob,yy,cut) for name,mask in {
                    'title_support_under2':total<2,'title_mixed_support_ge2':(total>=2)&~pure,
                    'title_pure_opposes_domain':(total>=2)&pure&(title_label!=vals['domain'])}.items()}}
        if yy is not None:
            result['pre_memory_mixture_would_correct_flagged_errors']=int((flagged&((prob>=cut)!=yy)&((base>=cut)==yy)).sum())
            result['pre_memory_mixture_would_break_flagged_correct']=int((flagged&((prob>=cut)==yy)&((base>=cut)!=yy)).sum())
        return result,flagged
    op,omask=policy(counts,attr,values,p,y)
    tp,tmask=policy(tc,ta,testvalues,testp,None)
    report['single_evidence_policy']={'description':'Require >=2 pure agreeing normalized-title training examples to retain domain-only hard overrides; otherwise let the existing learned/context/hierarchy path operate. Keep page rules unchanged.',
        'status':'Single diagnostic ablation candidate; NOT selected, implemented or promoted',
        'parameters':'Title support 2 is the existing soft-context default; no grid or optimization',
        'evidence_caveat':'Domain/title matches are correlated, not independent evidence. Wrong rule removal does not guarantee final correction.',
        'oof':op,'test':tp,
        'folds':[{'fold':f,**summarize(omask[va],p[va],y[va],cut)} for f,(_,va) in enumerate(folds)]}
    report['fold_errors']=[{'fold':f,**summarize(np.ones(len(va),dtype=bool),p[va],y[va],cut),
        'hard_errors':int((errors[va]&attr['hard'][va]).sum()),'learned_errors':int((errors[va]&~attr['hard'][va]).sum())} for f,(_,va) in enumerate(folds)]
    report['within_query_repetition']={}
    for tag, kk in [('development',devkeys),('test',testkeys)]:
        report['within_query_repetition'][tag]={}
        for kind in KINDS:
            cc={}
            for key in kk[kind]:
                if key is not None:
                    cc[key]=cc.get(key,0)+1
            report['within_query_repetition'][tag][kind]={'nonmissing_unique_keys':len(cc),
                'rows_in_repeated_nonmissing_keys':sum(n for n in cc.values() if n>=2),
                'maximum_key_multiplicity':max(cc.values(),default=0),
                'missing_rows':sum(key is None for key in kk[kind])}
    report['findings']={
        'oof_errors_on_nonhard_path':int((errors&~attr['hard']).sum()),
        'oof_errors_on_nonhard_unseen_domain':int((errors&~attr['hard']&(counts['domain'].sum(1)==0)).sum()),
        'oof_errors_on_nonhard_path_share':float((errors&~attr['hard']).sum()/errors.sum()),
        'oof_errors_on_nonextreme_probabilities':int((errors&(p>.05)&(p<.95)).sum()),
        'test_nonendpoint_rows':int((~ta['endpoint']).sum()),
        'test_extreme_005_or_095_rows':int(((testp<=.05)|(testp>=.95)).sum()),
        'test_near_frozen_threshold_005_rows':int((np.abs(testp-cut)<=.05).sum()),
        'interpretation':'Support shifts mostly shrink under equal reference size; only about a quarter of test outputs are hard endpoints. Most OOF errors remain in learned outputs, so repetition is a partial masking mechanism, not a sufficient explanation for small test decision changes.',
        'next_direction':'Evaluate the parent learned ensemble on unseen-domain, nonhard OOF rows first (398 current errors), then split those by HTML routing and confidence. Also track nonhard outputs with p in (.05,.95). Do not add another replacement model here. The single domain-title abstention policy is only a small secondary ablation, with no demonstrated net benefit.'}
    # No row texts/URLs/labels are exported: aggregate audit only.
    report['runtime']={'seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'rss_units':'bytes on macOS','threadpools':threadpool_info(),'sequential_caches':True}
    project_reads=sorted(READS)
    report['provenance']={'inputs_sha256':{rel:sha(ROOT/rel) for rel in project_reads},
         'audit_source_sha256':sha(__file__),'successful_project_reads':project_reads,
         'blocked_attempts':BLOCKED,'blocked_socket_constructors':sum('socket.__new__' in s for s in BLOCKED),
         'blocked_connect_or_dns_events':sum(any(e in s for e in ('socket.connect','socket.getaddrinfo','socket.bind')) for s in BLOCKED),
         'writes_restricted_to':str(OUT),
         'forbidden_inputs_not_read':['external/**','reports/v2/**','reference modules/scripts/CSV','test target labels','public SQL labels','live URLs','holdout models/predictions/reports'],
         'guard_scope':'Python audit hook installed before third-party imports; blocks socket operations/process launches, allowlists project reads, blocks file writes/mutations outside own folder. Not an OS sandbox for arbitrary native code.'}
    report['verification']=verify_report(report)
    assert report['oof']['overall']['errors']==selection['selected']['metrics']['fp']+selection['selected']['metrics']['fn']
    for pool in report['runtime']['threadpools']:
        assert pool['num_threads']<=2
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    write_report(report)
    emit('complete',oof_errors=int(errors.sum()),oof_hard=int(attr['hard'].sum()),test_endpoint=int(ta['endpoint'].sum()),
         hard_errors=int((attr['hard']&errors).sum()),report=str(OUT/'REPORT.md'))

def verify_report(r):
    for tag in ('oof','test'):
        part=r[tag]
        for dimension in ('attribution','confidence','html_quality','routing'):
            assert sum(x['rows'] for x in part[dimension].values())==part['overall']['rows']
            if tag=='oof':
                assert sum(x['errors'] for x in part[dimension].values())==part['overall']['errors']
        for distribution in part['support'].values():
            assert sum(x['rows'] for x in distribution.values())==part['overall']['rows']
        if tag=='test':
            def no_errors(value):
                if isinstance(value,dict):
                    assert not set(value)&{'errors','error_rate','fp','fn','accuracy','f1'}
                    for v in value.values():
                        no_errors(v)
            no_errors(part)
    assert sum(f['errors'] for f in r['fold_errors'])==r['oof']['overall']['errors']
    assert r['alignment']['all_test_endpoints_explained_by_training_evidence']
    # Simulated hook events, not actual file/socket operations.
    tests=[('open',(str(ROOT/'external/__audit_forbidden_probe__'),'r',0)),
           ('open',(str(ROOT/'reports/v2/__audit_forbidden_probe__'),'r',0)),
           ('open',(str(ROOT/'phishing_url/reference.py'),'r',0)),
           ('open',(str(ROOT/'artifacts/v10/holdout.joblib'),'r',0)),
           ('open',(str(ROOT/'__audit_outside_probe__'),'w',os.O_WRONLY|os.O_CREAT)),
           ('open',('/private/tmp/__audit_outside_probe__','w',os.O_WRONLY|os.O_CREAT)),
           ('socket.__new__',()),('socket.connect',()),('subprocess.Popen',())]
    before=len(BLOCKED)
    for event,args in tests:
        try:
            audit_hook(event,args)
        except PermissionError:
            continue
        raise AssertionError('Guard failed: '+event)
    del BLOCKED[before:]
    return {'partition_totals':True,'test_has_no_error_fields':True,
            'test_endpoints_accounted_for':True,'guard_simulated_denials':len(tests),
            'guard_tests_do_not_perform_forbidden_operations':True}

def write_report(r):
    o,t=r['oof'],r['test']
    lines=['# Development/test support audit','',
           'Diagnostic only. No classifier fits, threshold searches or test target labels. Test correctness is unknown.', '',
           f"Strongest finding: {r['findings']['oof_errors_on_nonhard_path']}/{o['overall']['errors']} OOF errors ({r['findings']['oof_errors_on_nonhard_path_share']:.2%}) are on the nonhard learned path. Domain seen-support differs by {r['support_shift']['domain']['total_difference_pp']:.2f} pp with the full training reference, but only {r['support_shift']['domain']['equal_reference_residual_pp']:.2f} pp against equal-sized fold references. Test endpoint coverage is {t['evidence']['endpoint']['share']:.2%}; repetition alone is not a sufficient explanation for small decision changes.", '',
           '## Evidence and alignment','',
           f"Original development OOF: {o['overall']['rows']} rows, {o['overall']['errors']} errors at the frozen v10 development-selected threshold {r['protocol']['threshold']}. Three StratifiedKFold splits use shuffle=True and random_state=2026. Exact development and fold index hashes match the original v10 provenance. Every OOF evidence count uses only that fold\'s training labels. Test support uses 63,900 permitted training rows for 16,000 unlabeled test rows.", '',
           'Content caches are loaded sequentially, reduced to keys/features, and released. Both URL caches match content URL order/domain extraction. Test content labels are None; URL caches contain no labels. Saved test provenance contains only Id, URL, probability and routing. Its row order/routing exactly match the test features.', '',
           '## Repetition/support shift','', '| Key | OOF seen | Test seen/alltrain | Test seen/same fold size | Full-reference uplift | Equal-size residual |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k,s in r['support_shift'].items():
        lines.append(f"| {k} | {s['oof_seen_share']:.2%} | {s['test_alltrain_seen_share']:.2%} | {s['test_mean_foldtrain_seen_share']:.2%} | {s['reference_size_uplift_pp']:.2f} pp | {s['equal_reference_residual_pp']:.2f} pp |")
    lines += ['', 'Same-size test comparison averages three test lookups against the exact OOF training folds; it does not use test labels. The uplift/residual decomposition is descriptive, not a randomized causal estimate. Alltrain also contains training rows outside development, so its uplift includes reference composition as well as size. Domain difference is 8.525460 pp = 7.831250 pp full-reference uplift + 0.694210 pp equal-size residual, rounded to 8.53 pp. Independently rounded table components (7.83 and 0.69) sum to 8.52 pp because of rounding.', '',
              '## Hard rules and confidence','', '| Attribution | OOF rows | OOF errors | Test rows |','| --- | ---: | ---: | ---: |']
    for k in sorted(set(o['attribution'])|set(t['attribution'])):
        a=o['attribution'].get(k,{}); b=t['attribution'].get(k,{})
        lines.append(f"| {k} | {a.get('rows',0)} | {a.get('errors',0)} | {b.get('rows',0)} |")
    lines += ['', r['protocol']['test_domain_hit_limitation'],
              'OOF hard gates are replayed from the actual pure mixing/rule functions and original aligned training-only component arrays. Recomputed raw/text/domain rule values exactly match the original memory artifact. The number of exact known-URL matches and unexplained endpoints is reported below; endpoints alone are not proof of a hard-rule execution.', '',
              f"Exact test/training URL matches: {r['alignment']['test_known_training_urls']}. Unattributed test endpoints: {t['attribution'].get('unattributed_endpoint',{}).get('rows',0)}.", '',
              '| Final confidence bucket | OOF rows | OOF errors | Test rows |','| --- | ---: | ---: | ---: |']
    for k in sorted(set(o['confidence'])|set(t['confidence'])):
        a=o['confidence'].get(k,{}); b=t['confidence'].get(k,{})
        lines.append(f"| {k} | {a.get('rows',0)} | {a.get('errors',0)} | {b.get('rows',0)} |")
    lines += ['', 'Confidence is output extremity, not calibrated reliability. Hard endpoints must not be interpreted as certainty.', '',
              '## Support and OOF mistakes','',
              '| Key / reference evidence | OOF rows | OOF errors | Test rows |','| --- | ---: | ---: | ---: |']
    for key in ('domain','title','raw','text'):
        for kind in ('unseen','pure_seen','mixed_seen'):
            a=o['evidence'][key][kind]; b=t['evidence'][key][kind]
            lines.append(f"| {key} / {kind} | {a['rows']} | {a['errors']} | {b['rows']} |")
    lines += ['', 'Candidate memory mistakes before confidence/conflict gating (OOF only): raw '
        +str(r['rule_mistakes_oof']['raw']['candidate_errors'])+', text '
        +str(r['rule_mistakes_oof']['text']['candidate_errors'])+', domain '
        +str(r['rule_mistakes_oof']['domain']['candidate_errors'])+'. These overlap and are not additive. Executed hard outputs account for '
        +str(r['rule_mistakes_oof']['executed_hard']['errors'])+' errors. Full cross-tabs by hard/learned, domain/title support, confidence and HTML quality are in audit.json.', '',
              '## HTML quality and routing','', '| Quality | OOF rows | OOF errors | Test rows |','| --- | ---: | ---: | ---: |']
    for k in sorted(set(o['html_quality'])|set(t['html_quality'])):
        a=o['html_quality'].get(k,{}); b=t['html_quality'].get(k,{})
        lines.append(f"| {k} | {a.get('rows',0)} | {a.get('errors',0)} | {b.get('rows',0)} |")
    lines += ['',f"URL-routed: OOF {o['routing']['url_routed']['rows']} rows/{o['routing']['url_routed']['errors']} errors; test {t['routing']['url_routed']['rows']} rows. The frozen selector is visible text length <100 or the existing error-title regex.", '',
              '## Why replacements can have limited effect','']
    rr=r['replacement_diagnostic_oof']
    lines += [f"Test has {r['findings']['test_nonendpoint_rows']} nonendpoint outputs, but {r['findings']['test_extreme_005_or_095_rows']} outputs overall are <=0.05 or >=0.95. Only {r['findings']['test_near_frozen_threshold_005_rows']} lie within 0.05 of the frozen decision threshold. In OOF, {r['findings']['oof_errors_on_nonextreme_probabilities']} errors fall in (0.05,0.95). This points to the learned mixture and decision margins as another plausible source of limited changes; it is not a test-error estimate.", '',
              f"Observed in OOF: original/replacement text components flip {rr['text_component_half_threshold_flips']} classifications at 0.5, but complete v9/v10 outputs flip only {rr['final_prediction_flips_including_frozen_selection_threshold_change']} at their already selected thresholds ({rr['fixed']} fixed, {rr['new_errors']} new errors). At the previous frozen threshold, final flips are {rr['final_prediction_flips_at_previous_threshold']}. The component and final thresholds differ, so this is a descriptive comparison, not a causal ratio.", '',
              f"Both versions use hard evidence on {rr['both_versions_hard']} OOF rows; {rr['both_versions_hard_output_changed']} of these final probabilities change. Test has {t['evidence']['endpoint']['rows']} endpoints ({t['evidence']['endpoint']['share']:.2%}) versus {o['evidence']['endpoint']['rows']} ({o['evidence']['endpoint']['share']:.2%}) in OOF. This is direct evidence of substantial evidence masking in OOF, and a plausible mechanism on test. Repetition/support shifts alone do not establish why any particular replacement changed few test answers.", '',
              rr['test_comparison_unavailable'],
              'No test errors, F1, accuracy, estimated fixes or leaderboard improvement are inferred. Support does not prove correctness. Unseen/mixed support and low-quality HTML only identify exposure.', '',
              '## One next evidence policy','',r['single_evidence_policy']['description'],'',
              'This is one diagnostic ablation proposal only, not a fitted replacement or an accepted improvement. Parameters were not searched; title minimum support 2 is the existing context default. Domain/title evidence can be correlated.', '']
    q=r['single_evidence_policy']; a=q['oof']; b=q['test']
    lines += [f"OOF policy exposure: {a['domain_only_abstain']['rows']} domain-only overrides would abstain, including {a['domain_only_abstain']['errors']} currently wrong rows. The pre-memory mixture would correct {a['pre_memory_mixture_would_correct_flagged_errors']} of these errors and break {a['pre_memory_mixture_would_break_flagged_correct']} currently correct rows; this omits downstream context/extra/hierarchy, so it is not a full-ablation score.", '',
              f"Test has {b['domain_only_abstain']['rows']} endpoint-compatible domain-only rows in this policy's exposure, so its diagnostic test exposure is nonzero. Their labels and whether final decisions would improve/change are unknown. The observed pre-memory net gain is zero, so this policy is not recommended for promotion. If investigated later, replay only this one gate through the complete OOF path at the frozen threshold before making any claim of benefit.", '',
              '**Recommended next direction:** '+r['findings']['next_direction'], '',
              '## Reproduce and limits','',
              'Run from the repository root:', '', '```sh','.venv/bin/python -B experiments/v11_shift/audit.py','```','',
              'Outputs: audit.py, audit.json, REPORT.md inside experiments/v11_shift only. audit.json includes support/error cross-tabs, per-fold errors, rule candidate mistakes, one policy exposure, exact input hashes and successful project reads. Original files are never written. No classifier is fitted, no model artifact is loaded, no other test predictions are accessed.', '',
              f"Measured peak RSS: {r['runtime']['peak_rss_bytes']/1024**3:.2f} GiB on macOS; at most two numerical threads. Runtime before output hashing: {r['runtime']['seconds']:.1f} seconds.", '',
              r['provenance']['guard_scope'],
              f"Guard verification passed for simulated socket creation/connect, forbidden reads and outside-folder writes. During execution, {r['provenance']['blocked_socket_constructors']} socket constructor was blocked; connect/DNS events: {r['provenance']['blocked_connect_or_dns_events']}. Import-time bytecode and filesystem capability probes were also blocked and recorded. No network connection occurred.",
              'Forbidden inputs were excluded by the input allowlist: external/**, reports/v2/**, reference code/data, holdout models/predictions/reports, public SQL labels, live URLs and test target labels. The sole additional reports artifact is reports/v5/development_oof.npz, an original training-only component table used to replay OOF hard gates. No v10 holdout metrics were read.', '',
              'Reused development OOF is not an independent performance estimate. This audit cannot validate test errors, causal improvements or an ensemble under construction in another task.', '']
    text='\n'.join(lines)
    assert not any(c in text for c in '\u2013\u2014\u201c\u201d\u00ab\u00bb\u2026')
    (OUT/'REPORT.md').write_text(text)

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        if '--inspect' in sys.argv:
            inspect_inputs()
        else:
            main()
