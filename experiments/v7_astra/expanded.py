"""User-specified second candidate: double all vocabulary caps, unchanged C16.

Only a fixed half-text/half-CatBoost replacement is considered for selection.
OOF requires >=2 fewer inner errors AND higher inner F1 than the 52-error base.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.content_checkpoints import data_digest
from phishing_url.content_validation import aligned_splits
from phishing_url.evaluation import measure
from experiments.v7_astra import run as core

BASE_OUT = core.OUT
CONFIG = {**core.CONFIG, 'word_features':280000, 'char_features':200000,
          'markup_features':140000, 'url_features':180000}
core.CONFIG = CONFIG
core.OUT = BASE_OUT / 'expanded'


def sources():
    return {**core.provenance(), 'experiments/v7_astra/expanded.py':core.sha(__file__)}


def inner(cache,splits,digest):
    tr,va = splits['inner_train'],splits['inner_validation']
    y = cache['labels'][va]
    old = core.aligned_prediction('experiments/html_astra/inner_predictions.npz',va,y)
    cat = core.aligned_prediction('experiments/v5/cat_inner_1.npz',va,y)
    baseline = core.metrics(y,.5*old+.5*cat)
    nbase = baseline['tuned']['fp']+baseline['tuned']['fn']
    assert nbase == 52
    protocol = dict(parameters=CONFIG,text_weight=.5,cat_weight=.5,
                    threshold_selection='inner best_threshold F1 only',
                    gate='at least 2 fewer inner net errors and positive F1 gain over baseline',
                    baseline_errors=nbase,candidate_count=1,additional_weight_search=False,
                    sources=sources(),data_sha256=digest,
                    user_specified_candidate=True)
    core.write_json('preregistered.json',protocol)
    np.savez_compressed(core.OUT/'inner_split.npz',inner_train=tr,inner_validation=va)
    p,timing = core.fit_one(cache,tr,va,'inner_full_vocabulary',digest,sources())
    standalone = core.metrics(y,p)
    replacement = core.metrics(y,.5*p+.5*cat)
    net_gain = nbase-replacement['tuned']['fp']-replacement['tuned']['fn']
    f1_gain = replacement['tuned']['f1']-baseline['tuned']['f1']
    selection = dict(parameters=CONFIG,text_weight=.5,cat_weight=.5,
                     threshold=replacement['tuned']['threshold'],
                     standalone_threshold=standalone['tuned']['threshold'],
                     baseline_threshold=baseline['tuned']['threshold'],
                     old_text_threshold=core.metrics(y,old)['tuned']['threshold'],
                     inner_net_errors_gain=net_gain,inner_f1_gain=f1_gain,
                     run_oof=bool(net_gain>=2 and f1_gain>0),
                     sources=sources(),data_sha256=digest,selection_scope='inner only')
    core.write_json('selection.json',selection)
    result = dict(baseline=baseline,new_text=standalone,replacement=replacement,
                  selection=selection,timing=timing)
    core.write_json('inner_results.json',result)
    core.emit('expanded_inner_complete',**{k:v for k,v in result.items() if k!='timing'})


def oof(cache,splits,digest):
    selected = json.loads((core.OUT/'selection.json').read_text())
    assert selected['sources']==sources() and selected['data_sha256']==digest
    assert selected['parameters']==CONFIG
    if not selected['run_oof']:
        core.emit('expanded_oof_skipped',reason='Inner gate of >=2 net errors and positive F1 gain failed')
        return
    indices = splits['development']
    assert len(indices)==50977
    y = cache['labels'][indices]
    p = np.full(len(indices),np.nan)
    folds = np.full(len(indices),-1,dtype=np.int8)
    counts = np.zeros(len(indices),dtype=np.int8)
    timings = []
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(indices,y)):
        fp,timing = core.fit_one(cache,indices[tr],indices[va],f'fold_{fold}',digest,sources())
        p[va],folds[va] = fp,fold
        counts[va] += 1
        timings.append(timing)
    assert np.all(counts==1) and np.isfinite(p).all()
    paths = ['experiments/html_astra/development_oof.npz','experiments/v5/cat_development_oof.npz']
    old,cat = [core.aligned_prediction(path,indices,y) for path in paths]
    for path in paths:
        with np.load(ROOT/path) as a:
            np.testing.assert_array_equal(a['folds'],folds)
    base,new = .5*old+.5*cat,.5*p+.5*cat
    report = dict(rows=len(indices),selection=selected,
                  standalone=measure(y,p,selected['standalone_threshold']),
                  standalone_at_05=measure(y,p),
                  baseline=measure(y,base,selected['baseline_threshold']),baseline_at_05=measure(y,base),
                  replacement=measure(y,new,selected['threshold']),replacement_at_05=measure(y,new),
                  limitation='Inner-selected parameters on overlapping original development; not nested evaluation. Exact v6 rule/context replacement is separate.',
                  measured_fit_seconds=sum(t['fit_seconds'] for t in timings),
                  measured_fold_total_seconds=sum(t['total_seconds'] for t in timings))
    np.savez_compressed(core.OUT/'development_oof.npz',indices=indices,labels=y,
                        probabilities=p,replacement=new,baseline=base,folds=folds)
    core.write_json('oof_results.json',report)
    core.emit('expanded_oof_complete',**report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['inner','oof'])
    args=parser.parse_args()
    sys.addaudithook(core.audit)
    core.OUT.mkdir(exist_ok=True)
    started=time.perf_counter()
    cache=joblib.load(ROOT/'data/content_train.joblib')
    assert len(cache['records'])==63900
    splits=aligned_splits(ROOT,cache)
    digest=data_digest(ROOT)
    (inner if args.stage=='inner' else oof)(cache,splits,digest)
    core.write_json(f'{args.stage}_execution.json',dict(seconds=time.perf_counter()-started,
                    project_files_read=sorted(core.READS)))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
