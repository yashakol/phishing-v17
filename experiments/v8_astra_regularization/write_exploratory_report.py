"""Append a separate exploratory report/manifest; preserve original artifacts."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent

def read(name):return json.loads((OUT/name).read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
s=read('exploratory_oof_selection.json');v=read('exploratory_oof_verification.json');p=read('exploratory_oof_protocol.json');e=read('exploratory_oof_execution.json')
assert v['original_gate_still_false'] and v['coverage_exactly_once'] and v['classifiers_total']==6
for name,h in p['old_artifacts_sha256'].items():assert sha(OUT/name)==h,name
lines=[
    'Exploratory C32 text OOF after failed original half-mixture gate','',
    'Status: generated and independently verified. Original selection.json gate_passed remains false.',
    'Reason: standalone two-error gain (C32 66 errors vs C16 68) despite unchanged 51-error half-Cat mixture.',
    'This is an explicitly authorized follow-up, not success under the original gate. Old reports, selections and protocols remain historical artifacts and were not edited.','',
    'Output: /Users/iashin/phishing-hackathon/experiments/v8_astra_regularization/development_oof.npz',
    'Fields: indices, labels, probabilities, folds. Pure text probabilities, no Cat or other model mixing.',
    'Rows: 50977 original development rows, in original aligned_splits development order.',
    'Fold assignment: StratifiedKFold(n_splits=3, shuffle=True, random_state=2026). Every row has exactly one out-of-fold prediction.','',
    'Parameters: FullVocabularyTextModel(c=32, min_df=2, word_features=140000, char_features=100000, markup_features=70000, url_features=90000, char_text_chars=2000, max_iter=500, threads=2, random_state=20261006, transform_batch_size=2048, verbose=True).',
    'Classifier: LogisticRegression(C=32, solver=liblinear, max_iter=500, random_state=20261006); all resolved defaults recorded in exploratory_oof_selection.json.',
    'Parameters path: exploratory_oof_selection.json, JSON pointer /parameters.',
    'Source estimator: phishing_url/v7_text.py. Training/reuse code and original sources are pinned by SHA256 in selection and per-model guards.','',
    'Representation and fitting:',
    'Each fold reused only its corresponding independently train-fitted experiments/v7_astra/models/fold_#.joblib vocabulary/IDF, after exact matching against the original source/data/ordered-training guard.',
    'Each full train and validation sparse matrix was transformed once, on a copy. Classifier fitting consumed only that fold training matrix and y_train.',
    'Three new classifier fits, sequential, threadpool_limits(2). Total branch budget: six fits (three original inner, three exploratory folds). No final/development-wide model was fitted.',
    'Original base parameters, vocabulary, IDF, coefficients, predictions and model file hashes remained unchanged. Only sklearn process-local _stop_words_id is excluded from semantic hashing.',
    'Original training used float32 fit_transform; these refits use transform with frozen vocabulary/IDF. Numerical identity with a fresh full fit is not claimed.',
    'Cloning a saved estimator retains C32 and the exact declared configuration without fitted state. A fresh clone.fit learns vocabulary/IDF from its new training rows.','',
    'Verification:',
    'Separate verification code reconstructed original fold assignments, training guards, source/data hashes and full OOF assembly.',
    'Every one of 50977 OOF probabilities equals the matching fold archive probability; labels/indices/folds match exactly.',
    'Each saved model passed bitwise before/after serialization replay on 32 real validation rows.',
    'The independent checker manually computed sigmoid(X @ coef + intercept) and replayed each estimator on 32 real validation rows per fold; it did not retransform all 50977 validation rows.',
    'All coefficients are finite, differ from C16, match the selected configuration and feature dimensions; all reused vocabularies/IDFs match their original guarded models.',
    'All original branch artifacts retain their original hashes. Raw sparse feature matrices were never saved.',
    'No holdout/test prediction or label files, external data, submissions, network or URL visits. The monolithic organizer training cache is loaded for existing split alignment; only development rows enter fits and predictions.','',
    'Selection boundaries:',
    'No OOF thresholds or mixture weights were tuned, and no half-Cat OOF mixture was evaluated.',
    'Parent will choose thresholds/weights in the full v7 mixture later and handle any full/dev fitting.',
    'This is post-inner exploratory output; inner tuning rows overlap development, so it is not nested generalization evaluation.','',
    'Artifacts:',
    'exploratory_oof_protocol.json: explicit reason, three-fit budget and old-artifact hashes, saved before fitting.',
    'exploratory_oof_selection.json: C32 parameters, sources, per-fold guards and exact assignment hashes, saved before fitting.',
    'models/exploratory_fold_0.joblib through exploratory_fold_2.joblib: guarded fold estimators.',
    'guards/exploratory_fold_#.json and exploratory_fold_#.npz: training scope and aligned validation predictions.',
    'exploratory_oof_execution.json and exploratory_oof_verification.json: fit/read audit and independent checks.',
    'exploratory_manifest.json: file SHA256 inventory for the separate follow-up.',
    '', 'OOF SHA256: '+e['oof_sha256']]
(OUT/'EXPLORATORY_REPORT.txt').write_text('\n'.join(lines)+'\n')
old=set(p['old_artifacts_sha256'])
new={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and str(f.relative_to(OUT)) not in old and f.name!='exploratory_manifest.json'}
(OUT/'exploratory_manifest.json').write_text(json.dumps(dict(files_sha256=new,original_files_unchanged=True),indent=2,sort_keys=True)+'\n')
print('Exploratory OOF report and manifest written; original artifacts preserved.')
