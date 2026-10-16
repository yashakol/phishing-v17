import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent

def read(p):return json.loads((OUT/p).read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
s=read('selection.json');inner=read('inner_results.json');v=read('verification.json');e=read('execution.json')
lines=['Bounded FullVocabularyTextModel regularization ablation','',
       'Decision: '+s['status'],
       f"Inner winner: C={s['chosen_c']}; mixture errors={s['winner']['mix']['errors']}; baseline C=16 mixture errors=51.",
       'Meaningful advance gate: at least two fewer errors; <=49. '+('PASSED.' if s['gate_passed'] else 'FAILED. No OOF was generated.'),'',
       'Tuned inner results (6118 rows; maximum F1 cutoff, ties nearest 0.5):',
       'C | standalone errors | standalone F1 | standalone cutoff | 0.5 text + 0.5 Cat errors | mixture F1 | mixture cutoff']
for row in [dict(c=16,**inner['baseline'])]+inner['candidates']:
    a,b=row['standalone'],row['mix']
    lines.append(f"{row['c']} | {a['errors']} | {a['f1']:.12f} | {a['threshold']:.16g} | {b['errors']} | {b['f1']:.12f} | {b['threshold']:.16g}")
lines += ['', 'Protocol and exact parameters:',
          'Preregistered C grid [4, 8, 32], existing C=16 baseline, fixed 0.5/0.5 mixture; no other tuning.',
          'Classifier: LogisticRegression(C=chosen C, solver=liblinear, max_iter=500, random_state=20261006); remaining parameters recorded in selection.json.',
          'word_features=140000, char_features=100000, markup_features=70000, url_features=90000; min_df=2, char_text_chars=2000, transform_batch_size=2048, threads=2.',
          'All classifier fits sequential under threadpool_limits(2). No vectorizer fits.',
          'Original training-only v7 vocabulary and IDF reused after exact data/source/ordered-training guard matching.',
          'Training rows=34663; validation rows=6118. Models are independent deep copies with estimator c and classifier C consistent.',
          '', 'Validation:',
          'Independent verification implementation recomputed aligned indices, labels, threshold selection and gate.',
          'All saved models: finite, changed coefficients; 32 real validation predictions identical before/after serialization.',
          'Independent manual sigmoid(X @ coef + intercept) agrees with saved probabilities on 32 real validation rows per model.',
          'Full vectorizer vocabularies and IDF match guarded originals exactly; clone retains chosen parameters and no fitted state.',
          'Inner word IDF independently recounted for 16 terms on all 34663 training records.',
          'Original sources and input hashes checked again at completion. Raw sparse matrices were never saved.',
          '', 'Recovery and numerical limitations:',
          'The first run stopped after C=4 because whole-object hashing included sklearn process-local _stop_words_id.',
          'Diagnosis confirmed only this analyzer cache changed; vocabularies and IDF were unchanged.',
          'C=4 checkpoint reused with exact alignment/hash checks and full validation probability replay. It was NOT fitted again.',
          'The two in-memory inner matrices were lost at process exit and reconstructed once: two full training transforms and two full validation transforms in total, plus small 32-row verification transforms.',
          'This is an explicit deviation from the transform-once execution requirement. It does not change the three-fit grid or fit/selection scopes.',
          'run.py and run.log preserve the initial attempt; resume.py, resume.log, recovery.json and recovery_diagnosis.json document recovery.',
          'Original C=16 used float32 fit_transform. Refits used transform with frozen train-fitted representations; numerical equivalence to fitting from scratch is not claimed or measured.',
          'The original C=16 validation probabilities were reproduced exactly from its guarded estimator on the transformed validation matrix.',
          'clone(model).fit(new_training_records, y) will learn fresh train-only vocabularies/IDF with the chosen C and may differ numerically from these frozen-representation refits.',
          'Inner results are tuning evidence, not an independent generalization estimate.',
          '', 'Data boundary:',
          'Only organizer content_train.joblib, existing split/alignment metadata, the authorized inner baselines, guarded v7 models/sources, and own outputs were used by execution.',
          'The monolithic 63900-row organizer training cache was loaded; aligned_splits performs its existing original-train label alignment. Only inner/development indices enter model fitting and scoring.',
          'No holdout/test prediction or label files, submissions, external data, network, or live URL visits.',
          'reports/v7/selection.json was read only for permitted integration context before execution.',
          '', 'Artifacts:', 'selection.json: exact selection, parameters, input/source/data hashes and recovery disclosure.',
          'inner_results.json: all three candidates and C=16 baseline.',
          'inner_c4.npz, inner_c8.npz, inner_c32.npz: aligned train/validation indices, validation labels and text probabilities.',
          'models/ and guards/: saved independent estimators and exact training/provenance guards.',
          'verification.json: independent checks; execution.json: execution read audit.']
if s['gate_passed']:
    oof=read('oof_results.json')
    lines += ['development_oof.npz: pure text probabilities with original development indices, labels and original threefold assignments.',
              'OOF path: '+s['oof_path'], 'OOF standalone fixed-inner-cutoff F1: '+str(oof['standalone_at_frozen_inner_threshold']['f1']),
              'OOF was not used to retune C, thresholds or weights. Inner tuning overlaps development, so OOF is not nested evaluation.']
else: lines += ['Pure text OOF path: none (gate failed). Keep C=16 baseline; this branch does not justify promotion.']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(dict(files_sha256=files),indent=2,sort_keys=True)+'\n')
print('\n'.join(lines[:13]))
