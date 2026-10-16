"""Package the verified new learned ensemble and training-only development records."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from package_v3 import included as included_v3


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def included(path):
    relative=path.relative_to(ROOT)
    if any(part in {'.venv','.git','__pycache__','.pytest_cache'} for part in relative.parts):
        return False
    if path.suffix in {'.pyc','.tmp'} or path.name=='.DS_Store':
        return False
    if relative.as_posix() in {'reports/v3/file_manifest.json','reports/v3/reproduced_submission.csv'}:
        return False
    if relative.as_posix() in {
        'experiments/html_astra/component_cache/holdout_text.joblib',
        'artifacts/v3/component_cache/holdout_boost.joblib',
        'artifacts/v4/component_cache/development_url_boost.joblib',
        'experiments/astra/development_model.joblib'}:
        return True
    if relative.parts[:2]==('experiments','v5'):
        if path.suffix=='.joblib':
            return path.name in {'development_memory.joblib','soft_keys.joblib'} or any(p.endswith('_oof') for p in relative.parts)
        return path.suffix in {'.py','.json','.npz'}
    if relative.parts[:2] in {('reports','v5'),('artifacts','v5')}:
        return 'component_cache' not in relative.parts and path.name!='cli_sample_cache.joblib' and path.name!='cli_sample_url.joblib'
    if relative.as_posix()=='submissions/submission_content_v5.csv':
        return True
    if relative.parts[0]=='scripts' and path.name in {
            'train_v5.py','verify_v5.py','report_v5.py','package_v5.py','verify_package_v5.py','setup_openmp.py',
            'promote_soft_selection.py','promote_hybrid_selection.py'}:
        return True
    if relative.as_posix() in {'README_V5.md','pytest.ini'}:
        return True
    return included_v3(path)


def main():
    verify=json.loads((ROOT/'reports/v5/verification.json').read_text())
    assert verify['offline_full_inference_byte_identical']
    assert json.loads((ROOT/'reports/v5/tests.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'reports/v5/cli_verification.json').read_text())['status']=='passed'
    manifest_path=ROOT/'reports/v5/file_manifest.json'
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p) and p!=manifest_path)
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(manifest_path)
    archive=ROOT.parent/'phishing-hackathon-solution-v5.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        for p in files:
            target.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert set(target.namelist())=={str(Path(ROOT.name)/p.relative_to(ROOT)) for p in files}
        assert not any('/external/' in name or 'with_content.csv' in name for name in target.namelist())
    checksum=digest(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files),
                      'sha256':checksum,'crc_ok':True}),flush=True)


if __name__=='__main__':
    main()
