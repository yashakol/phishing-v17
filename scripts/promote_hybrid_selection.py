"""Freeze the positive development OOF hybrid before holdout inference."""

import json
import shutil
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/v5'
    selected=json.loads((out/'hybrid_selection.json').read_text())
    assert selected['selected']['metrics']['f1']>selected['baseline_v3']['f1']
    actual=np.load(out/'hybrid_development_oof.npz')
    previous=np.load(ROOT/'reports/v3/development_oof.npz')
    np.testing.assert_array_equal(actual['indices'],previous['indices'])
    np.testing.assert_array_equal(actual['labels'],previous['labels'])
    (out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    shutil.copy2(out/'hybrid_development_oof.npz',out/'development_oof.npz')
    print('Positive hybrid OOF result frozen; no holdout or test labels loaded.')


if __name__=='__main__':
    main()
