"""Freeze the positive development-only context result for the v5 training stage."""

import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/v5'
    selected=json.loads((out/'soft_selection.json').read_text())
    assert selected['selected']['metrics']['f1']>selected['baseline_v3']['f1']
    (out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    soft=np.load(out/'soft_development_oof.npz')
    old=np.load(ROOT/'reports/v3/development_oof.npz')
    np.testing.assert_array_equal(soft['indices'],old['indices'])
    np.testing.assert_array_equal(soft['labels'],old['labels'])
    np.savez_compressed(out/'development_oof.npz',indices=soft['indices'],labels=soft['labels'],
                        probabilities=soft['probabilities'],text=old['text_probabilities'],
                        numeric=old['numeric_probabilities'],url=old['url_probabilities'],
                        leaf=np.zeros(len(soft['labels'])),cat=np.zeros(len(soft['labels'])),
                        hashed=np.zeros(len(soft['labels'])))
    print('Positive development context result frozen; no holdout or test labels loaded.')


if __name__=='__main__':
    main()
