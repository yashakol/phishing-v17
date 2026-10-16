"""Fit checked CatBoost components before reading any validation predictions."""

import argparse
import gc
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.content.boost import load_frame
from phishing_url.content_checkpoints import data_digest
from phishing_url.content_validation import aligned_splits
from scripts.train_v5 import complementary


def main(stage):
    cache=joblib.load(ROOT/'data/content_train.joblib')
    frame=load_frame(cache)
    splits=aligned_splits(ROOT,cache)
    indices=splits['development'] if stage=='holdout' else np.arange(len(cache['urls']))
    del cache['records'],cache['hashes'],cache['features']
    gc.collect()
    complementary(cache,frame,indices,'cat',stage,data_digest(ROOT))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['holdout','full'],required=True)
    with threadpool_limits(limits=2):
        main(parser.parse_args().stage)
