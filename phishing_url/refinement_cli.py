"""Offline CSV prediction from the fitted content refinement bundle."""

import argparse
import json

import joblib
import numpy as np

from .data import write_submission
from .refined_bundle import predict_refined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--url-features',required=True)
    parser.add_argument('--sample',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cache=joblib.load(args.cache)
    url=joblib.load(args.url_features)
    if cache['urls']!=url['urls']:
        raise ValueError('Content and URL feature order differs')
    bundle=joblib.load(args.model)
    p,hits=predict_refined(bundle,cache,url['features'])
    report=write_submission(args.sample,args.output,(p>=bundle['threshold']).astype(np.int8))
    report['refinement_matches']=int(hits['refinement_match'].sum())
    report['known_training_urls']=int(hits['known_url'].sum())
    print(json.dumps(report))


if __name__=='__main__':
    main()
