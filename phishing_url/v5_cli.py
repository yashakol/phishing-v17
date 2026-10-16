"""Create an offline CSV submission from the fitted complementary ensemble."""

import argparse
import json

import joblib
import numpy as np

from .data import write_submission
from .v5_bundle import predict_v5


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','cache','url-features','sample','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    cache=joblib.load(args.cache)
    url=joblib.load(args.url_features)
    if cache['urls']!=url['urls']:
        raise ValueError('Content and URL feature order differs')
    bundle=joblib.load(args.model)
    p,hits=predict_v5(bundle,cache,url['features'])
    report=write_submission(args.sample,args.output,(p>=bundle['threshold']).astype(np.int8))
    report['training_rule_matches']=int(hits['training_match'].sum())
    report['known_training_urls']=int(hits['known_url'].sum())
    print(json.dumps(report))


if __name__=='__main__':
    main()
