"""Offline inference for a saved own-trained specialist-pair ensemble."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import load_test, write_submission
from .v14_bundle import predict_v14


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--content-cache", type=Path, required=True)
    parser.add_argument("--extra-cache", type=Path, required=True)
    parser.add_argument("--url-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    cache, extra, url = [joblib.load(path) for path in
                         (args.content_cache, args.extra_cache, args.url_cache)]
    if cache["labels"] is not None or extra["labels"] is not None:
        raise ValueError("Prediction caches must have no labels")
    if cache["urls"] != extra["urls"] or cache["urls"] != url["urls"] or \
            cache["urls"] != load_test(args.test):
        raise ValueError("Prediction inputs differ in URL order")
    with threadpool_limits(limits=2):
        p, hits = predict_v14(model, cache, extra, url["features"])
    info = write_submission(args.sample, args.output, (p >= model["threshold"]).astype(np.int8))
    info.update(known_training_urls=int(hits["known_url"].sum()),
                training_rule_matches=int(hits["training_match"].sum()),
                url_routed_rows=int(hits["url_routed"].sum()))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
