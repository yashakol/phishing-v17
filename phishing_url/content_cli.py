"""Reproduce a submission from saved static content, without network access."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .content_bundle import predict_content
from .data import write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--url-features", type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    cache = joblib.load(args.cache)
    frame = None
    if args.url_features:
        url_cache = joblib.load(args.url_features)
        if url_cache["urls"] != cache["urls"]:
            raise ValueError("URL feature order differs")
        frame = url_cache["features"]
    probabilities, hits = predict_content(bundle, cache, frame)
    report = write_submission(args.sample, args.output, (probabilities >= bundle["threshold"]).astype(np.int8))
    print(json.dumps({**report, "training_rule_matches": int(hits["training_match"].sum()),
                     "known_training_urls": int(hits["known_url"].sum())}))


if __name__ == "__main__":
    main()
