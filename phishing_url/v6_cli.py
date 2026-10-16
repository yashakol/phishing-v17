"""Generate a submission from the saved learned extra-HTML ensemble offline."""

import argparse
import json

import joblib
import numpy as np

from .data import write_submission
from .v6_bundle import predict_v6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "cache", "extra-cache", "url-features", "sample", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cache, extra, url = [joblib.load(path) for path in (args.cache, args.extra_cache, args.url_features)]
    if cache["urls"] != url["urls"]:
        raise ValueError("URL feature order differs from the content cache")
    bundle = joblib.load(args.model)
    p, hits = predict_v6(bundle, cache, extra, url["features"])
    info = write_submission(args.sample, args.output, (p >= bundle["threshold"]).astype(np.int8))
    info.update(training_rule_matches=int(hits["training_match"].sum()), known_training_urls=int(hits["known_url"].sum()))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
