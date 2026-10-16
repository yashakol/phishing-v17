"""Create a submission from saved v7 supervised models without network access."""

import argparse
import json

import joblib
import numpy as np

from .data import write_submission
from .v7_bundle import predict_v7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "cache", "extra-cache", "url-features", "sample", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cache, extra, url = [joblib.load(path) for path in (args.cache, args.extra_cache, args.url_features)]
    if cache["urls"] != url["urls"]:
        raise ValueError("URL feature order differs")
    bundle = joblib.load(args.model)
    probabilities, hits = predict_v7(bundle, cache, extra, url["features"])
    info = write_submission(args.sample, args.output, (probabilities >= bundle["threshold"]).astype(np.int8))
    info["known_training_urls"] = int(hits["known_url"].sum())
    info["training_rule_matches"] = int(hits["training_match"].sum())
    if "hierarchy_match" in hits:
        info["hierarchy_matches"] = int(hits["hierarchy_match"].sum())
    print(json.dumps(info))


if __name__ == "__main__":
    main()

