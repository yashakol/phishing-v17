"""Create the frozen routed submission from saved learned models, offline."""

import argparse
import json

import joblib
import numpy as np

from .data import write_submission
from .v9_routing import predict_v9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "cache", "extra-cache", "url-features", "sample", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    cache, extra, url = [joblib.load(p) for p in (args.cache, args.extra_cache, args.url_features)]
    if cache["urls"] != url["urls"]:
        raise ValueError("URL features have a different row order")
    model = joblib.load(args.model)
    p, hits = predict_v9(model, cache, extra, url["features"])
    result = write_submission(args.sample, args.output, (p >= model["threshold"]).astype(np.int8))
    result.update(known_training_urls=int(hits["known_url"].sum()),
        training_rule_matches=int(hits["training_match"].sum()), url_routed_rows=int(hits["url_routed"].sum()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
