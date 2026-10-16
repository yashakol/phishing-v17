"""Evaluate frozen HTML ensemble, then fit all uncorrupted organizer train rows."""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.content.boost import fit
from phishing_url.content import EXTRACTOR_VERSION
from phishing_url.content_bundle import prediction_frame, predict_content
from phishing_url.content_memory import fit_memory, text_fingerprint
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.content_checkpoints import guard, data_digest, save_component, load_component
from phishing_url.data import write_submission
from phishing_url.evaluation import measure
from phishing_url.features import group_for_url
from phishing_url.html_text import ContentTextModel
from phishing_url.models import fit_boost
from phishing_url.rich_text import URLModel

WAIT_FOR_TEXT_COMPONENTS = False


def subset(cache, indices):
    return {"urls": [cache["urls"][int(i)] for i in indices],
            "records": records_at(cache, indices),
            "hashes": [cache["hashes"][int(i)] for i in indices],
            "features": cache["features"].iloc[indices].reset_index(drop=True)}


def fit_bundle(cache, frame, indices, selection, text_config, numeric_config, stage):
    y = cache["labels"][indices]
    records = records_at(cache, indices)
    urls = [cache["urls"][int(i)] for i in indices]
    weight, support = selection["text_weight"], selection["memory_minimum_support"]
    digest = data_digest(ROOT)
    text = None
    if weight:
        text_path = ROOT / f"experiments/html_astra/component_cache/{stage}_text.joblib"
        text_scope = guard(cache, indices, text_config, digest)
        if WAIT_FOR_TEXT_COMPONENTS and not text_path.exists():
            print("Waiting for the separately fitted text component:", stage, flush=True)
            deadline = time.monotonic() + 3600
            while not text_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Text component is not ready: {text_path}")
                time.sleep(10)
        if text_path.exists():
            text = load_component(text_path, text_scope)
            print("Reusing checked text component:", stage, flush=True)
        else:
            text = ContentTextModel(**text_config).fit(records, y)
            save_component(text_path, text, text_scope)
    boost = None
    if weight < 1:
        boost_path = ROOT / f"artifacts/v3/component_cache/{stage}_boost.joblib"
        boost_scope = guard(cache, indices, numeric_config, digest)
        if boost_path.exists():
            boost = load_component(boost_path, boost_scope)
            print("Reusing checked numeric component:", stage, flush=True)
        else:
            boost = fit(frame.iloc[indices], y, numeric_config)
            save_component(boost_path, boost, boost_scope)
    domain_support = selection.get("domain_minimum_support", 0)
    memory = fit_memory(records, [cache["hashes"][int(i)] for i in indices], y, support, domain_support) if support or domain_support else {}
    auxiliary_weight = selection.get("aux_url_weight", 0.)
    auxiliary_text = URLModel(representation="rich", c=16, nb=False).fit(urls, y) if auxiliary_weight else None
    # The original URL feature columns precede the HTML columns.
    html_columns = set(cache["features"].columns)
    url_frame = frame[[column for column in frame if column not in html_columns]]
    auxiliary_boost = fit_boost(url_frame.iloc[indices], y, threads=4) if auxiliary_weight else None
    return {"text": text, "boost": boost, "memory": memory,
            "text_weight": weight, "threshold": selection["threshold"],
            "aux_url_weight": auxiliary_weight, "aux_url_text": auxiliary_text,
            "aux_url_boost": auxiliary_boost,
            "memory_minimum_support": support, "known_urls": dict(zip(urls, y.tolist())),
            "domain_minimum_support": domain_support,
            "feature_columns": list(frame.columns), "extractor_version": EXTRACTOR_VERSION,
            "text_config": text_config, "numeric_config": numeric_config, "training_rows": len(indices)}


def evaluate(cache, frame, url_frame, training, validation, config, name):
    print("Frozen content validation:", name, len(training), len(validation), flush=True)
    start = time.perf_counter()
    bundle = fit_bundle(cache, frame, training, config["selected"], config["text_parameters"], config["numeric_parameters"], name)
    val = subset(cache, validation)
    p, hits = predict_content(bundle, val, url_frame.iloc[validation])
    metrics = measure(cache["labels"][validation], p, bundle["threshold"])
    raw_seen = set(cache["hashes"][int(i)] for i in training)
    text_seen = {text_fingerprint(cache["records"][int(i)]) for i in training}
    novel = np.asarray([raw_hash not in raw_seen and text_fingerprint(record) not in text_seen
                        for record, raw_hash in zip(val["records"], val["hashes"], strict=True)])
    record = {"metrics": metrics, "training_rows": len(training),
              "training_rule_matches": int(hits["training_match"].sum()),
              "known_training_url_matches": int(hits["known_url"].sum()),
              "raw_html_seen_rows": sum(h in raw_seen for h in val["hashes"]),
              "unseen_content_rows": int(novel.sum()), "seconds": time.perf_counter()-start}
    if len(np.unique(cache["labels"][validation][novel])) == 2:
        record["unseen_content_metrics"] = measure(cache["labels"][validation][novel], p[novel], bundle["threshold"])
    pd.DataFrame({"cache_index": validation, "url": val["urls"], "label": cache["labels"][validation],
                  "probability": p, "prediction": p >= bundle["threshold"],
                  "training_rule_match": hits["training_match"], "unseen_content": novel}).to_csv(
                      ROOT / f"reports/v3/{name}_predictions.csv", index=False)
    print("Content validation result:", name, record, flush=True)
    del bundle
    gc.collect()
    return record


def main():
    started = time.perf_counter()
    reports, artifacts = ROOT / "reports/v3", ROOT / "artifacts/v3"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    selection = json.loads((reports / "ensemble_selection.json").read_text())["selected"]
    text = json.loads((ROOT / "experiments/html_astra/selection.json").read_text())["model_parameters"]
    numeric = json.loads((ROOT / "experiments/content/selection.json").read_text())["selected"]["config"]
    config = {"selected": selection, "text_parameters": text, "numeric_parameters": numeric,
              "protocol": "Frozen on original development only; old holdout reused solely for comparison; new 178 URLs used only in final fit."}
    (artifacts / "config.json").write_text(json.dumps(config, indent=2))
    cache = joblib.load(ROOT / "data/content_train.joblib")
    url_cache = joblib.load(ROOT / "data/content_url_train.joblib")
    if cache["urls"] != url_cache["urls"]:
        raise ValueError("URL/HTML training order differs")
    url_frame = url_cache["features"]
    frame = prediction_frame(cache, url_frame)
    splits = aligned_splits(ROOT, cache)
    metrics = {"config": config, "development_oof": selection,
               "same_holdout_note": "Previously used v1/v2 holdout, not a fresh independently sampled test."}
    metrics["holdout"] = evaluate(cache, frame, url_frame, splits["development"], splits["holdout"], config, "holdout")
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    group = np.load(ROOT / "reports/domain_split_indices.npz")
    tr, va = splits["old_to_cache"][group["training"]], splits["old_to_cache"][group["validation"]]
    training_domains = {group_for_url(cache["urls"][int(i)]) for i in tr}
    validation_domains = {group_for_url(cache["urls"][int(i)]) for i in va}
    if training_domains & validation_domains:
        raise ValueError("Domain validation partitions overlap")
    metrics["domain_disjoint"] = evaluate(cache, frame, url_frame, tr, va, config, "domain")
    metrics["domain_disjoint"]["domain_overlap"] = 0
    metrics["domain_disjoint_note"] = "Different registered domains; identical page templates across domains may remain. Novel-content subset is reported separately." 
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Training final content model:", len(cache["urls"]), flush=True)
    bundle = fit_bundle(cache, frame, np.arange(len(cache["urls"])), selection, text, numeric, "full")
    bundle["source_metadata"] = cache["source"]
    bundle["validation"] = metrics
    joblib.dump(bundle, artifacts / "content.joblib", compress=3)
    del cache, frame, url_frame, url_cache
    gc.collect()
    test_cache = joblib.load(ROOT / "data/content_test.joblib")
    test_url_cache = joblib.load(ROOT / "data/content_url_test.joblib")
    if test_cache["urls"] != test_url_cache["urls"]:
        raise ValueError("Test URL/HTML order differs")
    p, hits = predict_content(bundle, test_cache, test_url_cache["features"])
    submission = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_content_v3.csv",
                                  (p >= bundle["threshold"]).astype(np.int8))
    metrics["submission"] = {**submission, "training_rule_matches": int(hits["training_match"].sum()),
                             "known_training_urls": int(hits["known_url"].sum()),
                             "ordered_test_rows": len(test_cache["urls"])}
    metrics["total_seconds"] = time.perf_counter()-started
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"Id": range(len(p)), "url": test_cache["urls"], "probability": p,
                  "training_rule_match": hits["training_match"], "known_training_url": hits["known_url"]}).to_csv(
                      reports / "test_prediction_provenance.csv", index=False)
    print("Final content solution:", metrics, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-text-components", action="store_true")
    WAIT_FOR_TEXT_COMPONENTS = parser.parse_args().wait_for_text_components
    with threadpool_limits(limits=2):
        main()
