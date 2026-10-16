"""Choose content ensemble weights/thresholds from independent OOF predictions."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from phishing_url.content_memory import fit_memory, apply_memory, text_fingerprint
from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.evaluation import best_threshold, measure
from phishing_url.features import group_for_url


def main():
    numeric = np.load(ROOT / "experiments/content/development_oof.npz")
    text = np.load(ROOT / "experiments/html_astra/development_oof.npz")
    # Accept both explicit cache_indices and the common indices name.
    text_indices = text["cache_indices"] if "cache_indices" in text.files else text["indices"]
    text_labels = text["labels"] if "labels" in text.files else text["y"]
    if not np.array_equal(numeric["indices"], text_indices) or not np.array_equal(numeric["labels"], text_labels):
        raise ValueError("Text/numeric OOF alignment differs")
    cache = joblib.load(ROOT / "data/content_train.joblib")
    dev = aligned_splits(ROOT, cache)["development"]
    if not np.array_equal(dev, numeric["indices"]):
        raise ValueError("OOF predictions do not match the protected development partition")
    labels = numeric["labels"]
    original_oof = np.load(ROOT / "reports/development_oof.npz")
    rich_oof = np.load(ROOT / "experiments/astra/development_oof.npz")
    old_dev = np.load(ROOT / "reports/split_indices.npz")["development"]
    if not np.array_equal(rich_oof["indices"], old_dev) or not np.array_equal(original_oof["labels"], labels):
        raise ValueError("Original URL OOF alignment differs")
    url_probability = .5 * rich_oof["probabilities"] + .5 * original_oof["boost"]
    memory_predictions = {}
    fingerprints = [text_fingerprint(r) for r in cache["records"]]
    domains = [group_for_url(u) for u in cache["urls"]]
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=2026).split(dev, labels))
    for support in (0, 2, 3, 5):
        for domain_support in (0, 5, 10, 20):
            if not support and not domain_support:
                continue
            values = {kind: np.full(len(dev), -1, dtype=np.int8) for kind in ("raw", "text", "domain")}
            for fold, (tr, va) in enumerate(cv):
                memory = fit_memory(
                    records_at(cache, dev[tr]), [cache["hashes"][int(i)] for i in dev[tr]], labels[tr],
                    support, domain_support, [fingerprints[int(i)] for i in dev[tr]])
                for kind, keys in (("raw", cache["hashes"]), ("text", fingerprints), ("domain", domains)):
                    values[kind][va] = np.fromiter((memory.get((kind, keys[int(i)]), -1) for i in dev[va]), dtype=np.int8)
            memory_predictions[(support, domain_support)] = values
    candidates = []
    candidate_probabilities = {}
    for weight in (0., .15, .3, .5, .7, .85, 1.):
        html = weight * text["probabilities"] + (1-weight) * numeric["probabilities"]
        for url_weight in (0., .1, .2, .3):
            base = (1-url_weight) * html + url_weight * url_probability
            for support in (0, 2, 3, 5):
                for domain_support in (0, 5, 10, 20):
                    p = base.copy()
                    if support or domain_support:
                        values = memory_predictions[(support, domain_support)]
                        raw, txt, dom = values["raw"], values["text"], values["domain"]
                        page = np.where(raw >= 0, raw, txt)
                        page_present = page >= 0
                        domain_eligible = (dom >= 0) & (page_present | ((base >= .05) & (base <= .95)))
                        conflict = ((raw >= 0) & (txt >= 0) & (raw != txt)) | (domain_eligible & page_present & (dom != page))
                        take_page = page_present & ~conflict
                        take_domain = domain_eligible & ~page_present & ~conflict
                        p[take_page] = page[take_page]
                        p[take_domain] = dom[take_domain]
                    threshold, _ = best_threshold(labels, p)
                    result = {"text_weight": weight, "aux_url_weight": url_weight,
                              "memory_minimum_support": support, "domain_minimum_support": domain_support,
                              **measure(labels, p, threshold)}
                    candidates.append(result)
                    candidate_probabilities[(weight, url_weight, support, domain_support)] = p
    selected = max(candidates, key=lambda row: (row["f1"], row["memory_minimum_support"] == 0,
                                               row["domain_minimum_support"] == 0,
                                               -row["aux_url_weight"],
                                               -abs(row["text_weight"]-.5)))
    output = ROOT / "reports/v3"
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "ensemble_selection.json.tmp"
    temporary.write_text(json.dumps({
        "selected": selected, "candidates": sorted(candidates, key=lambda r:r["f1"], reverse=True),
        "protocol": "Original three-fold development OOF; vocabularies, models and page/domain memory fitted on each fold's training URLs only; no holdout/test labels used."}, indent=2))
    temporary.replace(output / "ensemble_selection.json")
    p = candidate_probabilities[(selected["text_weight"], selected["aux_url_weight"], selected["memory_minimum_support"], selected["domain_minimum_support"])]
    np.savez_compressed(output / "development_oof.npz", indices=dev, labels=labels, probabilities=p,
                        text_probabilities=text["probabilities"], numeric_probabilities=numeric["probabilities"],
                        url_probabilities=url_probability)
    print("Selected content ensemble:", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
