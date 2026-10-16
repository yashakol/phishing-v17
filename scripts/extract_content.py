"""Stream organizer CSVs once and cache bounded, reproducible HTML features."""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from phishing_url.content import extract_html, EXTRACTOR_VERSION, MAX_PARSE_CHARS
from phishing_url.data import load_train, load_test


def transform(item):
    url, content = item
    record, numeric = extract_html(url, content)
    return record, numeric, hashlib.sha256(content.encode("utf-8")).hexdigest()


def build(path, expected_urls, labels, output, workers):
    csv.field_size_limit(sys.maxsize)
    expected = set(expected_urls)
    by_url = {}
    duplicate_hash_conflicts = set()
    duplicates = 0
    started = time.perf_counter()
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected_header = ["url", "result", "content"] if labels is not None else ["url", "content"]
        if reader.fieldnames != expected_header:
            raise ValueError(f"Unexpected content CSV schema: {reader.fieldnames}")
        label_by_url = dict(zip(expected_urls, labels.tolist())) if labels is not None else {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while True:
                batch = []
                for _ in range(256):
                    row = next(reader, None)
                    if row is None:
                        break
                    rows += 1
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Malformed content record {rows}")
                    url = row["url"].strip()
                    if url not in expected:
                        # Conflicting train URLs were removed by the strict train loader.
                        if labels is None:
                            raise ValueError("Unexpected test URL")
                        continue
                    if labels is not None and int(row["result"]) != label_by_url[url]:
                        raise ValueError(f"Train/content label disagreement: {url}")
                    batch.append((url, row["content"]))
                if not batch and row is None:
                    break
                for item, value in zip(batch, pool.map(transform, batch, chunksize=16), strict=True):
                    url = item[0]
                    if url in by_url:
                        duplicates += 1
                        if by_url[url][2] != value[2]:
                            duplicate_hash_conflicts.add(url)
                        # Same deterministic first-record policy as load_train.
                        continue
                    by_url[url] = value
                if rows % 4096 < 256 or row is None:
                    print(json.dumps({"file": path.name, "read_rows": rows, "unique_kept": len(by_url),
                                      "seconds": round(time.perf_counter() - started, 1)}), flush=True)
                if row is None:
                    break
    missing = expected - set(by_url)
    if missing:
        raise ValueError(f"Missing HTML for {len(missing)} URLs")
    # A conflicting copy of HTML is ambiguous. Do not silently choose its content.
    if duplicate_hash_conflicts:
        raise ValueError(f"Different HTML for duplicate URLs: {len(duplicate_hash_conflicts)}")
    records = [by_url[url][0] for url in expected_urls]
    features = pd.DataFrame([by_url[url][1] for url in expected_urls])
    hashes = [by_url[url][2] for url in expected_urls]
    audit = {"path": str(path), "bytes": path.stat().st_size, "csv_rows": rows,
             "cached_rows": len(records), "unique_urls": len(expected), "duplicate_url_rows": duplicates,
             "different_html_duplicate_urls": len(duplicate_hash_conflicts), "extractor_version": EXTRACTOR_VERSION,
             "max_parse_chars": MAX_PARSE_CHARS, "parse_errors": int(features.html_parse_error.sum()),
             "truncated_html_rows": int(features.html_truncated.sum()),
             "empty_html_rows": int(features.html_empty.sum()),
             "unique_content_hashes": len(set(hashes)), "seconds": round(time.perf_counter() - started, 2)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump({"records": records, "features": features, "labels": labels,
                 "urls": expected_urls, "hashes": hashes, "source": audit}, temporary, compress=3)
    temporary.replace(output)
    print(json.dumps({"saved": str(output), "cache_bytes": output.stat().st_size, **audit}), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = ROOT / "reports/v3"
    out.mkdir(parents=True, exist_ok=True)
    urls, y, train_audit = load_train(args.archive / "train.csv", out / "archive_train")
    test_urls = load_test(args.archive / "test.csv")
    sources = {}
    for name, us, labels in (("train", urls, y), ("test", test_urls, None)):
        sources[name] = build(args.archive / f"{name}_with_content.csv", us, labels,
                              ROOT / f"data/content_{name}.joblib", args.workers)
    (out / "content_sources.json").write_text(json.dumps({"sources": sources, "train_audit": train_audit},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
