"""Revisit only organizer raw HTML; verify every selected row against the v3 cache."""

import argparse
import csv
import gc
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from phishing_url.extra_html import extract_extra, VERSION


def transform(item):
    url, content = item
    record, features = extract_extra(url, content)
    return record, features, hashlib.sha256(content.encode("utf-8")).hexdigest()


def build(part, archive, workers):
    output = ROOT / f"data/extra_html_{part}.joblib"
    if output.exists():
        print(json.dumps({"event": "extra_cache_exists", "part": part}), flush=True)
        return
    started = time.perf_counter()
    original = joblib.load(ROOT / f"data/content_{part}.joblib")
    urls, labels, hashes = original["urls"], original["labels"], original["hashes"]
    lookup = {url: i for i, url in enumerate(urls)}
    del original
    gc.collect()
    source = archive / f"{part}_with_content.csv"
    seen = {}
    read_rows = 0
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="", encoding="utf-8-sig") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        reader = csv.DictReader(handle)
        while True:
            batch = []
            for _ in range(256):
                row = next(reader, None)
                if row is None:
                    break
                read_rows += 1
                url = row["url"].strip()
                if url not in lookup or url in seen:
                    continue
                if labels is not None:
                    assert int(row["result"]) == int(labels[lookup[url]])
                batch.append((url, row["content"]))
            for (url, _), (record, features, actual_hash) in zip(batch, pool.map(transform, batch, chunksize=8), strict=True):
                assert hashes[lookup[url]] == actual_hash, f"HTML mismatch for {url}"
                if url not in seen:
                    seen[url] = (record, features)
            if read_rows % 8192 < 256 or row is None:
                print(json.dumps({"event": "extra_extract_progress", "part": part,
                                  "read_rows": read_rows, "unique_kept": len(seen),
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
            if row is None:
                break
    assert set(seen) == set(lookup)
    records = [seen[url][0] for url in urls]
    features = pd.DataFrame([seen[url][1] for url in urls])
    source_metadata = {"path": str(source), "source_bytes": source.stat().st_size,
                       "read_rows": read_rows, "rows": len(records), "extractor_version": VERSION,
                       "original_raw_hashes_verified": True, "seconds": time.perf_counter() - started}
    joblib.dump({"records": records, "features": features, "urls": urls, "labels": labels,
                 "source": source_metadata}, output, compress=3)
    (ROOT / f"reports/v6/extra_{part}_source.json").write_text(json.dumps(source_metadata, indent=2) + "\n")
    print(json.dumps({"event": "extra_extract_complete", "part": part, **source_metadata,
                      "cache_bytes": output.stat().st_size}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("/Users/iashin/Downloads/Архив 3 (1)"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--part", choices=("train", "test", "both"), default="both")
    args = parser.parse_args()
    for part in (("train", "test") if args.part == "both" else (args.part,)):
        build(part, args.archive, args.workers)
