"""Read CSV strictly; reject damaged training records without guessing labels."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


def valid_url(url):
    try:
        parsed = urlsplit(url.strip())
        host = parsed.hostname or ""
        return (parsed.scheme.lower() in {"http", "https"} and bool(host)
                and not host.startswith(".") and not any(c in url for c in "<>\r\n"))
    except ValueError:
        return False


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_train(path, report_dir=None):
    accepted = []
    rejected = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["url", "result"]:
            raise ValueError("Expected train header: url,result")
        for record, row in enumerate(reader, start=1):
            line = reader.line_num
            if len(row) != 2:
                reason = "wrong_column_count"
            elif row[1] not in {"0", "1"}:
                reason = "invalid_label"
            elif not valid_url(row[0]):
                reason = "invalid_url"
            else:
                accepted.append((row[0].strip(), int(row[1]), line))
                continue
            rejected.append({"record": record, "physical_line": line,
                             "reason": reason, "raw_fields": json.dumps(row, ensure_ascii=False)})
    by_url = defaultdict(list)
    for url, label, line in accepted:
        by_url[url].append((label, line))
    conflicts = {url for url, records in by_url.items() if len({r[0] for r in records}) > 1}
    clean = [(url, records[0][0]) for url, records in by_url.items() if url not in conflicts]
    duplicate_rows = sum(len(records) - 1 for url, records in by_url.items() if url not in conflicts)
    audit = {
        "input_sha256": sha256(path), "parsed_records": len(accepted) + len(rejected),
        "accepted_before_deduplication": len(accepted), "rejected_records": len(rejected),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "duplicate_rows_removed": duplicate_rows, "conflicting_urls_removed": len(conflicts),
        "conflicting_records_removed": sum(len(by_url[url]) for url in conflicts),
        "clean_unique_rows": len(clean), "label_counts": dict(Counter(str(label) for _, label in clean)),
        "rejected_physical_line_min": min((r["physical_line"] for r in rejected), default=None),
        "rejected_physical_line_max": max((r["physical_line"] for r in rejected), default=None),
    }
    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        with (report_dir / "rejected_train_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record", "physical_line", "reason", "raw_fields"])
            writer.writeheader()
            writer.writerows(rejected)
        with (report_dir / "clean_train.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["url", "result"])
            writer.writerows(clean)
    return [row[0] for row in clean], np.array([row[1] for row in clean], dtype=np.int8), audit


def load_test(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["url"]:
            raise ValueError("Expected test header: url")
        urls = []
        for row in reader:
            if len(row) != 1 or not valid_url(row[0]):
                raise ValueError(f"Invalid test record at physical line {reader.line_num}; cannot discard test rows")
            urls.append(row[0].strip())
    return urls


def write_submission(sample_path, output_path, predictions):
    with Path(sample_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError("Expected sample header: Id,Predicted")
        rows = list(reader)
    if len(rows) != len(predictions):
        raise ValueError("Sample and prediction row counts differ")
    if [row["Id"] for row in rows] != [str(i) for i in range(len(rows))]:
        raise ValueError("Sample IDs must be consecutive and match test order")
    predictions = np.asarray(predictions)
    if not set(predictions.tolist()).issubset({0, 1}):
        raise ValueError("Predictions must be binary")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Id", "Predicted"])
        writer.writerows((row["Id"], int(label)) for row, label in zip(rows, predictions, strict=True))
    return {"rows": len(rows), "positive_predictions": int(predictions.sum()), "sha256": sha256(path)}
