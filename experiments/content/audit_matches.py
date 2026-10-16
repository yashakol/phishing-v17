"""Content duplicates and label conflicts, without using test labels."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib

from phishing_url.content_validation import aligned_splits, records_at
from phishing_url.content_memory import fit_memory, apply_memory, text_fingerprint


def main():
    train = joblib.load(ROOT / "data/content_train.joblib")
    splits = aligned_splits(ROOT, train)
    result = {}
    for name, keys in (("raw", train["hashes"]),
                       ("text", [text_fingerprint(r) for r in train["records"]])):
        counts = defaultdict(Counter)
        for key, label in zip(keys, train["labels"], strict=True):
            counts[key][int(label)] += 1
        result[name] = {"unique_pages": len(counts),
                        "repeated_pages": sum(sum(c.values()) > 1 for c in counts.values()),
                        "ambiguous_pages": sum(len(c) > 1 for c in counts.values()),
                        "rows_in_ambiguous_pages": sum(sum(c.values()) for c in counts.values() if len(c) > 1)}
    tr, va = splits["inner_train"], splits["inner_validation"]
    memory = fit_memory(records_at(train, tr), [train["hashes"][int(i)] for i in tr], train["labels"][tr])
    probability, hits = apply_memory(memory, records_at(train, va),
                                    [train["hashes"][int(i)] for i in va], [.5] * len(va))
    result["inner_validation_matches"] = {"rows": len(va), "matches": int(hits.sum()),
                                           "incorrect_matches": int((probability[hits] != train["labels"][va][hits]).sum())}
    (ROOT / "reports/v3/content_match_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
