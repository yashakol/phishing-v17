"""Guard component reuse by exact training rows, data hash and parameters."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def data_digest(root):
    with (Path(root) / "data/content_train.joblib").open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def guard(cache, indices, config, digest):
    sample = hashlib.sha256()
    for i in indices:
        sample.update(cache["urls"][int(i)].encode("utf-8"))
        sample.update(b"\x00")
        sample.update(bytes([int(cache["labels"][int(i)])]))
    return {"training_rows": len(indices), "ordered_training_sha256": sample.hexdigest(),
            "data_sha256": digest, "config": json.loads(json.dumps(config, sort_keys=True))}


def save_component(path, model, scope):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump({"model": model, "guard": scope}, temporary, compress=3)
    temporary.replace(path)


def load_component(path, expected):
    stored = joblib.load(path)
    if stored["guard"] != expected:
        raise ValueError(f"Component cache scope differs: {path}")
    return stored["model"]
