"""Verify and restore Git-tracked assets using only the Python standard library."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_name(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Asset path must be a nonempty POSIX relative path")
    name = PurePosixPath(value)
    if name.is_absolute() or ".." in name.parts or name.as_posix() != value or not name.parts:
        raise ValueError("Unsafe asset path: " + value)
    return Path(*name.parts)


def checked_path(base: Path, name: str) -> Path:
    path = base / relative_name(name)
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Asset path escapes its root")
    return path


def _record(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("Asset record must be an object")
    if not isinstance(value.get("bytes"), int) or isinstance(value["bytes"], bool) or value["bytes"] < 0:
        raise ValueError("Invalid asset size")
    if not isinstance(value.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError("Invalid asset SHA256")


def manifest(root: Path) -> dict:
    root = Path(root).resolve()
    document = json.loads((root / "payload/manifest.json").read_text())
    if document.get("version") != 1 or not isinstance(document.get("files"), list) or not document["files"]:
        raise ValueError("Unsupported payload manifest")
    seen = set()
    for item in document["files"]:
        relative_name(item["path"])
        if item["path"] in seen:
            raise ValueError("Duplicate logical asset path")
        seen.add(item["path"])
        _record(item)
        if not isinstance(item.get("parts"), list) or not item["parts"]:
            raise ValueError("Asset parts are required")
        for part in item["parts"]:
            relative_name(part["path"])
            _record(part)
        if sum(part["bytes"] for part in item["parts"]) != item["bytes"]:
            raise ValueError("Asset part sizes do not cover the file")
    return document


def restore(root: Path) -> dict:
    root = Path(root).resolve()
    source = root / "payload"
    runtime = root / ".runtime"
    if runtime.is_symlink():
        raise ValueError("Runtime directory must not be a symlink")
    runtime.mkdir(exist_ok=True)
    document = manifest(root)
    parts_checked, restored, total = 0, 0, 0
    for item in document["files"]:
        target = checked_path(runtime, item["path"])
        parts = []
        for part in item["parts"]:
            file = checked_path(source, part["path"])
            if file.stat().st_size != part["bytes"] or digest(file) != part["sha256"]:
                raise ValueError("Payload part differs: " + part["path"])
            parts.append(file)
            parts_checked += 1
        total += item["bytes"]
        if target.is_file() and target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # The root may already contain cached files; parent symlinks remain forbidden.
        target = checked_path(runtime, item["path"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restore-", delete=False) as stream:
                temporary = Path(stream.name)
                for file in parts:
                    with file.open("rb") as incoming:
                        while block := incoming.read(1024 * 1024):
                            stream.write(block)
            if temporary.stat().st_size != item["bytes"] or digest(temporary) != item["sha256"]:
                raise ValueError("Restored asset differs: " + item["path"])
            os.replace(temporary, target)
            restored += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"status": "passed", "files": len(document["files"]), "parts_checked": parts_checked,
            "bytes": total, "restored_files": restored}


def verify_sources(root: Path) -> int:
    document = json.loads((root / "SOURCE_SNAPSHOT.json").read_text())
    for name, item in document["files"].items():
        if digest(checked_path(root, name)) != item["sha256"]:
            raise ValueError("Historical source differs: " + name)
    return len(document["files"])


def check_csv(path: Path, rows: int, expected_sha256: str | None = None) -> dict:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError("Submission header must be Id,Predicted")
        count, positive = 0, 0
        for index, row in enumerate(reader):
            if row != {"Id": str(index), "Predicted": row.get("Predicted")} or row["Predicted"] not in {"0", "1"}:
                raise ValueError("Invalid submission row or order")
            positive += int(row["Predicted"])
            count += 1
    if count != rows:
        raise ValueError("Submission row count differs")
    actual = digest(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Submission SHA256 differs from the delivered file")
    return {"rows": count, "positive_predictions": positive, "sha256": actual}
