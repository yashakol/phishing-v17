"""Delivery checks: corruption, traversal, atomic restoration and CSV order."""
from pathlib import Path
import hashlib
import json

import pytest

from delivery_assets import check_csv, manifest, relative_name, restore


def fixture_payload(root):
    (root / "payload/parts").mkdir(parents=True)
    data = b"two " + b"chunks"
    parts = []
    for i, block in enumerate([data[:4], data[4:]]):
        path = f"parts/{i}.bin"
        (root / "payload" / path).write_bytes(block)
        parts.append({"path": path, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    value = {"version": 1, "files": [{"path": "data/a.bin", "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}]}
    (root / "payload/manifest.json").write_text(json.dumps(value))
    return value, data


def test_exact_restore_and_repeat(tmp_path):
    _, data = fixture_payload(tmp_path)
    assert restore(tmp_path)["restored_files"] == 1
    assert (tmp_path / ".runtime/data/a.bin").read_bytes() == data
    assert restore(tmp_path)["restored_files"] == 0


def test_corrupt_part_rejected_before_replacing_valid_runtime(tmp_path):
    fixture_payload(tmp_path)
    restore(tmp_path)
    before = (tmp_path / ".runtime/data/a.bin").read_bytes()
    (tmp_path / "payload/parts/0.bin").write_bytes(b"bad!")
    with pytest.raises(ValueError, match="part differs"):
        restore(tmp_path)
    assert (tmp_path / ".runtime/data/a.bin").read_bytes() == before


def test_missing_part_rejected(tmp_path):
    fixture_payload(tmp_path)
    (tmp_path / "payload/parts/0.bin").unlink()
    with pytest.raises(FileNotFoundError):
        restore(tmp_path)


@pytest.mark.parametrize("path", ["../outside", "/outside", "a/../../outside", "a\\outside", "", "."])
def test_unsafe_names_rejected(path):
    with pytest.raises(ValueError):
        relative_name(path)


def test_duplicate_logical_asset_rejected(tmp_path):
    value, _ = fixture_payload(tmp_path)
    value["files"].append(value["files"][0])
    (tmp_path / "payload/manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Duplicate"):
        manifest(tmp_path)


def test_runtime_escape_through_symlink_rejected(tmp_path):
    fixture_payload(tmp_path)
    outside = tmp_path / "outside";outside.mkdir()
    runtime = tmp_path / ".runtime";runtime.mkdir()
    (runtime / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        restore(tmp_path)


def test_submission_exact_bytes_and_order(tmp_path):
    file = tmp_path / "submission.csv";file.write_bytes(b"Id,Predicted\r\n0,1\r\n1,0\r\n")
    result = check_csv(file, 2, hashlib.sha256(file.read_bytes()).hexdigest())
    assert result["positive_predictions"] == 1
    file.write_text("Id,Predicted\n1,1\n0,0\n")
    with pytest.raises(ValueError, match="order"):
        check_csv(file, 2)


@pytest.mark.parametrize("body", ["id,Predicted\n0,1\n", "Id,Predicted\n0,2\n", "Id,Predicted\n0,1,extra\n", "Id,Predicted\n"])
def test_invalid_submission_rejected(tmp_path, body):
    file = tmp_path / "submission.csv";file.write_text(body)
    with pytest.raises(ValueError):
        check_csv(file, 1)
