"""Package a verified v8 learned candidate, preserving the previous archive."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v7 import included as included_v7


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def included(path):
    rel = path.relative_to(ROOT)
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".tmp"} or path.name == ".DS_Store":
        return False
    if rel.as_posix() == "reports/v7/file_manifest.json":
        return False
    if rel.parts[:2] in {("reports", "v8"), ("artifacts", "v8")}:
        return path.suffix in {".json", ".npz", ".xml", ".csv", ".joblib"} and path.name != "reproduced_submission.csv"
    if rel.parts[:2] in {("experiments", "v8_ensemble"), ("experiments", "v8_astra_regularization"), ("experiments", "v8_resources")}:
        return path.suffix in {".py", ".json", ".npz", ".md", ".txt", ".joblib", ".log"}
    if rel.as_posix() in {"README_V8.md", "submissions/submission_content_v8.csv"}:
        return True
    if rel.parts[0] == "scripts" and path.name in {"select_v8.py", "train_v8.py", "refit_v8_regularization.py",
            "verify_v8.py", "verify_v8_cli.py", "report_v8.py", "package_v8.py", "verify_package_v8.py"}:
        return True
    return included_v7(path)


def main():
    verify = json.loads((ROOT / "reports/v8/verification.json").read_text())
    assert verify["offline_full_inference_byte_identical"] and verify["holdout_model_replay"] == "passed"
    assert verify["external_test_labels_used"] is False
    assert json.loads((ROOT / "reports/v8/tests.json").read_text())["status"] == "passed"
    assert json.loads((ROOT / "reports/v8/cli_verification.json").read_text())["status"] == "passed"
    assert digest(ROOT / "artifacts/v8/ensemble.joblib") == verify["model_sha256"]
    manifest_path = ROOT / "reports/v8/file_manifest.json"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p) and p != manifest_path)
    manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    assert not any(name.startswith("external/") or "reference.csv" in name or "with_content.csv" in name for name in manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(manifest_path)
    archive = ROOT.parent / "phishing-hackathon-solution-v8.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert set(target.namelist()) == {str(Path(ROOT.name) / path.relative_to(ROOT)) for path in files}
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files), "sha256": checksum, "crc_ok": True}), flush=True)


if __name__ == "__main__":
    main()
