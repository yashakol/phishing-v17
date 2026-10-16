"""Package a verified supervised candidate and its reproducibility records."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v6 import included as included_v6


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def included(path):
    rel = path.relative_to(ROOT)
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".tmp"} or path.name == ".DS_Store":
        return False
    if path.suffix == ".log" and rel.parts[:2] != ("experiments", "v7_astra"):
        return False
    if rel.as_posix() in {"reports/v6/file_manifest.json", "reports/v5/file_manifest.json",
                          "reports/v3/file_manifest.json"}:
        return False
    if rel.parts[:2] == ("reports", "v7"):
        return path.suffix in {".json", ".npz", ".xml", ".csv"} and path.name != "reproduced_submission.csv"
    if rel.parts[:2] == ("artifacts", "v7"):
        return path.suffix in {".joblib", ".json"}
    if rel.parts[:2] in {("experiments", "v7_astra"), ("experiments", "v7_hierarchy")}:
        return path.suffix in {".py", ".md", ".json", ".npz", ".joblib", ".log"}
    if rel.parts[:2] in {("experiments", "v7_signatures"), ("experiments", "v7_stacking")}:
        return path.suffix in {".py", ".json", ".npz", ".joblib"}
    if rel.as_posix() in {"experiments/v7_astra_selection.py", "README_V7.md",
                          "submissions/submission_content_v7.csv"}:
        return True
    if rel.parts[0] == "scripts" and path.name in {
        "train_v7.py", "verify_v7.py", "verify_v7_cli.py", "report_v7.py",
        "package_v7.py", "verify_package_v7.py",
    }:
        return True
    return included_v6(path)


def main():
    verification = json.loads((ROOT / "reports/v7/verification.json").read_text())
    assert verification["offline_full_inference_byte_identical"]
    assert verification["holdout_model_replay"] == verification["oof_mixture_direct_replay"] == "passed"
    assert verification["external_test_labels_used"] is False
    assert json.loads((ROOT / "reports/v7/tests.json").read_text())["status"] == "passed"
    assert json.loads((ROOT / "reports/v7/cli_verification.json").read_text())["status"] == "passed"
    assert digest(ROOT / "artifacts/v7/ensemble.joblib") == verification["model_sha256"]
    manifest_path = ROOT / "reports/v7/file_manifest.json"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p) and p != manifest_path)
    manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    assert not any(name.startswith("external/") or "reference.csv" in name or "with_content.csv" in name for name in manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(manifest_path)
    archive = ROOT.parent / "phishing-hackathon-solution-v7.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert set(target.namelist()) == {str(Path(ROOT.name) / p.relative_to(ROOT)) for p in files}
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files),
                      "sha256": checksum, "crc_ok": True}), flush=True)


if __name__ == "__main__":
    main()
