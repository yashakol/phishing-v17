"""Package the verified full-data refit without replacing older deliveries."""

import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from package_v15 import digest, included as included_v15

EXPERIMENTS = {"v16_categories", "v16_neighbor", "v18_neighbor_vote"}


def included(path):
    rel = path.relative_to(ROOT)
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache", "external"} for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".tmp"} or path.name == ".DS_Store" or "reference" in path.name.lower():
        return False
    if rel.parts[:2] in {("reports", "v2"), ("artifacts", "v2")}:
        return False
    if path.name == "file_manifest.json" and rel.parts[:2] != ("reports", "v17"):
        return False
    if rel.parts[:2] in {("reports", "v17"), ("artifacts", "v17")}:
        return path.suffix in {".json", ".npz", ".xml", ".csv", ".joblib", ".log"} and path.name not in {
            "reproduced_submission.csv", "package_run.log", "package_verification_run.log"}
    if len(rel.parts) > 1 and rel.parts[0] == "experiments" and rel.parts[1] in EXPERIMENTS:
        return path.suffix in {".py", ".json", ".npz", ".md", ".txt", ".joblib", ".log"}
    if rel.as_posix() in {"README_V17.md", "submissions/submission_content_v17.csv"}:
        return True
    if rel.parts[0] == "scripts" and "_v17" in path.stem:
        return path.suffix == ".py"
    return included_v15(path)


def files_to_package():
    result = []
    for directory, names, files in os.walk(ROOT):
        names[:] = sorted(name for name in names if name not in {".venv", ".git", "__pycache__", ".pytest_cache", "external"})
        for name in files:
            path = Path(directory)/name
            if included(path):
                result.append(path)
    return sorted(result)


def main():
    out = ROOT/"reports/v17"
    verification = json.loads((out/"verification.json").read_text())
    assert verification["status"] == "passed" and verification["offline_full_inference_byte_identical"]
    assert verification["external_test_labels_used"] is False
    for name in ("tests.json", "cli_verification.json", "retraining_verification.json"):
        assert json.loads((out/name).read_text())["status"] == "passed"
    assert digest(ROOT/"artifacts/v17/ensemble.joblib") == verification["model_sha256"]
    archive = ROOT.parent/"phishing-hackathon-solution-v17.zip"
    assert not archive.exists(), "Do not replace a delivered archive"
    manifest_path = out/"file_manifest.json"
    files = [path for path in files_to_package() if path != manifest_path]
    assert ROOT/"README_V17.md" in files
    manifest = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in files}
    assert not any(name.startswith(("external/", "reports/v2/", "artifacts/v2/")) or "reference" in name.lower() or "with_content.csv" in name for name in manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    files.append(manifest_path)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert set(target.namelist()) == {str(Path(ROOT.name)/path.relative_to(ROOT)) for path in files}
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "files": len(files), "bytes": archive.stat().st_size,
                      "sha256": checksum, "crc_ok": True}), flush=True)


if __name__ == "__main__":
    main()
