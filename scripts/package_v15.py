"""Package the verified resource specialist and preserve all delivered archives."""

import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v10 import digest, included as included_v10

VERSIONS = {"v11", "v12", "v14", "v15"}
EXPERIMENTS = {"v11_meta", "v11_shift", "v12_expert", "v12_review", "v13_static", "v14_pair", "v15_resources", "v15_review"}


def included(path):
    rel = path.relative_to(ROOT)
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache", "external"} for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".tmp"} or path.name == ".DS_Store":
        return False
    if rel.as_posix() in {"phishing_url/reference.py", "tests/test_reference.py"}:
        return False
    if rel.parts[:2] in {("reports", "v2"), ("artifacts", "v2")}:
        return False
    if rel.parts[:2] == ("reports", "v15") and path.name in {"package_run.log", "package_verification_run.log"}:
        return False
    if path.name == "file_manifest.json" and rel.parts[:2] != ("reports", "v15"):
        return False
    if len(rel.parts) > 1 and rel.parts[0] in {"reports", "artifacts"} and rel.parts[1] in VERSIONS:
        return path.suffix in {".json", ".npz", ".xml", ".csv", ".joblib", ".log"} and path.name != "reproduced_submission.csv"
    if len(rel.parts) > 1 and rel.parts[0] == "experiments" and rel.parts[1] in EXPERIMENTS:
        return path.suffix in {".py", ".json", ".npz", ".md", ".txt", ".joblib", ".log"}
    if rel.as_posix() in {"README_V15.md", "submissions/submission_content_v15.csv"}:
        return True
    if rel.parts[0] == "scripts" and any(f"_{v}" in path.stem for v in VERSIONS):
        return path.suffix == ".py"
    return included_v10(path)


def files_to_package():
    result = []
    for directory, names, files in os.walk(ROOT):
        names[:] = sorted(n for n in names if n not in {".venv", ".git", "__pycache__", ".pytest_cache", "external"})
        for name in files:
            path = Path(directory) / name
            if included(path):
                result.append(path)
    return sorted(result)


def main():
    out = ROOT / "reports/v15"
    verify = json.loads((out / "verification.json").read_text())
    assert verify["offline_full_inference_byte_identical"] and verify["holdout_model_replay"] == "passed"
    assert verify["external_test_labels_used"] is False
    assert json.loads((out / "tests.json").read_text())["status"] == "passed"
    assert json.loads((out / "cli_verification.json").read_text())["status"] == "passed"
    assert digest(ROOT / "artifacts/v15/ensemble.joblib") == verify["model_sha256"]
    manifest_path = out / "file_manifest.json"
    files = [p for p in files_to_package() if p != manifest_path]
    assert ROOT / "README_V15.md" in files
    manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    assert not any(name.startswith("external/") or "reference" in name.lower() or "with_content.csv" in name for name in manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    files.append(manifest_path)
    archive = ROOT.parent / "phishing-hackathon-solution-v15.zip"
    assert not archive.exists(), "Do not overwrite a delivered archive"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert set(target.namelist()) == {str(Path(ROOT.name) / path.relative_to(ROOT)) for path in files}
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files),
                      "sha256": checksum, "crc_ok": True}), flush=True)


if __name__ == "__main__":
    main()
