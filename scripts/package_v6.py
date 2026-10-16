"""Package the verified extra-HTML model, data, source and development records."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v5 import included as included_v5


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def included(path):
    rel = path.relative_to(ROOT)
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".tmp"} or path.name == ".DS_Store":
        return False
    if rel.as_posix() == "reports/v5/file_manifest.json":
        return False
    if rel.as_posix() in {"data/extra_html_train.joblib", "data/extra_html_test.joblib", "README_V6.md",
                          "submissions/submission_content_v6.csv",
                          "artifacts/v5/component_cache/holdout_cat.joblib", "artifacts/v5/component_cache/holdout_hash.joblib"}:
        return True
    if rel.parts[:2] == ("experiments", "v6"):
        return path.suffix in {".py", ".json", ".npz"} or path.suffix == ".joblib" and any(part.endswith("_oof") for part in rel.parts)
    if rel.parts[:2] in {("reports", "v6"), ("artifacts", "v6")}:
        return "component_cache" not in rel.parts and path.name not in {"cli_sample_cache.joblib", "cli_sample_extra.joblib", "cli_sample_url.joblib"}
    if rel.parts[0] == "scripts" and path.name in {"extract_extra_html.py", "train_v6.py", "verify_v6.py", "package_v6.py", "verify_package_v6.py", "report_v6.py"}:
        return True
    return included_v5(path)


def main():
    verify = json.loads((ROOT / "reports/v6/verification.json").read_text())
    assert verify["offline_full_inference_byte_identical"] and verify["holdout_model_replay"] == "passed"
    assert json.loads((ROOT / "reports/v6/tests.json").read_text())["status"] == "passed"
    assert json.loads((ROOT / "reports/v6/cli_verification.json").read_text())["status"] == "passed"
    manifest_path = ROOT / "reports/v6/file_manifest.json"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p) and p != manifest_path)
    manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(manifest_path)
    archive = ROOT.parent / "phishing-hackathon-solution-v6.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert set(target.namelist()) == {str(Path(ROOT.name) / p.relative_to(ROOT)) for p in files}
        assert not any("with_content.csv" in name or "/external/" in name for name in target.namelist())
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files),
                      "sha256": checksum, "crc_ok": True}), flush=True)


if __name__ == "__main__":
    main()
