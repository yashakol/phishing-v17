"""Bundle compact HTML data, trained model and reproducible source, no raw 9GB CSV."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def included(path):
    relative = path.relative_to(ROOT)
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache", "component_cache"} for part in relative.parts):
        return False
    if path.suffix in {".pyc", ".tmp"} or path.name == ".DS_Store":
        return False
    first = relative.parts[0]
    if first == "phishing_url" or first == "tests":
        return path.suffix == ".py"
    if first == "scripts":
        return path.name in {"extract_content.py", "prepare_content_url_features.py", "prepare_content_components.py",
                             "train_v3.py", "verify_v3.py", "verify_content_selection.py", "package_v3.py", "report_v3.py"}
    if first == "data":
        return path.name in {"train.csv", "test.csv", "sample.csv", "content_train.joblib", "content_test.joblib",
                             "content_url_train.joblib", "content_url_test.joblib"}
    if first == "artifacts":
        return relative.parts[:2] == ("artifacts", "v3")
    if first == "reports":
        return relative.parts[:2] == ("reports", "v3") or relative.as_posix() in {
            "reports/clean_train.csv", "reports/split_indices.npz", "reports/domain_split_indices.npz", "reports/development_oof.npz", "reports/metrics.json", "reports/v2/url_model_metrics.json"}
    if first == "experiments":
        if relative.parts[1] not in {"content", "html_astra"}:
            # Include sources and development-only records for the URL auxiliary model.
            return relative.parts[1] == "astra" and (path.suffix == ".py" or path.name in {
                "README.md", "selection.json", "results.json", "internal_split.npz",
                "development_oof.npz", "oof_results.json"})
        return True
    if first == "submissions":
        return path.name == "submission_content_v3.csv"
    return path.name in {"README_CONTENT.md", "HOW_IT_WORKS.md", "requirements.txt"}


def main():
    if not (ROOT / "reports/v3/verification.json").exists():
        raise ValueError("Verify the new solution before packaging it")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p))
    manifest_path = ROOT / "reports/v3/file_manifest.json"
    manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files if p != manifest_path}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if manifest_path not in files:
        files.append(manifest_path)
    archive = ROOT.parent / "phishing-hackathon-solution-v3.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert f"{ROOT.name}/artifacts/v3/content.joblib" in target.namelist()
        assert f"{ROOT.name}/data/content_test.joblib" in target.namelist()
        assert not any("with_content.csv" in name or "/external/" in name for name in target.namelist())
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files), "crc_ok": True}))


if __name__ == "__main__":
    main()
