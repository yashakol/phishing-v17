"""Check manifest, imports, tests and offline CLI in a fresh extracted directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, root, env):
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


def main(archive):
    archive = archive.resolve()
    checksum = digest(archive)
    assert archive.with_suffix(".zip.sha256").read_text().split()[0] == checksum
    with tempfile.TemporaryDirectory(prefix="phishing-v9-verify-") as directory:
        destination = Path(directory)
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            assert len(names) == len(set(names))
            assert source.testzip() is None
            for name in names:
                relative = Path(name)
                assert relative.parts[0] == ROOT.name and not relative.is_absolute()
                assert ".." not in relative.parts
            source.extractall(destination)
        project = destination / ROOT.name
        manifest = json.loads((project / "reports/v9/file_manifest.json").read_text())
        expected = {f"{ROOT.name}/{name}" for name in manifest}
        expected.add(f"{ROOT.name}/reports/v9/file_manifest.json")
        assert set(names) == expected
        for name, entry in manifest.items():
            path = project / name
            assert path.stat().st_size == entry["bytes"], name
            assert digest(path) == entry["sha256"], name
        print(json.dumps({"event": "extracted_manifest_verified", "files": len(names)}), flush=True)
        env = dict(os.environ, PYTHONPATH=str(project), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        import_probe = "from pathlib import Path; import phishing_url; " \
            "assert Path(phishing_url.__file__).resolve().is_relative_to(Path.cwd()); " \
            "print(phishing_url.__file__)"
        run([sys.executable, "-c", import_probe], project, env)
        junit = project / "portable_tests.xml"
        output = run([sys.executable, "-m", "pytest", "-q", "--import-mode=importlib",
                      f"--junitxml={junit}"], project, env)
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        assert totals["tests"] == json.loads((project / "reports/v9/tests.json").read_text())["tests"]
        assert totals["failures"] == totals["errors"] == totals["skipped"] == 0
        print(output, flush=True)
        print(json.dumps({"event": "portable_tests_passed", **totals}), flush=True)
        # Use real initial test rows, with extracted source and saved model. No
        # sockets are permitted while loading inputs or running the entry point.
        cli_probe = r'''
from pathlib import Path
import socket,sys
def blocked(*args,**kwargs):
    raise AssertionError("Network is disabled during portable inference")
socket.create_connection=blocked
socket.getaddrinfo=blocked
import joblib,numpy as np,pandas as pd
from phishing_url import v9_cli
root=Path.cwd()
assert Path(v9_cli.__file__).resolve().is_relative_to(root)
cache=joblib.load(root/"data/content_test.joblib")
url=joblib.load(root/"data/content_url_test.joblib")
extra=joblib.load(root/"data/extra_html_test.joblib")
assert cache["labels"] is None and extra["labels"] is None and cache["urls"]==url["urls"]==extra["urls"]
sample={**cache,"urls":cache["urls"][:8],"records":cache["records"][:8],
        "hashes":cache["hashes"][:8],"features":cache["features"].iloc[:8].reset_index(drop=True)}
aux={**url,"urls":url["urls"][:8],"features":url["features"].iloc[:8].reset_index(drop=True)}
extra_sample={**extra,"urls":extra["urls"][:8],"records":extra["records"][:8],"features":extra["features"].iloc[:8].reset_index(drop=True)}
joblib.dump(extra_sample,root/"portable_extra.joblib")
joblib.dump(sample,root/"portable_cache.joblib")
joblib.dump(aux,root/"portable_url.joblib")
pd.read_csv(root/"data/sample.csv").head(8).to_csv(root/"portable_sample.csv",index=False)
sys.argv=["v9_cli","--model","artifacts/v9/ensemble.joblib","--cache","portable_cache.joblib",
          "--extra-cache","portable_extra.joblib","--url-features","portable_url.joblib","--sample","portable_sample.csv",
          "--output","portable_submission.csv"]
v9_cli.main()
pd.testing.assert_frame_equal(pd.read_csv(root/"portable_submission.csv"),
                              pd.read_csv(root/"submissions/submission_content_v9.csv").head(8))
print("Portable offline CLI matches all 8 selected submission rows.")
'''
        print(run([sys.executable, "-c", cli_probe], project, env), flush=True)
    result = {"archive": str(archive), "sha256": checksum, "bytes": archive.stat().st_size,
              "files": len(names), "crc_ok": True, "manifest_sha256_ok": True,
              "extracted_source_imported": True, "portable_tests": totals,
              "portable_offline_cli_rows": 8, "portable_cli_matches_submission": True,
              "full_16000_row_offline_replay": "reports/v9/verification.json",
              "environment": "Current verified Python environment; source, data and models freshly extracted"}
    archive.with_suffix(".zip.verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "phishing-hackathon-solution-v9.zip")
    main(parser.parse_args().archive)
