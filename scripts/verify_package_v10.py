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
    with tempfile.TemporaryDirectory(prefix="phishing-v10-verify-") as directory:
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
        manifest = json.loads((project / "reports/v10/file_manifest.json").read_text())
        expected = {f"{ROOT.name}/{name}" for name in manifest}
        expected.add(f"{ROOT.name}/reports/v10/file_manifest.json")
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
        assert totals["tests"] == json.loads((project / "reports/v10/tests.json").read_text())["tests"]
        assert totals["failures"] == totals["errors"] == totals["skipped"] == 0
        print(output, flush=True)
        print(json.dumps({"event": "portable_tests_passed", **totals}), flush=True)
        # Use real initial test rows, with extracted source and saved model. No
        # sockets are permitted while loading inputs or running the entry point.
        from verify_v10_cli import PROBE as cli_probe
        probe_directory = project / "portable_probe"
        probe_directory.mkdir()
        print(run([sys.executable, "-c", cli_probe, str(probe_directory)], project, env), flush=True)
    result = {"archive": str(archive), "sha256": checksum, "bytes": archive.stat().st_size,
              "files": len(names), "crc_ok": True, "manifest_sha256_ok": True,
              "extracted_source_imported": True, "portable_tests": totals,
              "portable_offline_cli_rows": 8, "portable_cli_matches_submission": True,
              "full_16000_row_offline_replay": "reports/v10/verification.json",
              "environment": "Current verified Python environment; source, data and models freshly extracted"}
    archive.with_suffix(".zip.verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "phishing-hackathon-solution-v10.zip")
    main(parser.parse_args().archive)
