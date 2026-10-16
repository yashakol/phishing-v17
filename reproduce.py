"""One entry point for environment setup, offline prediction and training replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def environment_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def setup(wheelhouse: Path | None = None) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12; verified patch version is 3.12.10")
    python = environment_python()
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if wheelhouse is not None:
        command += ["--no-index", "--find-links", str(wheelhouse.resolve())]
    command += ["-r", str(ROOT / "requirements.txt")]
    subprocess.run(command, cwd=ROOT, check=True)
    # This script changes only an environment-local library if the verified
    # macOS wheels need an OpenMP linkage repair; it never changes host libraries.
    repair = ROOT / "scripts/setup_openmp.py"
    if repair.is_file():
        subprocess.run([str(python), str(repair)], cwd=ROOT, check=True)


def run(arguments: list[str]) -> None:
    python = environment_python()
    if not python.is_file():
        raise RuntimeError("Environment is missing. Run: python3.12 reproduce.py setup")
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    subprocess.run([str(python), "-B", str(ROOT / "run_solution.py"), *arguments], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="reproduce", choices=["setup", "verify", "predict", "reproduce", "train", "test"])
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "setup":
        setup(args.wheelhouse)
        return
    if args.action == "verify":
        from delivery_assets import restore, verify_sources
        print(json.dumps({"assets": restore(ROOT), "historical_source_files_checked": verify_sources(ROOT)}))
        return
    if args.action == "test":
        python = environment_python()
        subprocess.run([str(python), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, check=True)
        return
    if args.action == "reproduce":
        if not environment_python().is_file():
            setup(args.wheelhouse)
        action = "predict"
    else:
        action = args.action
    command = [action]
    if args.output is not None:
        command += ["--output", str(args.output.resolve())]
    run(command)


if __name__ == "__main__":
    main()
