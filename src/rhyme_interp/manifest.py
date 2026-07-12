"""Small, dependency-free run manifests for reproducible experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys

PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "cmudict", "numpy", "pandas")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def build_manifest(*, model: str, revision: str | None, precision: str, seed: int | None,
                   datasets: list[str | Path], command: list[str] | None = None) -> dict:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or sys.argv,
        "model": {"id": model, "revision": revision, "precision": precision},
        "seed": seed,
        "datasets": [{"path": str(path), "sha256": sha256_file(path)} for path in datasets],
        "git": {"commit": _git(["rev-parse", "HEAD"]), "dirty": bool(_git(["status", "--porcelain"]))},
        "environment": {"python": platform.python_version(), "packages": package_versions()},
    }


def write_manifest(output: str | Path, **kwargs) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(**kwargs), indent=2, sort_keys=True) + "\n")
    return path
