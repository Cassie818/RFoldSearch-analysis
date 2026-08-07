"""Install an already-downloaded DSSR Basic zip into the local repo.

This script does not download DSSR. DSSR Basic is distributed through the
Columbia Technology Ventures license portal. After downloading the official
zip, run this helper to copy the macOS executable to tools/x3dna-dssr.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INSTALL = REPO / "tools" / "x3dna-dssr"


def newest_download_zip() -> Path | None:
    candidates = sorted(
        Path.home().joinpath("Downloads").glob("dssr-basic-*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def file_type(path: Path) -> str:
    proc = subprocess.run(["file", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout.strip()


def looks_like_macos_binary(path: Path) -> bool:
    text = file_type(path)
    return "Mach-O" in text


def find_dssr_binary(extract_dir: Path) -> Path:
    names = {"x3dna-dssr", "dssr"}
    candidates = [path for path in extract_dir.rglob("*") if path.is_file() and path.name in names]
    macos = [path for path in candidates if looks_like_macos_binary(path)]
    if not macos:
        details = "\n".join(f"{path}: {file_type(path)}" for path in candidates)
        raise SystemExit(f"No macOS DSSR executable found in zip.\n{details}")
    arm64 = [path for path in macos if "arm64" in file_type(path)]
    return arm64[0] if arm64 else macos[0]


def verify(binary: Path) -> None:
    proc = subprocess.run([str(binary), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    if proc.returncode not in {0, 1}:
        raise SystemExit(f"Installed binary did not run cleanly:\n{proc.stderr}\n{proc.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", nargs="?", type=Path, help="Official DSSR Basic zip from Columbia.")
    parser.add_argument("--install-path", type=Path, default=DEFAULT_INSTALL)
    args = parser.parse_args()

    zip_path = args.zip_path or newest_download_zip()
    if not zip_path:
        raise SystemExit("No DSSR Basic zip found. Download it first, or pass the zip path explicitly.")
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")

    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        binary = find_dssr_binary(extract_dir)
        args.install_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, args.install_path)

    mode = args.install_path.stat().st_mode
    args.install_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    xattr = shutil.which("xattr")
    if xattr:
        subprocess.run([xattr, "-d", "com.apple.quarantine", str(args.install_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    verify(args.install_path)
    os.environ["DSSR_BIN"] = str(args.install_path)
    print(f"Installed DSSR: {args.install_path}")
    print(file_type(args.install_path))


if __name__ == "__main__":
    main()
