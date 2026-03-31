#!/usr/bin/env python3
"""
Build a .conda package for SPONGE.

Usage:
    python conda.py [--env ENV_NAME] [--output-dir DIR]

Examples:
    python conda.py --env dev-cpu
    python conda.py --env dev-cuda --output-dir ../output
"""

import argparse
import io
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import zstandard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = PROJECT_ROOT / ".pixi" / "envs"

PACKAGE_FILES = [
    "bin/SPONGE",
]
PORTABLE_RPATH_LINUX = "$ORIGIN/../lib"
PORTABLE_RPATH_MACOS = "@loader_path/../lib"
PATCH_TOOL_BY_SYSTEM = {
    "linux": "patchelf",
    "darwin": "install_name_tool",
}
RUNTIME_DEPENDENCIES = {
    "cpu": {
        "linux-64": [
            "mkl >=2025",
            "libllvm22 >=22.1,<23",
            "libclang-cpp >=22.1,<23",
            "libgomp",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
        "linux-aarch64": [
            "openblas >=0.3",
            "fftw >=3.3",
            "liblapacke >=3.11",
            "libllvm22 >=22.1,<23",
            "libclang-cpp >=22.1,<23",
            "libgomp",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
        "win-64": [
            "mkl >=2025",
            "libllvm22 >=22.1,<23",
            "libclang-cpp >=22.1,<23",
            "vc14_runtime",
            "ucrt",
            "llvm-openmp",
        ],
        "osx-arm64": [
            "openblas >=0.3",
            "fftw >=3.3",
            "liblapacke >=3.11",
            "libllvm22 >=22.1,<23",
            "libclang-cpp >=22.1,<23",
            "llvm-openmp",
            "libcxx",
        ],
    },
    "cpu-mpi": {
        "linux-64": [
            "mkl >=2025",
            "openmpi >=5,<6",
            "libllvm22 >=22.1,<23",
            "libclang-cpp >=22.1,<23",
            "libgomp",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
        "linux-aarch64": [
            "openblas >=0.3",
            "fftw >=3.3",
            "liblapacke >=3.11",
            "openmpi >=5,<6",
            "libllvm22 >=22.1,<23",
            "libclang-cpp >=22.1,<23",
            "libgomp",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
    },
    "cuda12": {
        "linux-64": [
            "cuda-version >=12,<13",
            "cuda-nvrtc >=12,<13",
            "libcublas >=12,<13",
            "libcufft >=11,<12",
            "libcurand >=10,<11",
            "libcusolver >=11,<12",
            "libnvjitlink >=12,<13",
            "libgomp",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
        "win-64": [
            "cuda-version >=12,<13",
            "cuda-nvrtc >=12,<13",
            "libcublas >=12,<13",
            "libcufft >=11,<12",
            "libcurand >=10,<11",
            "libcusolver >=11,<12",
            "libnvjitlink >=12,<13",
            "vc14_runtime",
            "ucrt",
            "llvm-openmp",
        ],
    },
    "cuda13": {
        "linux-64": [
            "cuda-version >=13,<14",
            "cuda-nvrtc >=13,<14",
            "libcublas >=13,<14",
            "libcufft >=12,<13",
            "libcurand >=10,<11",
            "libcusolver >=12,<13",
            "libcusparse >=12,<13",
            "libnvjitlink >=13,<14",
            "libgomp",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
        "win-64": [
            "cuda-version >=13,<14",
            "cuda-nvrtc >=13,<14",
            "libcublas >=13,<14",
            "libcufft >=12,<13",
            "libcurand >=10,<11",
            "libcusolver >=12,<13",
            "libcusparse >=12,<13",
            "libnvjitlink >=13,<14",
            "vc14_runtime",
            "ucrt",
            "llvm-openmp",
        ],
    },
    "hip": {
        "linux-64": [
            "hip-runtime-amd >=6.3,<7",
            "rocfft >=1.0,<2",
            "libstdcxx-ng",
            "libgcc-ng",
        ],
    },
}


def get_version() -> str:
    """Read version from pixi.toml."""
    pixi_toml = PROJECT_ROOT / "pixi.toml"
    for line in pixi_toml.read_text().splitlines():
        line = line.strip()
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def normalize_version_for_conda(version: str) -> str:
    """Normalize version string used in conda package metadata and filename."""
    return version.replace("-", "_")


def get_subdir() -> str:
    """Determine the conda subdir (platform-arch) for the current system."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-64"
        elif machine == "aarch64":
            return "linux-aarch64"
    elif system == "darwin":
        if machine == "arm64":
            return "osx-arm64"
        else:
            return "osx-64"
    elif system == "windows":
        return "win-64"
    return f"{system}-{machine}"


def detect_variant(env_name: str) -> str:
    """Detect variant by stripping the 'dev-' prefix from the environment name."""
    if not env_name.startswith("dev-"):
        sys.exit(
            f"Error: Environment name must start with 'dev-', got '{env_name}'"
        )
    return env_name[len("dev-") :]


def collect_files(env_name: str) -> list[tuple[Path, str]]:
    """Collect (source_path, archive_path) pairs based on PACKAGE_FILES."""
    env_prefix = ENVS_DIR / env_name
    files = []
    missing = []
    is_windows = platform.system().lower() == "windows"
    for rel_path in PACKAGE_FILES:
        src = env_prefix / rel_path
        arc_path = rel_path

        if not src.exists() and is_windows:
            # Windows installs executables as .exe in conda/pixi envs.
            src_exe = env_prefix / f"{rel_path}.exe"
            if src_exe.exists():
                src = src_exe
                arc_path = f"{rel_path}.exe"

        if not src.exists():
            missing.append(str(src))
            continue

        files.append((src, arc_path))
    if missing:
        sys.exit(
            "Error: The following files were not found:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nBuild first with: pixi run -e <env> configure && pixi run -e <env> compile"
        )
    return files


def patch_binary_rpath(files: list[tuple[Path, str]]) -> None:
    """Patch packaged binaries to use a portable relative RPATH."""
    system = platform.system().lower()
    if system not in {"linux", "darwin"}:
        return

    if system == "linux":
        tool = PATCH_TOOL_BY_SYSTEM[system]
        rpath_arg = PORTABLE_RPATH_LINUX
        args = ["--set-rpath", rpath_arg]
    else:
        tool = PATCH_TOOL_BY_SYSTEM[system]
        rpath_arg = PORTABLE_RPATH_MACOS
        args = ["-add_rpath", rpath_arg]

    patch_cmd = [tool, *args, ""]

    for src_path, arc_path in files:
        if not arc_path.startswith("bin/"):
            continue
        patch_cmd[-1] = str(src_path)
        try:
            subprocess.run(
                patch_cmd, check=True, capture_output=True, text=True
            )
            print(f"Patched RPATH: {src_path} -> {rpath_arg}")
        except FileNotFoundError:
            if system == "linux":
                sys.exit(
                    "Error: patchelf not found in PATH. "
                    "Install patchelf in the packaging environment."
                )
            sys.exit(
                "Error: install_name_tool not found in PATH. "
                "Run packaging on macOS with Xcode command line tools."
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            if system == "darwin" and "would duplicate path" in details:
                print(f"RPATH already set: {src_path} -> {rpath_arg}")
                continue
            sys.exit(
                f"Error: Failed to patch RPATH for '{src_path}' with {tool}. "
                f"{details}"
            )


def make_metadata(
    name: str,
    version: str,
    build_string: str,
    subdir: str,
    variant: str,
) -> dict:
    """Create the conda index.json metadata."""
    if variant not in RUNTIME_DEPENDENCIES:
        known_variants = ", ".join(sorted(RUNTIME_DEPENDENCIES))
        sys.exit(
            f"Error: No dependency mapping for variant '{variant}'. "
            f"Known variants: {known_variants}"
        )

    variant_dependencies = RUNTIME_DEPENDENCIES[variant]
    if subdir not in variant_dependencies:
        known_subdirs = ", ".join(sorted(variant_dependencies))
        sys.exit(
            f"Error: No dependency mapping for subdir '{subdir}' under variant '{variant}'. "
            f"Known subdirs: {known_subdirs}"
        )

    depends = list(variant_dependencies[subdir])

    return {
        "name": name,
        "version": version,
        "build": build_string,
        "build_number": 0,
        "depends": depends,
        "license": "Apache-2.0",
        "platform": subdir.rsplit("-", 1)[0] if "-" in subdir else subdir,
        "arch": subdir.rsplit("-", 1)[1] if "-" in subdir else "unknown",
        "subdir": subdir,
        "timestamp": int(time.time() * 1000),
    }


def make_paths_json(files: list[tuple[Path, str]]) -> dict:
    """Create the paths.json metadata."""
    paths = []
    for src_path, archive_path in files:
        file_bytes = src_path.read_bytes()
        entry = {
            "_path": archive_path,
            "path_type": "hardlink",
            "sha256": hashlib.sha256(file_bytes).hexdigest(),
            "size_in_bytes": len(file_bytes),
        }
        if archive_path.startswith("bin/"):
            entry["file_mode"] = "binary"
        paths.append(entry)
    return {"paths": paths, "paths_version": 1}


def _compress_zst(tar_bytes: bytes) -> bytes:
    """Compress raw tar bytes with zstd."""
    cctx = zstandard.ZstdCompressor(level=19)
    return cctx.compress(tar_bytes)


def build_inner_tar(files: list[tuple[Path, str]]) -> bytes:
    """Build a tar.zst archive in memory from file list."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for src_path, arc_path in files:
            tar.add(str(src_path), arcname=arc_path)
    return _compress_zst(buf.getvalue())


def _add_text_entry(tar: tarfile.TarFile, name: str, data: bytes):
    """Add a text file entry to a tar archive."""
    entry = tarfile.TarInfo(name=name)
    entry.size = len(data)
    entry.mtime = int(time.time())
    entry.mode = 0o644
    tar.addfile(entry, io.BytesIO(data))


def build_info_tar(
    metadata: dict, paths_json: dict, archive_paths: list[str]
) -> bytes:
    """Build the info tar.zst archive containing index.json, paths.json, and files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        _add_text_entry(
            tar,
            "info/index.json",
            json.dumps(metadata, indent=2).encode("utf-8"),
        )
        _add_text_entry(
            tar,
            "info/paths.json",
            json.dumps(paths_json, indent=2).encode("utf-8"),
        )
        _add_text_entry(
            tar,
            "info/files",
            "\n".join(archive_paths).encode("utf-8"),
        )

    return _compress_zst(buf.getvalue())


def build_conda_v2(
    output_path: Path,
    files: list[tuple[Path, str]],
    metadata: dict,
    paths_json: dict,
    archive_paths: list[str],
) -> Path:
    """Build a .conda (v2 format) package.

    The .conda format is a ZIP file containing:
      - metadata.json          (package format metadata)
      - pkg-<name>-<ver>.tar.zst   (package files)
      - info-<name>-<ver>.tar.zst  (metadata files)
    """
    stem = f"{metadata['name']}-{metadata['version']}-{metadata['build']}"
    conda_path = output_path / f"{stem}.conda"

    pkg_tar = build_inner_tar(files)
    info_tar = build_info_tar(metadata, paths_json, archive_paths)

    format_metadata = {
        "conda_pkg_format_version": 2,
    }
    format_metadata_bytes = json.dumps(format_metadata, indent=2).encode(
        "utf-8"
    )

    with zipfile.ZipFile(conda_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata.json", format_metadata_bytes)
        zf.writestr(f"pkg-{stem}.tar.zst", pkg_tar)
        zf.writestr(f"info-{stem}.tar.zst", info_tar)

    return conda_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a .conda package for SPONGE"
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Pixi environment name (e.g. dev-cpu, dev-cuda).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Output directory for the .conda file (default: packaging/outputs)",
    )
    args = parser.parse_args()

    env_name = args.env
    variant = detect_variant(env_name)
    name = f"sponge-{variant}"
    raw_version = get_version()
    version = normalize_version_for_conda(raw_version)
    subdir = get_subdir()
    build_string = subdir.replace("-", "_")

    print(f"Package: {name}")
    print(f"Version: {version}")
    if raw_version != version:
        print(f"Raw version: {raw_version}")
    print(f"Variant: {variant}")
    print(f"Subdir:  {subdir}")
    print(f"Env:     {env_name}")
    print()

    # Collect files
    files = collect_files(env_name)
    archive_paths = [arc for _, arc in files]
    patch_binary_rpath(files)

    # Build metadata
    metadata = make_metadata(
        name=name,
        version=version,
        build_string=build_string,
        subdir=subdir,
        variant=variant,
    )
    paths_json = make_paths_json(files)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build .conda package
    conda_path = build_conda_v2(
        output_dir, files, metadata, paths_json, archive_paths
    )

    size_mb = conda_path.stat().st_size / (1024 * 1024)
    print(f"Built: {conda_path}")
    print(f"Size:  {size_mb:.2f} MB")
    print()
    print("Install with:")
    print(f"  conda install {conda_path}")


if __name__ == "__main__":
    main()
