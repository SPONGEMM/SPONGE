"""Build and run standalone CPU constraint probes in the active toolchain."""

import os
import shutil
import subprocess
from pathlib import Path

PROBE_PROJECT = Path(__file__).with_name("probes")


def run_constraint_probe(tmp_path, target):
    cmake = shutil.which("cmake")
    assert cmake is not None, "CMake is required; run in a Pixi dev environment"
    build_dir = tmp_path / "build"
    configure = [
        cmake,
        "-S",
        str(PROBE_PROJECT),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if os.environ.get("CONDA_PREFIX"):
        configure.append(f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}")
    # CMake honors CXX, locates MSVC on Windows, and supplies compiler-specific
    # OpenMP flags. Configure/build failures must fail CI rather than skip.
    for stage, command in [
        ("configure", configure),
        (
            "compile",
            [
                cmake,
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                target,
                "--parallel",
                "2",
            ],
        ),
    ]:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=180
        )
        assert result.returncode == 0, (
            f"failed to {stage} {target}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    suffix = ".exe" if os.name == "nt" else ""
    executable = build_dir / "bin" / f"{target}{suffix}"
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
