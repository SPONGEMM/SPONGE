import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROBE_SOURCE = Path(__file__).with_name(
    "constraint_velocity_projection_probe.cpp"
)


def _compiler_command():
    configured = os.environ.get("CXX")
    if configured:
        return shlex.split(configured)
    if sys.platform == "darwin" and Path("/usr/bin/clang++").is_file():
        return ["/usr/bin/clang++"]
    compiler = (
        shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    )
    if compiler is None:
        pytest.skip("a C++17 compiler is required for the constraint probe")
    return [compiler]


def _dependency_include():
    candidates = []
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(os.environ["CONDA_PREFIX"]) / "include")
    candidates.append(
        REPOSITORY_ROOT / ".pixi" / "envs" / "dev-cpu" / "include"
    )
    for candidate in candidates:
        # GCC provides omp.h through its compiler include path.
        if (candidate / "fftw3.h").is_file():
            return candidate
        if (candidate / "fftw" / "fftw3.h").is_file():
            return candidate / "fftw"
    pytest.skip("OpenMP and FFTW headers are required for the constraint probe")


def test_velocity_only_constraint_projection_converges_without_moving_crd(
    tmp_path,
):
    executable = tmp_path / "constraint_velocity_projection_probe"
    dead_code_flags = (
        ["-Wl,-dead_strip"]
        if sys.platform == "darwin"
        else ["-Wl,--gc-sections"]
    )
    compile_result = subprocess.run(
        [
            *_compiler_command(),
            "-std=c++17",
            "-DUSE_CPU",
            "-DNO_GLOBAL_CONTROLLER",
            "-O3",
            "-march=native",
            "-ffast-math",
            "-ffunction-sections",
            "-fdata-sections",
            "-w",
            f"-I{REPOSITORY_ROOT / 'SPONGE'}",
            f"-I{_dependency_include()}",
            str(PROBE_SOURCE),
            str(REPOSITORY_ROOT / "SPONGE/constrain/settle.cpp"),
            str(REPOSITORY_ROOT / "SPONGE/constrain/shake.cpp"),
            str(REPOSITORY_ROOT / "SPONGE/common.cpp"),
            *dead_code_flags,
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert compile_result.returncode == 0, (
        "failed to compile constraint projection probe\n"
        f"stdout:\n{compile_result.stdout}\nstderr:\n{compile_result.stderr}"
    )

    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
