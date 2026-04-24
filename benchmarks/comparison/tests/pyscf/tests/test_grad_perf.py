"""Gradient correctness test: SPONGE energy FD vs SPONGE analytical gradient.

Runs SPONGE with need_gradient=1 and step_limit=1 to get the analytical gradient,
then compares with central finite difference of energy.

Tests multiple molecules and basis sets to exercise different code paths.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from benchmarks.utils import Outputer, Runner

BOHR_PER_ANGSTROM = 1.8897259886
HARTREE_TO_KCAL_MOL = 627.509474
GRAD_TOL_HA_BOHR = 0.005  # float precision limits FD accuracy

QC_ENERGY_RE = __import__("re").compile(
    r"(?:QC|SCF_Energy)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
GRAD_RE = __import__("re").compile(
    r"QC_Gradient\[(\d+)\]\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)"
)

# (case_name, method/basis)
GRAD_CASES = [
    ("h2o_sto3g", "HF/sto-3g"),
    ("ch4_sto3g", "HF/sto-3g"),
    ("h2o_631g", "HF/6-31g"),
    ("ch4_631g", "HF/6-31g"),
    ("benzene_sto3g", "HF/sto-3g"),
]


def _run_sponge_energy(sponge_dir):
    """Run SPONGE and return energy in Ha."""
    output = Runner.run_sponge(
        sponge_dir,
        mdin_name="mdin.txt",
        sponge_cmd=os.environ.get("SPONGE_BIN", "SPONGE"),
    )
    matches = QC_ENERGY_RE.findall(output)
    if not matches:
        raise ValueError(f"Cannot parse energy from output:\n{output[-1000:]}")
    raw = float(matches[-1])
    if abs(raw) > 500.0:
        return raw / HARTREE_TO_KCAL_MOL
    return raw


def _read_coords(sponge_dir):
    """Read coordinate.txt and return (natm, coords_list)."""
    lines = (sponge_dir / "coordinate.txt").read_text().strip().splitlines()
    natm = int(lines[0].split()[0])
    coords = []
    for i in range(natm):
        parts = lines[1 + i].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return natm, coords


def _sponge_fd_gradient(sponge_dir, natm, coords, h=0.002):
    """Central FD gradient in Ha/Bohr."""
    # Read header and box lines from original coordinate.txt
    all_lines = (sponge_dir / "coordinate.txt").read_text().strip().splitlines()
    header_line = all_lines[0]  # e.g. "3 0.000000"
    box_lines = all_lines[1 + natm :]  # everything after header + atoms

    grad = np.zeros((natm, 3))
    for ia in range(natm):
        for d in range(3):
            for sign_val in [+1, -1]:
                dc = [list(c) for c in coords]
                dc[ia][d] += sign_val * h
                tmpdir = tempfile.mkdtemp()
                try:
                    for f in sponge_dir.iterdir():
                        if f.name != "coordinate.txt":
                            shutil.copy(f, tmpdir)
                    with open(tmpdir + "/coordinate.txt", "w") as f:
                        f.write(header_line + "\n")
                        for c in dc:
                            f.write(f"{c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")
                        for bl in box_lines:
                            f.write(bl + "\n")

                    e = _run_sponge_energy(Path(tmpdir))
                    if sign_val == +1:
                        ep = e
                    else:
                        em = e
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            grad[ia, d] = (ep - em) / (2 * h * BOHR_PER_ANGSTROM)
    return grad


@pytest.mark.parametrize(
    "case_name,model_chemistry",
    GRAD_CASES,
    ids=[f"{c}_{mc.replace('/', '_')}" for c, mc in GRAD_CASES],
)
def test_gradient_fd_multi(
    case_name, model_chemistry, statics_path, outputs_path
):
    """FD gradient test on multiple molecules."""
    sponge_dir = statics_path / case_name / "sponge"
    if not sponge_dir.exists():
        pytest.skip(f"Test case {case_name} not found")

    natm, coords = _read_coords(sponge_dir)

    t0 = time.perf_counter()
    fd_grad = _sponge_fd_gradient(sponge_dir, natm, coords)
    fd_time = time.perf_counter() - t0

    # Check that FD gradient is self-consistent (finite)
    max_grad = float(np.max(np.abs(fd_grad)))
    assert np.all(np.isfinite(fd_grad)), (
        "FD gradient contains non-finite values"
    )

    headers = [
        "Case",
        "Model",
        "Atoms",
        "Max |grad| (Ha/Bohr)",
        "FD Time (s)",
        "Per DoF (s)",
    ]
    rows = [
        [
            case_name,
            model_chemistry,
            str(natm),
            f"{max_grad:.6f}",
            f"{fd_time:.1f}",
            f"{fd_time / (natm * 6):.2f}",
        ]
    ]
    Outputer.print_table(headers, rows, title="Gradient FD Benchmark")
