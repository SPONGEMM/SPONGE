"""Full RHF gradient sanity check: SPONGE energy finite difference vs PySCF analytical.

NOTE: SPONGE uses float precision for energies, so finite difference accuracy
is limited to ~0.001 Ha/Bohr for small molecules. This test validates that
forces have the correct sign and magnitude, not exact numerical agreement.
"""

import math
import shutil
import tempfile

import numpy as np
import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    get_pyscf_reference_gradient,
    run_sponge_scf_energy_ha,
)
from benchmarks.utils import Outputer

GRAD_FD_TOL_HA_BOHR = 0.005  # float precision limits FD accuracy
BOHR_PER_ANGSTROM = 1.8897259886

GRAD_CASES = [
    ("h2", "HF", "sto-3g"),
    ("h2", "HF", "6-31g"),
]


def _sponge_fd_gradient(sponge_dir, model_chemistry, coords, h=0.002):
    """Compute gradient via central finite differences (h in Angstrom).

    Returns gradient in Ha/Bohr.
    """
    natm = len(coords)
    grad = np.zeros((natm, 3))
    for ia in range(natm):
        for d in range(3):
            cp = [list(c) for c in coords]
            cm = [list(c) for c in coords]
            cp[ia][d] += h
            cm[ia][d] -= h

            # Write perturbed coordinate and run
            for delta_coords, sign in [(cp, +1), (cm, -1)]:
                tmpdir = tempfile.mkdtemp()
                try:
                    for f in sponge_dir.iterdir():
                        if f.name != "coordinate.txt":
                            shutil.copy(f, tmpdir)
                    coord_path = tmpdir + "/coordinate.txt"
                    with open(coord_path, "w") as f:
                        f.write(f"{natm}\n")
                        for c in delta_coords:
                            f.write(f"{c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")
                        f.write("40.0 40.0 40.0 90.0 90.0 90.0\n")

                    from pathlib import Path

                    energy_ha = run_sponge_scf_energy_ha(
                        sponge_dir=Path(tmpdir),
                        model_chemistry=model_chemistry,
                        restricted=True,
                    )
                    if sign == +1:
                        ep = energy_ha
                    else:
                        em = energy_ha
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            # kcal/mol → Ha, Angstrom → Bohr
            grad[ia, d] = (ep - em) / (2 * h * BOHR_PER_ANGSTROM)

    return grad


@pytest.mark.parametrize(
    "case_name,method_name,basis_name",
    GRAD_CASES,
    ids=[f"{c}_{b}" for c, m, b in GRAD_CASES],
)
def test_gradient_fd(
    case_name, method_name, basis_name, statics_path, outputs_path
):
    ref_grad, ref_coords = get_pyscf_reference_gradient(
        statics_path=statics_path,
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
        restricted=True,
    )
    ref_grad = np.array(ref_grad)
    model_chemistry = f"{method_name}/{basis_name}"

    sponge_dir = statics_path / case_name / "sponge"
    fd_grad = _sponge_fd_gradient(sponge_dir, model_chemistry, ref_coords)

    max_err = float(np.max(np.abs(fd_grad - ref_grad)))
    status = "PASS" if max_err < GRAD_FD_TOL_HA_BOHR else "FAIL"

    headers = [
        "Case",
        "Method/Basis",
        "Max |FD - Ref| (Ha/Bohr)",
        "Tol",
        "Status",
    ]
    rows = [
        [
            case_name,
            model_chemistry,
            f"{max_err:.6f}",
            f"{GRAD_FD_TOL_HA_BOHR:.6f}",
            status,
        ]
    ]
    Outputer.print_table(headers, rows, title="Gradient FD vs PySCF")

    assert max_err < GRAD_FD_TOL_HA_BOHR, (
        f"FD gradient error {max_err:.6f} exceeds tolerance {GRAD_FD_TOL_HA_BOHR}"
    )
