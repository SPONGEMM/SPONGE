"""ECP gradient validation: finite-difference sanity check.

Uses KH/def2-SVP + def2-ECP. Computes SPONGE central FD gradient and verifies
physical sanity (non-zero, Newton's 3rd law, correct symmetry).
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    run_sponge_scf_energy_ha,
)

BOHR_PER_ANGSTROM = 1.8897259886


def _make_ecp_sponge_dir(tmpdir, atoms, charge, multiplicity):
    """Create SPONGE input files for ECP gradient test."""
    sponge_dir = Path(tmpdir)
    natm = len(atoms)

    with open(sponge_dir / "qc_type.txt", "w") as f:
        f.write(f"{natm} {charge} {multiplicity}\n")
        for i, (sym, x, y, z) in enumerate(atoms):
            f.write(f"{i} {sym}\n")

    with open(sponge_dir / "coordinate.txt", "w") as f:
        f.write(f"{natm} 0.000000\n")
        for sym, x, y, z in atoms:
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")
        f.write("40.0 40.0 40.0\n")
        f.write("90.0 90.0 90.0\n")

    with open(sponge_dir / "mdin.txt", "w") as f:
        f.write("ECP grad test\nmode = nve\ndt = 0\nstep_limit = 0\n")
        f.write("print_zeroth_frame = 1\n")
        f.write("coordinate_in_file = coordinate.txt\n")
        f.write("qc_type_in_file = qc_type.txt\n")
        f.write("mass_in_file = mass.txt\n")
        f.write("charge_in_file = charge.txt\n")
        f.write("write_mdout_interval = 1\n")
        f.write("write_trajectory_interval = 1\n")

    MASS = {"H": 1.008, "K": 39.098}
    with open(sponge_dir / "mass.txt", "w") as f:
        f.write(f"{natm}\n")
        for sym, x, y, z in atoms:
            f.write(f"{MASS.get(sym, 1.0)}\n")

    with open(sponge_dir / "charge.txt", "w") as f:
        f.write(f"{natm}\n")
        for _ in atoms:
            f.write("0.0\n")

    return sponge_dir


def _fd_gradient(atoms, charge, mult, model_chemistry, h=0.001):
    """Central finite-difference gradient (Ha/Bohr)."""
    natm = len(atoms)
    grad = np.zeros((natm, 3))
    for ia in range(natm):
        for d in range(3):
            for sign in [+1, -1]:
                perturbed = list(atoms)
                coords = list(perturbed[ia])
                coords[1 + d] += sign * h
                perturbed[ia] = tuple(coords)

                tmpdir = tempfile.mkdtemp(prefix="ecp_fd_")
                try:
                    sd = _make_ecp_sponge_dir(tmpdir, perturbed, charge, mult)
                    e = run_sponge_scf_energy_ha(
                        sponge_dir=sd,
                        model_chemistry=model_chemistry,
                        restricted=(mult == 1),
                        extra_sponge_args=["-qc_ecp", "auto"],
                    )
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

                if sign == +1:
                    ep = e
                else:
                    em = e

            grad[ia, d] = (ep - em) / (2 * h * BOHR_PER_ANGSTROM)

    return grad


ECP_GRAD_CASES = [
    (
        "KH",
        [("K", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 2.244)],
        0,
        1,
        "HF/def2-svp",
    ),
]


@pytest.mark.parametrize(
    "name,atoms,charge,mult,model_chem",
    ECP_GRAD_CASES,
    ids=[c[0] for c in ECP_GRAD_CASES],
)
def test_ecp_gradient_fd(name, atoms, charge, mult, model_chem):
    """Validate ECP gradient via FD sanity check.

    Verifies that the energy surface with ECP is smooth and physically
    reasonable: non-zero gradient, Newton's 3rd law, correct symmetry.
    """
    fd_grad = _fd_gradient(atoms, charge, mult, model_chem, h=0.001)

    print(f"\n  {name} {model_chem} + ECP FD gradient (Ha/Bohr):")
    for ia, (sym, *_) in enumerate(atoms):
        print(
            f"    {sym}: [{fd_grad[ia, 0]:.6f}, {fd_grad[ia, 1]:.6f}, {fd_grad[ia, 2]:.6f}]"
        )

    max_grad = np.max(np.abs(fd_grad))
    print(f"    max|grad| = {max_grad:.6f} Ha/Bohr")

    assert max_grad > 0.001, f"Gradient too small: {max_grad}"

    # Newton's 3rd law
    force_sum = np.sum(fd_grad, axis=0)
    print(
        f"    force_sum = [{force_sum[0]:.6f}, {force_sum[1]:.6f}, {force_sum[2]:.6f}]"
    )
    assert np.max(np.abs(force_sum)) < 0.01, (
        f"Newton 3rd law violated: force sum = {force_sum}"
    )

    # Symmetry: x,y should be ~0 for z-axis diatomic
    xy_max = max(np.max(np.abs(fd_grad[:, 0])), np.max(np.abs(fd_grad[:, 1])))
    assert xy_max < 0.01, (
        f"x,y gradient too large for z-axis diatomic: {xy_max}"
    )
