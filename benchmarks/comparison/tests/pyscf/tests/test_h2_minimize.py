"""H2 bond length optimization: SPONGE minimization vs PySCF reference."""

import math

import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    get_pyscf_reference_minimize,
    prepare_output_case,
    run_sponge_scf_energy_ha,
)
from benchmarks.utils import Outputer, Runner

BOND_LENGTH_TOL_A = 0.02  # Angstrom
ENERGY_TOL_HA = 0.001  # Hartree

MINIMIZE_CASES = [
    ("h2_min_sto3g", "HF", "sto-3g"),
]


def _parse_sponge_minimize_result(sponge_dir):
    """Parse restart_coordinate.txt for final geometry and mdout.txt for energy."""
    restart = sponge_dir / "restart_coordinate.txt"
    coords = []
    with open(restart) as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            coords.append([float(x) for x in parts])
        elif len(parts) >= 6:
            break

    mdout = sponge_dir / "mdout.txt"
    last_qc = None
    with open(mdout) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    int(parts[0])
                    last_qc = float(parts[3])
                except (ValueError, IndexError):
                    pass

    box_z = 40.0
    if len(coords) >= 2:
        dz = coords[1][2] - coords[0][2]
        if dz > box_z / 2:
            dz -= box_z
        if dz < -box_z / 2:
            dz += box_z
        bond_len = math.sqrt(
            (coords[1][0] - coords[0][0]) ** 2
            + (coords[1][1] - coords[0][1]) ** 2
            + dz**2
        )
    else:
        bond_len = None

    energy_ha = last_qc / HARTREE_TO_KCAL_MOL if last_qc else None
    return bond_len, energy_ha


@pytest.mark.parametrize(
    "case_name,method_name,basis_name",
    MINIMIZE_CASES,
    ids=[f"{c}_{b}" for c, m, b in MINIMIZE_CASES],
)
def test_h2_minimize(
    case_name, method_name, basis_name, statics_path, outputs_path
):
    ref_bond, ref_energy = get_pyscf_reference_minimize(
        statics_path=statics_path,
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
    )

    _run_dir, sponge_dir = prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        run_tag=f"minimize_{case_name}",
    )

    Runner.run_sponge(
        sponge_dir,
        mdin_name="mdin.txt",
    )

    sponge_bond, sponge_energy = _parse_sponge_minimize_result(sponge_dir)

    r_err = abs(sponge_bond - ref_bond) if sponge_bond else float("inf")
    e_err = abs(sponge_energy - ref_energy) if sponge_energy else float("inf")

    headers = [
        "Case",
        "Property",
        "PySCF Ref",
        "SPONGE",
        "|Delta|",
        "Tol",
        "Status",
    ]
    rows = [
        [
            case_name,
            "Bond (A)",
            f"{ref_bond:.4f}",
            f"{sponge_bond:.4f}" if sponge_bond else "N/A",
            f"{r_err:.4f}",
            f"{BOND_LENGTH_TOL_A:.4f}",
            "PASS" if r_err < BOND_LENGTH_TOL_A else "FAIL",
        ],
        [
            "",
            "Energy (Ha)",
            f"{ref_energy:.8f}",
            f"{sponge_energy:.8f}" if sponge_energy else "N/A",
            f"{e_err:.8f}",
            f"{ENERGY_TOL_HA:.8f}",
            "PASS" if e_err < ENERGY_TOL_HA else "FAIL",
        ],
    ]
    Outputer.print_table(headers, rows, title="H2 Minimization vs PySCF")

    assert r_err < BOND_LENGTH_TOL_A, (
        f"Bond length error {r_err:.4f} A exceeds tolerance {BOND_LENGTH_TOL_A}"
    )
    assert e_err < ENERGY_TOL_HA, (
        f"Energy error {e_err:.8f} Ha exceeds tolerance {ENERGY_TOL_HA}"
    )
