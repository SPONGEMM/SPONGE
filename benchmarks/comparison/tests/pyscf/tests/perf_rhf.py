import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    run_sponge_vs_pyscf,
)
from benchmarks.utils import Outputer

PERF_RHF_TOL_HA = 1.0e-3

PERF_RHF_CASES = [
    ("benzene", "def2-qzvp", ["-qc_level_shift", "1.0", "-qc_diis_damp", "1.0"]),
    ("ace_ala4_nme", "def2-svp", ["-qc_level_shift", "1.0", "-qc_diis_damp", "1.0"]),
]


@pytest.mark.parametrize(
    "case_name,basis_name,sponge_args",
    PERF_RHF_CASES,
    ids=[f"{case}_{basis}" for case, basis, _ in PERF_RHF_CASES],
)
def test_perf_rhf(
    case_name, basis_name, sponge_args, statics_path, outputs_path, mpi_np
):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name="HF",
        basis_name=basis_name,
        restricted=True,
        run_prefix="perf_rhf",
        mpi_np=mpi_np,
        extra_sponge_args=sponge_args,
    )

    tol_kcal = PERF_RHF_TOL_HA * HARTREE_TO_KCAL_MOL
    headers = [
        "Case",
        "Method/Basis",
        "PySCF (kcal/mol)",
        "SPONGE (kcal/mol)",
        "|Delta| (kcal/mol)",
        "Tol (kcal/mol)",
        "Status",
    ]
    rows = [
        [
            case_name,
            f"HF/{basis_name}",
            f"{result['pyscf_energy_kcal_mol']:.6f}",
            f"{result['sponge_energy_kcal_mol']:.6f}",
            f"{result['abs_diff_kcal_mol']:.6f}",
            f"{tol_kcal:.6f}",
            "PASS" if result["abs_diff_ha"] <= PERF_RHF_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="Perf RHF vs PySCF")

    assert result["abs_diff_ha"] <= PERF_RHF_TOL_HA
