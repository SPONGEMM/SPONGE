import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    run_sponge_vs_pyscf,
)
from benchmarks.utils import Outputer

UHF_TOL_HA = 1.0e-2

UHF_CASE_BASIS = [
    ("no_doublet", "sto-3g"),
    ("no_doublet", "3-21g"),
    ("o_triplet", "6-31g"),
    ("o_triplet", "cc-pvdz"),
]


@pytest.mark.parametrize(
    "case_name,basis_name",
    UHF_CASE_BASIS,
    ids=[f"{case}_{basis}" for case, basis in UHF_CASE_BASIS],
)
def test_uhf(case_name, basis_name, statics_path, outputs_path, mpi_np):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name="HF",
        basis_name=basis_name,
        restricted=False,
        run_prefix="uhf",
        mpi_np=mpi_np,
    )

    tol_kcal = UHF_TOL_HA * HARTREE_TO_KCAL_MOL
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
            "PASS" if result["abs_diff_ha"] <= UHF_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="UHF vs PySCF")

    assert result["abs_diff_ha"] <= UHF_TOL_HA
