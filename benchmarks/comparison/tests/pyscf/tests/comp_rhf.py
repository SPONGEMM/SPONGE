import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    run_sponge_vs_pyscf,
)
from benchmarks.utils import Outputer

RHF_TOL_HA = 1.0e-3

RHF_CASE_BASIS = [
    ("h2", "sto-3g"),
    ("he", "3-21g"),
    ("h2", "6-31g"),
    ("he", "6-31g*"),
    ("h2", "6-31g**"),
    ("he", "6-311g"),
    ("h2", "6-311g*"),
    ("he", "6-311g**"),
    ("h2", "def2-svp"),
    ("he", "def2-tzvp"),
    ("h2", "def2-tzvpp"),
    ("he", "def2-qzvp"),
    ("h2", "cc-pvdz"),
    ("he", "cc-pvtz"),
]


@pytest.mark.parametrize(
    "case_name,basis_name",
    RHF_CASE_BASIS,
    ids=[f"{case}_{basis}" for case, basis in RHF_CASE_BASIS],
)
def test_rhf(case_name, basis_name, statics_path, outputs_path, mpi_np):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name="HF",
        basis_name=basis_name,
        restricted=True,
        run_prefix="rhf",
        mpi_np=mpi_np,
    )

    tol_kcal = RHF_TOL_HA * HARTREE_TO_KCAL_MOL
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
            "PASS" if result["abs_diff_ha"] <= RHF_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="RHF vs PySCF")

    assert result["abs_diff_ha"] <= RHF_TOL_HA
