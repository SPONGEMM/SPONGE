import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    SUPPORTED_DFT_METHODS,
    run_sponge_vs_pyscf,
)
from benchmarks.utils import Outputer

DFT_TOL_HA = {
    "LDA": 5.0e-3,
    "PBE": 5.0e-3,
    "BLYP": 1.0e-2,
    "PBE0": 5.0e-3,
    "B3LYP": 1.0e-2,
}

UKS_CASES = [
    ("o_triplet", "6-31g", "LDA"),
    ("o_triplet", "6-31g", "PBE"),
    ("o_triplet", "6-31g", "BLYP"),
    ("o_triplet", "6-31g", "PBE0"),
    ("o_triplet", "6-31g", "B3LYP"),
]


def test_uks_functional_coverage():
    covered = {method for _case, _basis, method in UKS_CASES}
    assert covered == set(SUPPORTED_DFT_METHODS)


@pytest.mark.parametrize(
    "case_name,basis_name,method_name",
    UKS_CASES,
    ids=[f"{case}_{method}_{basis}" for case, basis, method in UKS_CASES],
)
def test_uks(
    case_name,
    basis_name,
    method_name,
    statics_path,
    outputs_path,
    mpi_np,
):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
        restricted=False,
        run_prefix="uks",
        mpi_np=mpi_np,
    )

    tol_ha = DFT_TOL_HA[method_name]
    tol_kcal = tol_ha * HARTREE_TO_KCAL_MOL
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
            f"{method_name}/{basis_name}",
            f"{result['pyscf_energy_kcal_mol']:.6f}",
            f"{result['sponge_energy_kcal_mol']:.6f}",
            f"{result['abs_diff_kcal_mol']:.6f}",
            f"{tol_kcal:.6f}",
            "PASS" if result["abs_diff_ha"] <= tol_ha else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="UKS vs PySCF")

    assert result["abs_diff_ha"] <= tol_ha
