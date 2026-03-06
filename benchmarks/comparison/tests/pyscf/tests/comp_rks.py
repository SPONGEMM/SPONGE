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

RKS_CASES = [
    ("h2", "6-31g", "LDA"),
    ("he", "6-31g", "PBE"),
    ("oh2", "6-31g", "BLYP"),
    ("ch4", "6-31g", "PBE0"),
    ("co2", "6-31g", "B3LYP"),
]


def test_rks_functional_coverage():
    covered = {method for _case, _basis, method in RKS_CASES}
    assert covered == set(SUPPORTED_DFT_METHODS)


@pytest.mark.parametrize(
    "case_name,basis_name,method_name",
    RKS_CASES,
    ids=[f"{case}_{method}_{basis}" for case, basis, method in RKS_CASES],
)
def test_rks(
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
        restricted=True,
        run_prefix="rks",
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
    Outputer.print_table(headers, rows, title="RKS vs PySCF")

    assert result["abs_diff_ha"] <= tol_ha
