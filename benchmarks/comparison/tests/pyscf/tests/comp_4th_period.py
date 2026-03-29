import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    run_sponge_vs_pyscf,
)
from benchmarks.utils import Outputer

TOL_HA = 1.0e-2

CASES_4TH = [
    # (case_name, method, basis, restricted)
    ("br_anion", "HF", "ma-def2-svp", True),
    ("fe_quintet", "HF", "6-31++g", False),
]


@pytest.mark.parametrize(
    "case_name,method_name,basis_name,restricted",
    CASES_4TH,
    ids=[f"{c}_{m}_{b}" for c, m, b, _r in CASES_4TH],
)
def test_4th_period(
    case_name,
    method_name,
    basis_name,
    restricted,
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
        restricted=restricted,
        run_prefix="p4",
        mpi_np=mpi_np,
    )

    tol_kcal = TOL_HA * HARTREE_TO_KCAL_MOL
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
            "PASS" if result["abs_diff_ha"] <= TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="4th-Period vs PySCF")

    assert result["abs_diff_ha"] <= TOL_HA
