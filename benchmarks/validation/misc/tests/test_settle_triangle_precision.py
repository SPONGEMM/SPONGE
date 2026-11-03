import pytest

from constraint_probe import run_constraint_probe


@pytest.mark.parametrize(
    "target",
    [
        "settle_triangle_precision_probe",
        "settle_triangle_precision_strict_probe",
    ],
)
def test_settle_triangle_translation_and_independent_position_solve(
    tmp_path, target
):
    run_constraint_probe(tmp_path, target)
