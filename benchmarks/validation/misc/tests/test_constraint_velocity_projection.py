from constraint_probe import run_constraint_probe


def test_velocity_only_constraint_projection_converges_without_moving_crd(
    tmp_path,
):
    run_constraint_probe(tmp_path, "constraint_velocity_projection_probe")
