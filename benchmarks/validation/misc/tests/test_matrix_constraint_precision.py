from constraint_probe import run_constraint_probe


def test_matrix_velocity_projection_against_mass_metric_reference(tmp_path):
    run_constraint_probe(tmp_path, "matrix_constraint_precision_probe")
