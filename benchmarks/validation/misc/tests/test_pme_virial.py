from constraint_probe import run_constraint_probe


def test_pme_conjugate_weights_and_updated_box_coefficients(tmp_path):
    run_constraint_probe(tmp_path, "pme_virial_probe")
