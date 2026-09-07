from constraint_probe import run_constraint_probe


def test_settle_pair_rejects_invalid_update_before_writing_state(tmp_path):
    run_constraint_probe(tmp_path, "settle_pair_guard_probe")
