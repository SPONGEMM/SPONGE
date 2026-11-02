from __future__ import annotations

import math
import statistics

import h5py
import pytest

from benchmarks.bundled_io.ab_statistics import (
    StatisticalEquivalencePolicy,
    block_means,
    compare_replicas,
    holm_correct_equivalence_family,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    TRAJECTORY_REL,
    _assert_chunk_boundary_layout,
    _assert_nonfinite_patterns_match,
    _assert_numeric_sequences_close,
    _assert_periodic_positions_close,
    _chunk_boundary_cases,
    _deterministic_tolerance,
    _statistical_policy,
    _trajectory_feature_names_for_inference,
)
from benchmarks.bundled_io.trajectory_statistics import (
    trajectory_observable_series,
)

POLICY = StatisticalEquivalencePolicy(
    burn_in_frames=2,
    block_size=2,
    minimum_blocks_per_replica=3,
    confidence_z=3.0,
    relative_margin=0.05,
    absolute_margin=0.01,
    maximum_std_ratio=1.5,
)

MUTATION_POLICY = StatisticalEquivalencePolicy(
    burn_in_frames=0,
    block_size=2,
    minimum_blocks_per_replica=3,
    confidence_z=2.0,
    relative_margin=0.01,
    absolute_margin=0.05,
    maximum_std_ratio=1.5,
)


def test_block_means_discards_warmup_and_uses_non_overlapping_blocks():
    assert block_means([99.0, 99.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0], POLICY) == [
        2.0,
        6.0,
        10.0,
    ]


def test_statistical_equivalence_accepts_matching_replica_ensembles():
    legacy = [
        [0.0, 0.0, 9.9, 10.1, 10.0, 10.2, 9.8, 10.0],
        [0.0, 0.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.0],
    ]
    bundled = [
        [0.0, 0.0, 10.0, 10.2, 9.9, 10.1, 9.8, 10.0],
        [0.0, 0.0, 10.1, 9.9, 10.0, 10.0, 10.1, 9.9],
    ]

    result = compare_replicas("temperature", legacy, bundled, POLICY)

    assert result["legacy_block_count"] == 6
    assert result["bundled_block_count"] == 6
    assert result["confidence_bound"] <= result["practical_margin"]


def test_statistical_equivalence_rejects_mean_shift_outside_margin():
    legacy = [[0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]
    bundled = [[0.0, 0.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]]

    with pytest.raises(
        AssertionError, match="mean is not statistically equivalent"
    ):
        compare_replicas("temperature", legacy, bundled, POLICY)


def test_statistical_equivalence_rejects_changed_fluctuations():
    legacy = [[0.0, 0.0, 9.8, 9.8, 10.2, 10.2, 10.0, 10.0]]
    bundled = [[0.0, 0.0, 9.9, 9.9, 10.1, 10.1, 10.0, 10.0]]

    with pytest.raises(AssertionError, match="fluctuation mismatch"):
        compare_replicas("temperature", legacy, bundled, POLICY)


def test_fluctuation_ratio_has_no_cliff_at_absolute_mean_margin():
    policy = StatisticalEquivalencePolicy(
        burn_in_frames=0,
        block_size=2,
        minimum_blocks_per_replica=3,
        confidence_z=3.0,
        relative_margin=0.0,
        absolute_margin=0.01,
        maximum_std_ratio=1.5,
    )
    legacy = [[-0.012, -0.012, 0.0, 0.0, 0.012, 0.012]]
    bundled = [[-0.0124, -0.0124, 0.0, 0.0, 0.0124, 0.0124]]

    result = compare_replicas("velocity quantile", legacy, bundled, policy)

    assert result["std_ratio"] == pytest.approx(0.0124 / 0.012)
    assert result["std_ratio"] < policy.maximum_std_ratio


def test_statistical_equivalence_reports_tost_p_value():
    replicas = [[0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]

    result = compare_replicas("temperature", replicas, replicas, POLICY)

    assert result["equivalence_p_value"] == 0.0


def test_replica_inference_uses_independent_replica_means_for_sem():
    policy = StatisticalEquivalencePolicy(
        burn_in_frames=0,
        block_size=2,
        minimum_blocks_per_replica=3,
        confidence_z=3.0,
        relative_margin=0.0,
        absolute_margin=1.0,
        maximum_std_ratio=1.5,
        inference_unit="replica",
    )
    legacy = [[9.0, 9.0, 10.0, 10.0, 11.0, 11.0]] * 3
    bundled = [
        [value + delta for value in replica]
        for replica, delta in zip(legacy, (0.1, -0.1, 0.2))
    ]

    result = compare_replicas("energy", legacy, bundled, policy)

    expected_deltas = [0.1, -0.1, 0.2]
    assert result["inference_unit"] == "replica"
    assert result["paired_block_count"] == 9
    assert result["paired_replica_count"] == 3
    assert result["inference_sample_count"] == 3
    assert result["mean_delta_sem"] == pytest.approx(
        statistics.stdev(expected_deltas) / math.sqrt(3)
    )


def test_correlated_blocks_cannot_masquerade_as_independent_replicas():
    common = dict(
        burn_in_frames=0,
        block_size=2,
        minimum_blocks_per_replica=100,
        confidence_z=3.0,
        relative_margin=0.0,
        absolute_margin=0.05,
        maximum_std_ratio=1.5,
    )
    legacy = [[8.0, 10.0, 10.0, 12.0] * 50] * 2
    bundled = [
        [value + 0.2 for value in legacy[0]],
        [value - 0.2 for value in legacy[1]],
    ]

    block_result = compare_replicas(
        "energy",
        legacy,
        bundled,
        StatisticalEquivalencePolicy(**common, inference_unit="block"),
    )
    assert block_result["confidence_bound"] < 0.05

    with pytest.raises(
        AssertionError, match="mean is not statistically equivalent"
    ):
        compare_replicas(
            "energy",
            legacy,
            bundled,
            StatisticalEquivalencePolicy(**common, inference_unit="replica"),
        )


def test_holm_correction_rejects_observable_that_only_passes_uncorrected_alpha():
    results = {
        "potential": {"equivalence_p_value": 0.03},
        "temperature": {"equivalence_p_value": 0.04},
    }

    with pytest.raises(AssertionError, match="after Holm correction"):
        holm_correct_equivalence_family("mdout", results, alpha=0.05)


def test_holm_correction_records_auditable_rank_and_threshold():
    results = {
        "potential": {"equivalence_p_value": 0.001},
        "temperature": {"equivalence_p_value": 0.01},
    }

    corrected = holm_correct_equivalence_family("mdout", results, alpha=0.05)

    assert corrected["potential"]["holm_rank"] == 1
    assert corrected["potential"]["holm_threshold"] == pytest.approx(0.025)
    assert corrected["temperature"]["holm_rank"] == 2


def _compare_trajectory_family(
    dataset: str,
    legacy_replicas: list[list[float]],
    bundled_replicas: list[list[float]],
    shape: tuple[int, ...],
) -> None:
    legacy = [
        trajectory_observable_series(dataset, values, shape)
        for values in legacy_replicas
    ]
    bundled = [
        trajectory_observable_series(dataset, values, shape)
        for values in bundled_replicas
    ]
    results = {}
    for feature in legacy[0]:
        results[feature] = compare_replicas(
            feature,
            [replica[feature] for replica in legacy],
            [replica[feature] for replica in bundled],
            MUTATION_POLICY,
        )
    holm_correct_equivalence_family("trajectory", results, alpha=0.05)


def _position_frames(frame_count: int) -> list[float]:
    values = []
    for frame in range(frame_count):
        values.extend((float(frame), 0.0, 0.0, 10.0, 0.0, 0.0))
    return values


def test_energy_offset_mutation_is_rejected():
    legacy = [[10.0, 10.1, 9.9, 10.0, 10.1, 9.9]]
    bundled = [[value + 1.0 for value in legacy[0]]]

    with pytest.raises(
        AssertionError, match="mean is not statistically equivalent"
    ):
        compare_replicas("potential", legacy, bundled, MUTATION_POLICY)


def test_variance_increase_mutation_is_rejected():
    legacy = [[9.9, 9.9, 10.1, 10.1, 10.0, 10.0]]
    bundled = [[9.5, 9.5, 10.5, 10.5, 10.0, 10.0]]

    with pytest.raises(AssertionError):
        compare_replicas("temperature", legacy, bundled, MUTATION_POLICY)


def test_atom_permutation_mutation_is_rejected():
    legacy = _position_frames(6)
    bundled = []
    for offset in range(0, len(legacy), 6):
        bundled.extend(legacy[offset + 3 : offset + 6])
        bundled.extend(legacy[offset : offset + 3])

    with pytest.raises(AssertionError):
        _compare_trajectory_family(
            "/particles/all/position/value", [legacy], [bundled], (6, 2, 3)
        )


def test_local_force_offset_mutation_is_rejected():
    legacy = []
    bundled = []
    for frame in range(6):
        legacy.extend((float(frame), 0.0, 0.0, -float(frame), 0.0, 0.0))
        bundled.extend((float(frame), 0.0, 0.0, -float(frame) + 1.0, 0.0, 0.0))

    with pytest.raises(AssertionError):
        _compare_trajectory_family(
            "/particles/all/force/value", [legacy], [bundled], (6, 2, 3)
        )


def test_missing_nonfinite_mutation_is_rejected():
    with pytest.raises(AssertionError, match="non-finite mismatch"):
        _assert_numeric_sequences_close(
            "potential",
            [1.0, float("nan"), float("inf"), -float("inf")],
            [1.0, 0.0, float("inf"), -float("inf")],
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )


@pytest.mark.parametrize(
    ("legacy", "bundled"),
    [
        ([float("inf")], [-float("inf")]),
        ([-float("inf")], [float("inf")]),
        ([float("nan")], [float("inf")]),
    ],
)
def test_nonfinite_kind_and_sign_mutations_are_rejected(legacy, bundled):
    with pytest.raises(AssertionError, match="non-finite mismatch"):
        _assert_nonfinite_patterns_match("potential", legacy, bundled)


def test_frame_schedule_mutation_is_rejected():
    with pytest.raises(AssertionError, match="mismatch at index 2"):
        _assert_numeric_sequences_close(
            "step schedule",
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 30.0],
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-12,
        )


def test_position_and_velocity_observables_use_atom_masses():
    position = trajectory_observable_series(
        "/particles/all/position/value",
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        (1, 2, 3),
        atom_weights=[1.0, 3.0],
    )
    velocity = trajectory_observable_series(
        "/particles/all/velocity/value",
        [1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        (1, 2, 3),
        atom_weights=[1.0, 3.0],
    )

    assert position["center_of_mass_component_0"] == [3.0]
    assert velocity["mass_weighted_mean_squared_speed"] == [7.0]


def test_position_observables_use_periodic_minimum_images():
    observables = trajectory_observable_series(
        "/particles/all/position/value",
        [9.9, 0.0, 0.0, 0.1, 0.0, 0.0],
        (2, 1, 3),
        box_values=[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
        box_shape=(1, 3, 3),
    )

    assert observables["atom_0_squared_displacement"] == [
        pytest.approx(0.0),
        pytest.approx(0.04),
    ]


def test_box_observables_include_matrix_volume_lengths_and_angles():
    observables = trajectory_observable_series(
        "/particles/all/box/edges/value",
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        (1, 3, 3),
    )

    assert observables["matrix_0_0"] == [1.0]
    assert observables["volume"] == [1.0]
    assert observables["length_2"] == [1.0]
    assert observables["angle_0_1"] == [pytest.approx(math.pi / 2.0)]


def test_replica_inference_uses_ensemble_not_per_atom_trajectory_features():
    observables = {
        "component_rms": [1.0, 2.0],
        "norm_median": [1.0, 2.0],
        "atom_1000_norm": [1.0, 2.0],
    }

    assert _trajectory_feature_names_for_inference(observables, "replica") == {
        "component_rms",
        "norm_median",
    }
    assert _trajectory_feature_names_for_inference(observables, "block") == set(
        observables
    )

    generated = trajectory_observable_series(
        "/particles/all/force/value",
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        (1, 2, 3),
        include_atom_features=False,
    )
    assert "component_rms" in generated
    assert not any(name.startswith("atom_") for name in generated)


def test_comparator_policies_are_quantity_specific():
    assert (
        _statistical_policy("pressure").absolute_margin
        != _statistical_policy("potential").absolute_margin
    )
    assert _deterministic_tolerance("position") != _deterministic_tolerance(
        "force"
    )
    assert _deterministic_tolerance("step") == (0.0, 1.0e-12)


def test_deterministic_force_tolerance_rejects_material_local_offset():
    relative, absolute = _deterministic_tolerance("force")
    with pytest.raises(AssertionError, match="mismatch at index 0"):
        _assert_numeric_sequences_close(
            "force mutation",
            [0.0],
            [1.0e-4],
            relative_tolerance=relative,
            absolute_tolerance=absolute,
        )


def test_deterministic_position_comparison_accepts_only_periodic_image_shift():
    box = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    relative, absolute = _deterministic_tolerance("position")
    _assert_periodic_positions_close(
        "periodic image",
        [9.9, 0.0, 0.0],
        [-0.1, 0.0, 0.0],
        (1, 1, 3),
        box,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
    )
    with pytest.raises(AssertionError, match="periodic mismatch"):
        _assert_periodic_positions_close(
            "position mutation",
            [9.9, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            (1, 1, 3),
            box,
            relative_tolerance=relative,
            absolute_tolerance=absolute,
        )


def test_nonfinite_pattern_check_does_not_force_finite_frames_to_match():
    _assert_nonfinite_patterns_match(
        "stochastic potential",
        [1.0, float("nan"), 2.0],
        [1.5, float("nan"), 2.5],
    )


def test_chunk_boundary_layout_rejects_frame_and_shard_mutations(tmp_path):
    case = _chunk_boundary_cases()[0]
    trajectory = tmp_path / TRAJECTORY_REL
    trajectory.parent.mkdir(parents=True)
    with h5py.File(trajectory, "w") as handle:
        handle.create_dataset("/parameters/sponge/output/frame_count", data=[3])
        handle.create_dataset(
            "/particles/all/position/value", shape=(3, 2, 3), dtype="f4"
        )
    shard_root = trajectory.parent / "ab.spg.shards"
    shard_root.mkdir()
    (shard_root / "segment_000000.spg.h5md").touch()

    assert _assert_chunk_boundary_layout(case, tmp_path) == {
        "frame_count": 3,
        "shard_count": 1,
        "expected_shard_count": 1,
    }

    with h5py.File(trajectory, "r+") as handle:
        handle["/parameters/sponge/output/frame_count"][0] = 4
    with pytest.raises(AssertionError, match="frame count mismatch"):
        _assert_chunk_boundary_layout(case, tmp_path)

    with h5py.File(trajectory, "r+") as handle:
        handle["/parameters/sponge/output/frame_count"][0] = 3
    (shard_root / "segment_000001.spg.h5md").touch()
    with pytest.raises(AssertionError, match="shard count mismatch"):
        _assert_chunk_boundary_layout(case, tmp_path)
