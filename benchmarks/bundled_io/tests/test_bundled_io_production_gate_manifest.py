from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from benchmarks.bundled_io.ab_contracts import (
    SCOPED_EQUIVALENCE_STATEMENT,
    VDS_CROSS_PROCESS_RESUME_CONTRACT,
    AssertionEvidence,
    ContractSpec,
    audit_input_evolution_contracts,
    load_contract_registry,
    load_implementation_inventory,
    registry_summary,
    required_scope_exclusions,
    validate_contract_registry,
    validate_implementation_inventory,
    validate_scope_boundary,
    validated_passed_contract_levels,
)
from benchmarks.bundled_io.execution_matrix import (
    MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS,
    MINIMUM_PRODUCTION_STATISTICAL_REPLICAS,
    ProductionRun,
    evaluate_promotion_readiness,
    load_execution_matrix,
    load_production_run_history,
    validate_execution_matrix,
)
from benchmarks.bundled_io.input_semantics import (
    REQUIRED_INPUT_SEMANTIC_CONTRACTS,
    InputSemanticSpec,
    assert_module_semantics,
)
from benchmarks.bundled_io.promotion_evidence import (
    REQUIRED_COMPARATOR_MUTATION_NODE_COUNTS,
    REQUIRED_COMPARATOR_MUTATION_TESTS,
    append_production_run_history,
    derive_production_run,
    inspect_clean_source_tree,
    merge_matrix_evidence,
    write_comparator_mutation_report,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_execution_matrix import (
    MATRIX_RUNTIME_CASES,
    _matrix_h5_families,
    _matrix_sample_plan,
    _runtime_keys,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    FAILURE_SEMANTICS,
    FOCUSED_CONSTRAINT_SIDECAR_FIXTURE,
    FOCUSED_CONSTRAINT_TYPED_FIXTURE,
    FOCUSED_CORE_TOPOLOGY_FIXTURE,
    FOCUSED_CUSTOM_PAIR_FIXTURE,
    FOCUSED_EAM_FUNCFL_FIXTURE,
    FOCUSED_EAM_SETFL_FIXTURE,
    FOCUSED_EDIP_FIXTURE,
    FOCUSED_EXCLUSIONS_FIXTURE,
    FOCUSED_GB_HYBRID_FIXTURE,
    FOCUSED_GB_NATIVE_FIXTURE,
    FOCUSED_IMPROPER_CONVERTED_ALIAS_FIXTURE,
    FOCUSED_IMPROPER_CONVERTED_CANONICAL_FIXTURE,
    FOCUSED_IMPROPER_NATIVE_FIXTURE,
    FOCUSED_LJ_SOFT_CORE_FIXTURE,
    FOCUSED_NB14_EXTRA_FIXTURE,
    FOCUSED_NB14_SCALED_FIXTURE,
    FOCUSED_POSITIONAL_RESTRAINT_FIXTURE,
    FOCUSED_RESIDUE_COM_RES_FIXTURE,
    FOCUSED_RESIDUE_SIDECAR_FIXTURE,
    FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE,
    FOCUSED_RESIDUE_TYPED_PBC_FIXTURE,
    FOCUSED_SITS_NK_TYPED_RESTART_FIXTURE,
    FOCUSED_SITS_TYPED_CONFIG_FIXTURE,
    FOCUSED_SITS_TYPED_INACTIVE_FIXTURE,
    FOCUSED_STEERING_CV_SIDECAR_FIXTURE,
    FOCUSED_STEERING_CV_TYPED_FIXTURE,
    FOCUSED_SUBSYSTEM_DIVISION_FIXTURE,
    FOCUSED_SW_SIDECAR_FIXTURE,
    FOCUSED_SW_TYPED_FIXTURE,
    FOCUSED_TERSOFF_SIDECAR_FIXTURE,
    FOCUSED_TERSOFF_TYPED_FIXTURE,
    FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
    FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
    FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
    INPUT_SEMANTIC_SPECS_BY_CASE,
    MDINFO_CONTRACT_KEYS,
    NONFINITE_PROPAGATION,
    NORMAL_SUCCESS,
    OUTPUT_FAMILY_COMBINATIONS,
    OUTPUT_WRITER_FAILURE_POINTS,
    PROFILE_LIMITS,
    RERUN_INPUT_SEMANTIC_SPECS,
    SUPPORTED_TOPOLOGY_SCHEMA_VERSIONS,
    AbRun,
    _assert_complete_prefix_noop_layout,
    _assert_constraint_projection_oracle,
    _assert_core_topology_payload_response,
    _assert_exclusion_coulomb_oracle,
    _assert_finite_values,
    _assert_focused_improper_oracle,
    _assert_gb_force_oracle,
    _assert_nontrivial_equivalent_forces,
    _assert_nonzero_dt_evolution,
    _assert_numeric_sequences_close,
    _assert_rerun_direct_payload,
    _assert_rerun_optional_presence,
    _assert_residue_com_res_virial_oracle,
    _assert_residue_pbc_mapping_oracle,
    _assert_sits_nk_typed_restart_oracle,
    _assert_sits_typed_control_response,
    _assert_steering_cv_oracle,
    _assert_subsystem_partition_response,
    _assert_sw_pair_three_body_oracle,
    _assert_tersoff_angular_oracle,
    _assert_virtual_atom_oracle,
    _attach_finite_scan,
    _audit_output_family_cases,
    _case_output_classification,
    _cases_for_profile,
    _expected_rerun_frame_indices,
    _h5_string_values,
    _insert_root_toml_keys,
    _parse_mdinfo_key_values,
    _restart_continuation_source,
    _scan_success_outputs,
    _summarize_finite_scans,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PIXI_TOML = REPO_ROOT / "pixi.toml"
H5_BUNDLE_RUNNER = REPO_ROOT / "tests" / "h5_bundle" / "run_h5_bundle_tests.sh"
VDS_WRITER_TEST = (
    REPO_ROOT
    / "tests"
    / "h5_bundle"
    / "test_vds_trajectory_writer_with_mock_backend.cpp"
)
VDS_TERMINAL_REPAIR_TEST = (
    REPO_ROOT / "tests/h5_bundle/test_h5_vds_terminal_resume_smoke.cpp"
)
OUTPUT_PLAN_TEST = REPO_ROOT / "tests/h5_bundle/test_h5_output_plan.cpp"
WRITER_TEST = (
    REPO_ROOT / "tests/h5_bundle/test_h5md_writers_with_mock_backend.cpp"
)
MODULE_TEST = (
    REPO_ROOT / "tests/h5_bundle/test_module_h5_mappings_with_mock_backend.cpp"
)
HIGHFIVE_TEST = REPO_ROOT / "tests/h5_bundle/test_highfive_backend_io.cpp"
AB_SHADOW_WORKFLOW = REPO_ROOT / ".github/workflows/bundled-io-ab-shadow.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/release.yml"
XPONGE_RUNTIME_PATCH = (
    REPO_ROOT / "benchmarks/bundled_io/contracts/xponge_converter_runtime.patch"
)
MATRIX_FIXTURE_ROOT = REPO_ROOT / "benchmarks/bundled_io/fixtures/tip3p_matrix"
FULL_CONTRACT_FIXTURE_ROOT = (
    REPO_ROOT / "tests/h5_bundle/fixtures/input_matrix/full_contract_rerun"
)


def _dev_tasks() -> dict[str, object]:
    with PIXI_TOML.open("rb") as handle:
        data = tomllib.load(handle)
    return data["feature"]["dev"]["tasks"]


def _task_command(tasks: dict[str, object], name: str) -> str:
    value = tasks[name]
    if isinstance(value, str):
        return value
    assert isinstance(value, dict), f"{name} must be a command task"
    command = value.get("cmd")
    assert isinstance(command, str), f"{name} must provide cmd"
    return command


def _task_depends(tasks: dict[str, object], name: str) -> list[str]:
    value = tasks[name]
    assert isinstance(value, dict), f"{name} must be a dependency task"
    depends = value.get("depends-on")
    assert isinstance(depends, list), f"{name} must use depends-on"
    assert all(isinstance(item, str) for item in depends)
    return depends


def test_bundled_io_production_gate_keeps_all_required_stages():
    tasks = _dev_tasks()

    assert _task_depends(tasks, "smoke-bundled-io") == [
        "smoke-bundled-io-contract",
        "smoke-bundled-io-runtime",
    ]
    assert _task_depends(tasks, "smoke-bundled-io-production") == [
        "smoke-bundled-io-contract",
        "smoke-bundled-io-runtime",
        "smoke-bundled-io-staged",
        "smoke-gpu-medium",
    ]

    contract = _task_command(tasks, "smoke-bundled-io-contract")
    assert "test_bundled_io_contract_smoke.py" in contract
    assert "test_bundled_io_ab_statistics.py" in contract
    assert "test_bundled_io_ab_contract_registry.py" in contract
    assert "test_bundled_io_contract_inventory.py" in contract
    assert "test_bundled_io_production_gate_manifest.py" in contract
    assert "test_bundled_io_ab_production.py" not in contract
    assert "test-smoke-runtime" in _task_command(
        tasks, "smoke-bundled-io-runtime"
    )
    assert "run_h5_bundle_tests.sh staged" in _task_command(
        tasks, "smoke-bundled-io-staged"
    )


def test_gpu_medium_smoke_uses_real_validation_and_performance_scenarios():
    tasks = _dev_tasks()

    assert _task_depends(tasks, "smoke-gpu-medium") == [
        "smoke-gpu-medium-preflight",
        "smoke-gpu-medium-validation",
        "smoke-gpu-medium-reaxff",
        "smoke-gpu-medium-nonortho",
    ]

    preflight = _task_command(tasks, "smoke-gpu-medium-preflight")
    assert "test_cuda_runtime_preflight.py" in preflight

    validation = _task_command(tasks, "smoke-gpu-medium-validation")
    assert (
        "benchmarks/validation/thermostat/tests/test_thermostat.py"
        in validation
    )
    assert "middle_langevin" in validation
    assert "benchmarks/validation/cv/tests/test_dihedral.py" in validation
    assert "benchmarks/validation/misc/tests/test_softcore.py" in validation

    reaxff = _task_command(tasks, "smoke-gpu-medium-reaxff")
    assert "benchmarks/performance/reaxff/tests/test_petn_nve_perf.py" in reaxff
    assert "--steps=1000" in reaxff

    nonortho = _task_command(tasks, "smoke-gpu-medium-nonortho")
    assert (
        "benchmarks/performance/nonortho/tests/test_long_run_rdf.py" in nonortho
    )
    assert "--mode=NVT" in nonortho
    assert "--steps=2000" in nonortho


def test_ab_bundled_io_gates_keep_medium_and_production_profiles():
    tasks = _dev_tasks()

    medium = _task_command(tasks, "ab-bundled-io-medium-runtime")
    assert "SPONGE_BUNDLED_IO_AB_PROFILE=medium" in medium
    assert "test_bundled_io_ab_production.py" in medium

    production = _task_command(tasks, "ab-bundled-io-production-runtime")
    assert "SPONGE_BUNDLED_IO_AB_PROFILE=production" in production
    assert "test_bundled_io_ab_production.py" in production

    assert _task_depends(tasks, "ab-bundled-io-medium") == [
        "smoke-bundled-io-contract",
        "smoke-bundled-io-runtime",
        "ab-bundled-io-medium-runtime",
    ]
    assert _task_depends(tasks, "ab-bundled-io-production") == [
        "smoke-bundled-io-production",
        "ab-bundled-io-production-runtime",
    ]


def test_ab_production_harness_has_executable_contract_coverage():
    contracts = load_contract_registry()
    cases = _cases_for_profile()
    summary = validate_contract_registry(contracts, cases)
    validate_implementation_inventory(
        contracts, load_implementation_inventory()
    )

    assert set(PROFILE_LIMITS) == {"medium", "production"}
    assert PROFILE_LIMITS["medium"] == {
        "normal_step_limit": 1000,
        "normal_interval": 20,
        "normal_replicas": 10,
        "normal_burn_in_frames": 10,
        "normal_block_size": 8,
        "normal_relative_margin": 5.0e-2,
        "normal_absolute_margin": 1.5e-1,
        "rerun_frame_limit": 2,
    }
    assert PROFILE_LIMITS["production"] == {
        "normal_step_limit": 10000,
        "normal_interval": 100,
        "normal_replicas": 5,
        "normal_burn_in_frames": 20,
        "normal_block_size": 10,
        "normal_relative_margin": 1.0e-2,
        "normal_absolute_margin": 5.0e-2,
        "rerun_frame_limit": 2,
    }
    assert {case.name for case in cases} == {
        "normal_structural_restart_continuation",
        "normal_core_h5_output",
        "normal_core_topology_payload_sensitivity",
        "normal_bond_payload_sensitivity",
        "normal_angle_payload_sensitivity",
        "normal_urey_bradley_payload_sensitivity",
        "normal_dihedral_payload_sensitivity",
        "normal_listed_payload_sensitivity",
        "normal_nb14_scaled_nonzero",
        "normal_nb14_extra_nonzero",
        "normal_nhc_dynamic_restart_continuation",
        "normal_bussi_dynamic_restart_continuation",
        "normal_pressure_barostat_dynamic_restart_continuation",
        "normal_middle_langevin_dynamic_restart_continuation",
        "normal_andersen_dynamic_restart_continuation",
        "normal_monte_carlo_barostat_dynamic_restart_continuation",
        "normal_meta_protocol_full_restart_continuation",
        "normal_sits_ff19sb_cmap_peptide",
        "normal_edip_nonzero",
        "normal_reaxff_payload_sensitivity",
        "normal_eam_funcfl_nonzero",
        "normal_eam_setfl_nonzero",
        "normal_positional_restraint_typed_nonzero",
        "normal_positional_restraint_sidecar_nonzero",
        "normal_soft_wall_typed_nonzero",
        "normal_soft_wall_sidecar_nonzero",
        "normal_cv_restraint_typed_nonzero",
        "normal_cv_restraint_sidecar_nonzero",
        "normal_sw_sidecar_pair_three_body",
        "normal_sw_typed_pair_three_body",
        "normal_tersoff_sidecar_angular",
        "normal_tersoff_typed_angular",
        "normal_custom_pair_nonzero",
        "normal_exclusions_coulomb_oracle",
        "normal_residue_sidecar_pbc_mapping",
        "normal_residue_sidecar_com_res_virial",
        "normal_residue_typed_pbc_mapping",
        "normal_residue_typed_com_res_virial",
        "normal_gb_hybrid_nonzero",
        "normal_gb_native_nonzero",
        "normal_improper_converted_canonical_nonzero",
        "normal_improper_converted_alias_nonzero",
        "normal_improper_native_nonzero",
        "normal_lj_soft_core_nonzero",
        "normal_subsystem_division_partition",
        "normal_virtual_atoms_all_types",
        "normal_virtual_atoms_pbc_boundary",
        "normal_virtual_atoms_plural_alias",
        "normal_constraint_sidecar_projection",
        "normal_constraint_typed_projection",
        "normal_sits_nk_typed_restart_nonzero",
        "normal_sits_typed_configuration_nonzero",
        "normal_sits_typed_inactive_configuration",
        "normal_steering_cv_sidecar_nonzero",
        "normal_steering_cv_typed_nonzero",
        "normal_vds_chunk_minus_one",
        "normal_vds_chunk_exact",
        "normal_vds_chunk_plus_one",
        "normal_vds_chunk_two_plus_one",
        "normal_vds_complete_prefix_noop",
        "rerun_full_contract_pure_vds_off",
        "rerun_full_contract_pure_vds_on",
        "rerun_full_contract_sidecar_vds_off",
        "rerun_full_contract_sidecar_vds_on",
        "rerun_restart_absent_same_bootstrap_vds_off",
        "rerun_boundary_start0_strip0_limit1_vds_off",
        "rerun_boundary_start1_strip0_unlimited_no_velocity_vds_on",
        "rerun_optional_no_velocity_no_force_vds_on",
        "rerun_selected_stream_velocity_force_vds_on",
        "rerun_boundary_start0_strip1_beyond_selected_vds_on",
        "rerun_boundary_start0_strip0_exact_eof_box_vds_off",
        "rerun_boundary_start1_strip1_limit1_selected_no_velocity_vds_off",
        "failure_missing_trajectory_binding",
        "failure_invalid_output_chunk_size",
        "failure_invalid_output_vds_value",
        "failure_invalid_output_repair_policy",
        "failure_invalid_restart_policy",
        "failure_missing_topology_binding",
        "failure_missing_protocol_binding",
        "failure_mixed_legacy_h5_trajectory",
        "failure_mixed_legacy_h5_restart",
        "failure_sidecar_unsupported_key",
        "failure_sidecar_key_path_length_mismatch",
        "failure_sidecar_path_conflict",
        "failure_h5_topology_atom_count_mismatch",
        "failure_h5_topology_mass_shape",
        "failure_h5_topology_mass_dtype",
        "failure_h5_topology_schema_version",
        "failure_h5_nb14_dual_root",
        "failure_h5_nb14_extra_param_shape",
        "failure_h5_eam_unknown_format",
        "failure_h5_eam_embed_shape",
        "failure_h5_positional_restraint_dual_owner",
        "failure_h5_positional_restraint_weight_shape",
        "failure_h5_soft_wall_dual_owner",
        "failure_h5_soft_wall_count_shape",
        "failure_h5_soft_wall_name_shape",
        "failure_h5_soft_wall_potential_shape",
        "failure_h5_cv_restraint_dual_owner",
        "failure_h5_cv_restraint_partial_owner",
        "failure_h5_cv_restraint_offset_mismatch",
        "failure_h5_cv_restraint_definition_conflict",
        "failure_restart_dynamic_without_owner",
        "failure_restart_protocol_without_owner",
        "failure_restart_full_without_owner",
    } | {
        f"normal_output_family_{'_'.join(families)}_{legacy_mode}"
        for families in OUTPUT_FAMILY_COMBINATIONS
        for legacy_mode in ("suppressed", "coexist")
    } | {
        f"failure_output_{family}_{phase}_isolation"
        for family, phase in OUTPUT_WRITER_FAILURE_POINTS
    }
    assert summary["status_counts"]["supported"] > 0
    assert contracts["output.vds.cross_process_append_resume"].status == (
        "unsupported"
    )
    repair = contracts["output.vds.complete_prefix_repair"]
    assert repair.status == "supported"
    assert repair.case_ids == ("normal_vds_complete_prefix_noop",)
    assert repair.assertion_ids == ("h5_complete_prefix_repair_equivalence",)


def test_vds_resume_scope_boundary_is_explicit_and_non_executable():
    contracts = load_contract_registry()
    exclusions = required_scope_exclusions(contracts)
    vds_resume = contracts[VDS_CROSS_PROCESS_RESUME_CONTRACT]
    same_process_repair = contracts["output.vds.complete_prefix_repair"]

    assert VDS_CROSS_PROCESS_RESUME_CONTRACT in exclusions
    assert vds_resume.status == "unsupported"
    assert vds_resume.minimum_evidence == "E4"
    assert vds_resume.case_ids == ()
    assert vds_resume.assertion_ids == ()
    assert not set(vds_resume.case_ids).intersection(
        same_process_repair.case_ids
    )

    payload = {
        "scope_exclusions": list(exclusions),
        "scope_statement": SCOPED_EQUIVALENCE_STATEMENT,
    }
    assert validate_scope_boundary(contracts, payload) == exclusions


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove_exclusion", "scope_exclusions"),
        ("claim_full_equivalence", "unbounded equivalence"),
        ("silently_support_vds_resume", "must remain unsupported"),
    ],
)
def test_vds_resume_scope_boundary_rejects_claim_mutations(mutation, message):
    contracts = load_contract_registry()
    payload = {
        "scope_exclusions": list(required_scope_exclusions(contracts)),
        "scope_statement": SCOPED_EQUIVALENCE_STATEMENT,
    }
    if mutation == "remove_exclusion":
        payload["scope_exclusions"].remove(VDS_CROSS_PROCESS_RESUME_CONTRACT)
    elif mutation == "claim_full_equivalence":
        payload["scope_statement"] = "Full legacy/bundled equivalence."
    else:
        contracts = dict(contracts)
        contracts[VDS_CROSS_PROCESS_RESUME_CONTRACT] = replace(
            contracts[VDS_CROSS_PROCESS_RESUME_CONTRACT],
            status="supported",
        )

    with pytest.raises(AssertionError, match=message):
        validate_scope_boundary(contracts, payload)


def test_same_process_vds_repair_cannot_forge_resume_evidence():
    contracts = load_contract_registry()
    cases = {
        "normal_vds_complete_prefix_noop": {
            "records": [
                {
                    "case_id": "normal_vds_complete_prefix_noop",
                    "contract_id": VDS_CROSS_PROCESS_RESUME_CONTRACT,
                    "assertion_id": "h5_complete_prefix_repair_equivalence",
                    "evidence_level": "E4",
                    "status": "passed",
                    "details": {"method": "same_process_tail_repair"},
                }
            ]
        }
    }

    with pytest.raises(AssertionError, match="forged assertion"):
        validated_passed_contract_levels(contracts, cases)


def test_output_family_process_matrix_is_complete_and_mutation_guarded():
    cases = _cases_for_profile()
    audit = _audit_output_family_cases(cases)
    assert audit == {
        "case_count": 14,
        "combination_count": 7,
        "legacy_modes": ("suppressed", "coexist"),
    }

    output_cases = [case for case in cases if case.mode == "output_family"]
    assert {
        (case.output_families, case.legacy_output_mode) for case in output_cases
    } == {
        (families, legacy_mode)
        for families in OUTPUT_FAMILY_COMBINATIONS
        for legacy_mode in ("suppressed", "coexist")
    }
    contract = load_contract_registry()["output.family.combinations"]
    assert contract.minimum_evidence == "E3"
    assert set(contract.case_ids) == {case.name for case in output_cases}
    assert contract.assertion_ids == ("output_family_process_matrix",)

    removed = next(
        case
        for case in output_cases
        if case.output_families == ("restart",)
        and case.legacy_output_mode == "suppressed"
    )
    with pytest.raises(AssertionError, match="process matrix is incomplete"):
        _audit_output_family_cases(
            [case for case in cases if case is not removed]
        )


def test_output_writer_failure_matrix_covers_append_and_finalize():
    cases = [
        case for case in _cases_for_profile() if case.mode == "output_fault"
    ]
    assert {
        (case.output_fault_family, case.output_fault_phase) for case in cases
    } == set(OUTPUT_WRITER_FAILURE_POINTS)
    assert len(cases) == 6
    assert all(
        case.output_families == ("trajectory", "observable", "restart")
        and case.legacy_output_mode == "suppressed"
        and case.contract_ids == ("output.writer.failure_isolation",)
        and case.assertion_ids == ("writer_failure_isolation",)
        for case in cases
    )
    contract = load_contract_registry()["output.writer.failure_isolation"]
    assert contract.minimum_evidence == "F1"
    assert set(contract.case_ids) == {case.name for case in cases}
    assert contract.assertion_ids == ("writer_failure_isolation",)

    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    guarded_definition = (
        "if(SPONGE_BUILD_TESTS)\n"
        "  target_compile_definitions(SPONGE PRIVATE "
        "SPONGE_H5_TEST_FAULT_INJECTION=1)"
    )
    assert guarded_definition in cmake
    assert cmake.count("SPONGE_H5_TEST_FAULT_INJECTION") == 1


def test_structural_restart_uses_bundled_producer_e4_continuation():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_structural_restart_continuation"
    )
    contract = contracts["input.restart_load.structural"]

    assert "input.restart_load.structural" in case.contract_ids
    assert "restart_continuation_equivalence" in case.assertion_ids
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E4"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("restart_continuation_equivalence",)


def test_structural_restart_source_guard_rejects_legacy_h5_swap(tmp_path):
    legacy_dir = tmp_path / "legacy"
    bundled_dir = tmp_path / "bundled"
    legacy_dir.mkdir()
    bundled_dir.mkdir()
    run = AbRun(
        replica_index=0,
        replica_seed=1,
        legacy_dir=legacy_dir,
        bundled_dir=bundled_dir,
        legacy_metrics={},
        bundled_metrics={},
        legacy_output_contract={},
        bundled_output_contract={},
    )

    with pytest.raises(
        AssertionError,
        match="bundled structural restart producer source mismatch",
    ):
        _restart_continuation_source(run, "bundled", legacy_dir)


def test_nhc_dynamic_restart_uses_one_checkpoint_and_e4_continuation():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_nhc_dynamic_restart_continuation"
    )
    expected_contracts = {
        "input.restart_load.dynamic",
        "input.restart.dynamic.integrator_state",
        "input.restart.dynamic.nose_hoover_chain",
        "input.bias.nhc",
        "output.restart.dynamic_continuation",
    }

    assert case.mode == "dynamic_continuation"
    assert case.restart_load_policy == "dynamic"
    assert case.statistical_md is False
    assert set(case.contract_ids) == expected_contracts
    assert case.assertion_ids == ("restart_dynamic_continuation_equivalence",)
    for contract_id in expected_contracts:
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert contract.case_ids == (case.name,)
        assert contract.assertion_ids == case.assertion_ids
    assert contracts["input.restart_load.dynamic"].minimum_evidence == "E4"
    assert (
        contracts["output.restart.dynamic_continuation"].minimum_evidence
        == "E4"
    )


@pytest.mark.parametrize(
    ("case_name", "contract_id", "assertion_id"),
    (
        (
            "normal_middle_langevin_dynamic_restart_continuation",
            "input.restart.dynamic.middle_langevin_rng",
            "restart_middle_langevin_continuation_equivalence",
        ),
        (
            "normal_andersen_dynamic_restart_continuation",
            "input.restart.dynamic.andersen_rng",
            "restart_andersen_continuation_equivalence",
        ),
        (
            "normal_monte_carlo_barostat_dynamic_restart_continuation",
            "input.restart.dynamic.monte_carlo_barostat_rng",
            "restart_monte_carlo_barostat_continuation_equivalence",
        ),
    ),
)
def test_portable_stochastic_restart_has_independent_e4_continuation_case(
    case_name, contract_id, assertion_id
):
    contracts = load_contract_registry()
    case = next(case for case in _cases_for_profile() if case.name == case_name)
    contract = contracts[contract_id]

    assert case.mode == "portable_stochastic_continuation"
    assert case.restart_load_policy == "dynamic"
    assert case.statistical_md is False
    assert case.contract_ids == (contract.contract_id,)
    assert case.assertion_ids == (assertion_id,)
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E4"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == case.assertion_ids
    assert contract.reason == ""


def test_bussi_dynamic_restart_has_an_independent_e4_continuation_case():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_bussi_dynamic_restart_continuation"
    )
    contract = contracts["input.restart.dynamic.bussi_thermostat"]

    assert case.mode == "bussi_continuation"
    assert case.restart_load_policy == "dynamic"
    assert case.statistical_md is False
    assert case.contract_ids == (contract.contract_id,)
    assert case.assertion_ids == ("restart_bussi_continuation_equivalence",)
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E4"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == case.assertion_ids
    assert "typed portable Philox" in contract.bundled_surface
    assert "read-only compatibility" in contract.legacy_surface
    assert contract.reason == ""


def test_pressure_barostat_restart_has_an_independent_e4_continuation_case():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_pressure_barostat_dynamic_restart_continuation"
    )
    contract = contracts["input.restart.dynamic.pressure_based_barostat"]

    assert case.mode == "pressure_barostat_continuation"
    assert case.restart_load_policy == "dynamic"
    assert case.statistical_md is False
    assert case.contract_ids == (contract.contract_id,)
    assert case.assertion_ids == (
        "restart_pressure_barostat_continuation_equivalence",
    )
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E4"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == case.assertion_ids
    assert "typed portable Philox" in contract.bundled_surface
    assert "read-only compatibility" in contract.legacy_surface
    assert contract.reason == ""


def test_bonded_and_listed_contracts_have_independent_payload_cases():
    contracts = load_contract_registry()
    cases = {case.name: case for case in _cases_for_profile()}
    expected = {
        "input.topology.bond": "normal_bond_payload_sensitivity",
        "input.topology.angle": "normal_angle_payload_sensitivity",
        "input.topology.urey_bradley": (
            "normal_urey_bradley_payload_sensitivity"
        ),
        "input.topology.dihedral": "normal_dihedral_payload_sensitivity",
        "input.custom.listed": "normal_listed_payload_sensitivity",
    }
    for contract_id, case_name in expected.items():
        case = cases[case_name]
        contract = contracts[contract_id]
        assert case.mode == "normal"
        assert case.statistical_md is False
        assert case.normal_step_limit == 1
        assert case.normal_dt == 0.0
        assert contract_id in case.contract_ids
        assert case_name in contract.case_ids
        assert contract.minimum_evidence == "E3"
        assert "input_semantic_equivalence" in contract.assertion_ids


def test_remaining_function_contracts_have_independent_payload_cases():
    contracts = load_contract_registry()
    cases = {case.name: case for case in _cases_for_profile()}
    expected = {
        "input.protocol.positional_restraint.sidecar": (
            "normal_positional_restraint_sidecar_nonzero"
        ),
        "input.protocol.soft_wall.sidecar": "normal_soft_wall_sidecar_nonzero",
        "input.protocol.cv": "normal_steering_cv_typed_nonzero",
        "input.manybody.reaxff": "normal_reaxff_payload_sensitivity",
        "input.manybody.edip": "normal_edip_nonzero",
    }
    for contract_id, case_name in expected.items():
        case = cases[case_name]
        contract = contracts[contract_id]
        assert case.statistical_md is False
        assert contract_id in case.contract_ids
        assert case_name in contract.case_ids
        assert contract.minimum_evidence == "E3"
        assert "input_semantic_equivalence" in contract.assertion_ids

    functional_full_contract_only = {
        contract_id
        for contract_id, contract in contracts.items()
        if contract.status == "supported"
        and contract.direction == "input"
        and contract.component
        not in {"full_contract_fixture", "legacy_sidecar_bridge"}
        and contract.case_ids
        and all(
            case_id.startswith("rerun_full_contract_")
            for case_id in contract.case_ids
        )
    }
    assert functional_full_contract_only == set()


def test_input_only_contracts_have_evolution_or_reviewed_single_point_reason():
    contracts = load_contract_registry()
    cases = _cases_for_profile()
    audit = audit_input_evolution_contracts(contracts, cases)
    assert audit["audited_contract_count"] == 23
    assert len(audit["raw_audit"]["input_behavior_only"]) == 23
    assert len(audit["raw_audit"]["single_step_only"]) == 23
    assert len(audit["raw_audit"]["zero_dt_only"]) == 21
    assert set(audit["dynamic_contracts"]) == {
        "input.protocol.sits",
        "input.protocol.sits.nk_typed_restart",
        "input.topology.residue",
        "input.topology.residue.sidecar",
        "input.manybody.edip",
    }
    assert len(audit["single_point_justifications"]) == 18
    assert audit["unresolved_contracts"] == ()
    assert audit["cohorts"] == [
        "manybody_custom",
        "protocol_stateful",
        "topology_pbc",
    ]

    mutated = dict(contracts)
    contract_id = "input.topology.exclusions"
    mutated[contract_id] = replace(
        mutated[contract_id], single_point_justification=""
    )
    with pytest.raises(AssertionError, match="lack nonzero-dt evolution"):
        audit_input_evolution_contracts(mutated, cases)


def test_nonzero_dt_evolution_oracle_rejects_schedule_and_frozen_coordinates():
    payload = {
        "mdout": {
            "columns": ["potential"],
            "rows": [
                {"potential": 1.0},
                {"potential": 1.1},
                {"potential": 1.2},
            ],
        },
        "step": [1.0, 2.0, 3.0],
        "time": [0.0, 0.001, 0.002],
        "position": [
            0.0,
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.2,
            0.0,
            0.0,
        ],
        "velocity": [0.1, 0.0, 0.0] * 3,
        "force": [1.0, 0.0, 0.0] * 3,
        "box": [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0] * 3,
    }
    result = _assert_nonzero_dt_evolution(
        "mutation", payload, payload, dt=0.001
    )
    assert result["maximum_coordinate_delta"] == pytest.approx(0.2)

    bad_schedule = dict(payload, step=[1.0, 3.0, 4.0])
    with pytest.raises(AssertionError, match="schedule changed"):
        _assert_nonzero_dt_evolution(
            "mutation", bad_schedule, bad_schedule, dt=0.001
        )

    frozen = dict(payload, position=[0.0, 0.0, 0.0] * 3)
    with pytest.raises(AssertionError, match="coordinates are frozen"):
        _assert_nonzero_dt_evolution("mutation", frozen, frozen, dt=0.001)


def test_rerun_direct_payload_oracle_rejects_payload_and_schedule_mutations():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    kwargs = {
        "case_name": "mutation",
        "name": "velocity",
        "legacy_values": values,
        "h5_values": values,
        "h5_shape": (2, 1, 3),
        "h5_steps": [10.0, 20.0],
        "h5_times": [0.1, 0.2],
        "expected_steps": [10.0, 20.0],
        "expected_times": [0.1, 0.2],
        "relative_tolerance": 0.0,
        "absolute_tolerance": 0.0,
    }
    assert _assert_rerun_direct_payload(**kwargs)["frame_count"] == 2

    with pytest.raises(AssertionError, match="value count mismatch"):
        _assert_rerun_direct_payload(**dict(kwargs, legacy_values=values[:3]))
    with pytest.raises(AssertionError, match="step schedule"):
        _assert_rerun_direct_payload(**dict(kwargs, h5_steps=[11.0, 20.0]))
    with pytest.raises(AssertionError, match="direct legacy/H5 rerun velocity"):
        _assert_rerun_direct_payload(
            **dict(kwargs, h5_values=[4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
        )


def test_rerun_optional_presence_oracle_rejects_one_sided_fields():
    _assert_rerun_optional_presence(
        "mutation",
        "force",
        expected=False,
        h5_present=False,
        legacy_present=False,
    )
    with pytest.raises(AssertionError, match="optional field force presence"):
        _assert_rerun_optional_presence(
            "mutation",
            "force",
            expected=False,
            h5_present=True,
            legacy_present=False,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_normal_success_finite_scan_rejects_nonfinite_h5_values(
    tmp_path, value
):
    output = tmp_path / "output"
    output.mkdir()
    trajectory = output / "ab.spg.h5md"
    with h5py.File(trajectory, "w") as handle:
        handle.create_dataset(
            "/particles/all/position/value", data=[1.0, value]
        )

    with pytest.raises(AssertionError, match="contains non-finite value"):
        _scan_success_outputs(tmp_path)
    with pytest.raises(
        AssertionError, match="normal-success comparison contains matching"
    ):
        _assert_numeric_sequences_close(
            "matching mutation",
            [value],
            [value],
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )


def test_normal_success_finite_scan_records_traceable_counts(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    trajectory = output / "ab.spg.h5md"
    with h5py.File(trajectory, "w") as handle:
        handle.create_dataset("/particles/all/step", data=[1, 2])
        handle.create_dataset("/particles/all/time", data=[0.1, 0.2])

    scan = _scan_success_outputs(tmp_path)
    summary = _summarize_finite_scans(scan)
    assert scan["file_count"] == 1
    assert scan["dataset_count"] == 2
    assert scan["value_count"] == 4
    assert summary == {
        "classification": NORMAL_SUCCESS,
        "run_directory_count": 1,
        "file_count": 1,
        "dataset_count": 2,
        "value_count": 4,
        "run_directories": [str(tmp_path.resolve())],
    }
    _assert_finite_values("focused oracle", [0.0, 1.0, -1.0])


def test_case_output_classification_separates_functionality_and_failures():
    normal = next(
        case for case in _cases_for_profile() if case.failure_mutation is None
    )
    failure = next(
        case
        for case in _cases_for_profile()
        if case.failure_mutation is not None
    )
    propagation = replace(
        normal,
        mode=NONFINITE_PROPAGATION,
        failure_mutation=None,
    )
    assert _case_output_classification(normal) == NORMAL_SUCCESS
    assert _case_output_classification(failure) == FAILURE_SEMANTICS
    assert _case_output_classification(propagation) == NONFINITE_PROPAGATION

    finite_scan = {
        "classification": NORMAL_SUCCESS,
        "run_directory_count": 1,
        "file_count": 2,
        "dataset_count": 3,
        "value_count": 4,
        "run_directories": ["/tmp/mutation"],
    }
    assertion = AssertionEvidence("mutation", "E3", {"oracle": "finite"})
    enriched = _attach_finite_scan(normal, (assertion,), finite_scan)
    assert enriched[0].details["finite_scan"] == finite_scan
    assert enriched[0].details["output_classification"] == NORMAL_SUCCESS
    with pytest.raises(
        AssertionError, match="cannot attach normal-success finite evidence"
    ):
        _attach_finite_scan(propagation, (assertion,), finite_scan)
    with pytest.raises(
        AssertionError, match="optional field velocity presence"
    ):
        _assert_rerun_optional_presence(
            "mutation",
            "velocity",
            expected=True,
            h5_present=True,
            legacy_present=False,
        )


def test_meta_protocol_full_restart_uses_one_checkpoint_and_e4_continuation():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_meta_protocol_full_restart_continuation"
    )
    expected_contracts = {
        "input.restart_load.protocol",
        "input.restart_load.full",
        "input.bias.metadynamics",
    }

    assert case.mode == "protocol_full_continuation"
    assert case.restart_load_policy == "protocol/full"
    assert case.statistical_md is False
    assert set(case.contract_ids) == expected_contracts
    assert case.assertion_ids == (
        "restart_protocol_full_continuation_equivalence",
    )
    for contract_id in expected_contracts:
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert contract.case_ids == (case.name,)
        assert contract.assertion_ids == case.assertion_ids
    assert contracts["input.restart_load.protocol"].minimum_evidence == "E4"
    assert contracts["input.restart_load.full"].minimum_evidence == "E4"


def test_vds_chunk_boundary_cases_cover_required_frame_transitions():
    contract = load_contract_registry()["output.trajectory.chunk_size"]
    cases = {
        case.name: case
        for case in _cases_for_profile()
        if case.mode == "chunk_boundary"
    }

    assert {
        name: (
            case.output_chunk_size,
            case.expected_trajectory_frames,
            case.normal_interval,
            case.normal_dt,
        )
        for name, case in cases.items()
    } == {
        "normal_vds_chunk_minus_one": (4, 3, 1, 0.0001),
        "normal_vds_chunk_exact": (4, 4, 1, 0.0001),
        "normal_vds_chunk_plus_one": (4, 5, 1, 0.0001),
        "normal_vds_chunk_two_plus_one": (4, 9, 1, 0.0001),
        "normal_vds_complete_prefix_noop": (4, 5, 1, 0.0001),
    }
    assert all(case.vds for case in cases.values())
    assert all(not case.statistical_md for case in cases.values())
    assert all(
        "h5_chunk_boundary_equivalence" in case.assertion_ids
        for case in cases.values()
    )
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E3"
    assert set(contract.case_ids) == set(cases)
    assert contract.assertion_ids == ("h5_chunk_boundary_equivalence",)


def test_vds_complete_prefix_case_combines_production_noop_and_tail_repair():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_vds_complete_prefix_noop"
    )
    contract = contracts["output.vds.complete_prefix_repair"]

    assert case.mode == "chunk_boundary"
    assert case.vds is True
    assert case.statistical_md is False
    assert case.output_chunk_size == 4
    assert case.output_repair_policy == "complete_prefix"
    assert case.expected_trajectory_frames == 5
    assert case.contract_ids == (
        "output.trajectory",
        "output.trajectory.vds_on",
        "output.trajectory.chunk_size",
        "output.vds.complete_prefix_repair",
    )
    assert case.assertion_ids == (
        "h5_chunk_boundary_equivalence",
        "h5_complete_prefix_repair_equivalence",
    )
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E3"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("h5_complete_prefix_repair_equivalence",)

    source = VDS_TERMINAL_REPAIR_TEST.read_text(encoding="utf-8")
    for token in (
        "SelectiveFailHighFiveBackendFactory",
        "Finalize_With_Repair",
        "Test_Vds_Terminal_Tail_Is_Repaired_To_Complete_Prefix",
        'Require_Common_Wrapper_Metadata(file, "applied", 1)',
        "Test_Vds_Resume_Policy_Noops_When_Terminal_Shards_Are_Complete",
        'Require_Common_Wrapper_Metadata(file, "not_applied", 0)',
    ):
        assert token in source


@pytest.mark.parametrize(
    ("policy", "status", "repaired_count", "frame_count", "manifest", "match"),
    [
        ("strict", "not_applied", 0, 5, ("complete", "complete"), "policy"),
        (
            "complete_prefix",
            "applied",
            0,
            5,
            ("complete", "complete"),
            "status",
        ),
        (
            "complete_prefix",
            "not_applied",
            1,
            5,
            ("complete", "complete"),
            "repaired shard count",
        ),
        (
            "complete_prefix",
            "not_applied",
            0,
            4,
            ("complete", "complete"),
            "completion frame count",
        ),
        (
            "complete_prefix",
            "not_applied",
            0,
            5,
            ("complete", "open"),
            "manifest is not",
        ),
    ],
)
def test_vds_complete_prefix_noop_gate_rejects_metadata_mutations(
    tmp_path, policy, status, repaired_count, frame_count, manifest, match
):
    trajectory = tmp_path / "trajectory.spg.h5md"
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(trajectory, "w") as handle:
        root = "/parameters/sponge/output"
        handle.create_dataset(
            f"{root}/repair_policy", data=policy, dtype=string_type
        )
        handle.create_dataset(
            f"{root}/repair_status", data=status, dtype=string_type
        )
        handle.create_dataset(
            f"{root}/repaired_shard_count", data=[repaired_count]
        )
        handle.create_dataset(f"{root}/frame_count", data=[frame_count])
        handle.create_dataset(
            f"{root}/shard_manifest/status",
            data=list(manifest),
            dtype=string_type,
        )

    with pytest.raises(AssertionError, match=match):
        _assert_complete_prefix_noop_layout(
            "mutation",
            trajectory,
            expected_frame_count=5,
            expected_shard_count=2,
        )


def test_rerun_boundary_matrix_covers_semantic_axes_and_eof_boundaries():
    cases = [
        case
        for case in _cases_for_profile()
        if case.name.startswith("rerun_boundary_")
    ]

    assert {(case.rerun_start, case.rerun_strip) for case in cases} == {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    }
    assert {case.rerun_frame_limit for case in cases} == {1, 2, 3, None}
    assert {case.rerun_need_box_update for case in cases} == {False, True}
    assert {case.rerun_velocity_present for case in cases} == {False, True}
    assert {case.trajectory_particle_stream for case in cases} == {
        "all",
        "selected",
    }
    assert {case.vds for case in cases} == {False, True}
    assert all(
        "rerun_selection_equivalence" in case.assertion_ids for case in cases
    )


@pytest.mark.parametrize(
    ("case_name", "expected_indices"),
    [
        ("rerun_boundary_start0_strip0_limit1_vds_off", [0]),
        (
            "rerun_boundary_start1_strip0_unlimited_no_velocity_vds_on",
            [1],
        ),
        ("rerun_boundary_start0_strip1_beyond_selected_vds_on", [0]),
        ("rerun_boundary_start0_strip0_exact_eof_box_vds_off", [0, 1]),
        (
            "rerun_boundary_start1_strip1_limit1_selected_no_velocity_vds_off",
            [1],
        ),
    ],
)
def test_rerun_selection_oracle_is_independent_of_runtime_frame_counter(
    case_name, expected_indices
):
    case = next(case for case in _cases_for_profile() if case.name == case_name)

    assert (
        _expected_rerun_frame_indices(case, frame_count=2) == expected_indices
    )


def test_restart_absent_case_requires_same_bootstrap_and_behavior_outputs():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "rerun_restart_absent_same_bootstrap_vds_off"
    )
    contract = contracts["input.restart_load.absent"]

    assert case.restart_load_policy == "absent"
    assert case.statistical_md is False
    assert case.rerun_frame_limit == 2
    assert case.rerun_need_box_update is False
    assert "input.restart_load.structural" not in case.contract_ids
    assert {
        "input.restart_load.absent",
        "output.legacy.mdout",
        "output.trajectory",
        "output.observable",
        "output.trajectory.vds_off",
    } <= set(case.contract_ids)
    assert case.assertion_ids == (
        "mdout_deterministic_equivalence",
        "rerun_selection_equivalence",
        "h5_rerun_semantic_equivalence",
    )
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == case.assertion_ids


def test_failure_matrix_requires_exit_category_and_stable_tokens():
    cases = [
        case
        for case in _cases_for_profile()
        if case.failure_mutation is not None
    ]

    assert {case.failure_mutation for case in cases} == {
        "missing_trajectory",
        "invalid_chunk_size",
        "invalid_vds_value",
        "invalid_repair_policy",
        "invalid_restart_policy",
        "missing_topology",
        "missing_protocol",
        "mixed_trajectory",
        "mixed_restart",
        "unsupported_sidecar_key",
        "sidecar_length_mismatch",
        "sidecar_path_conflict",
        "h5_topology_atom_count_mismatch",
        "h5_topology_mass_shape",
        "h5_topology_mass_dtype",
        "h5_topology_schema_version",
        "h5_nb14_dual_root",
        "h5_nb14_extra_param_shape",
        "h5_eam_unknown_format",
        "h5_eam_embed_shape",
        "h5_positional_restraint_dual_owner",
        "h5_positional_restraint_weight_shape",
        "h5_soft_wall_dual_owner",
        "h5_soft_wall_count_shape",
        "h5_soft_wall_name_shape",
        "h5_soft_wall_potential_shape",
        "h5_cv_restraint_dual_owner",
        "h5_cv_restraint_partial_owner",
        "h5_cv_restraint_offset_mismatch",
        "h5_cv_restraint_definition_conflict",
        "restart_dynamic_without_owner",
        "restart_protocol_without_owner",
        "restart_full_without_owner",
    }
    assert all(case.expected_error_category for case in cases)
    assert all(case.expected_diagnostic_tokens for case in cases)
    assert all(
        case.assertion_ids == ("stable_failure_semantics",) for case in cases
    )
    assert {
        case.failure_mutation
        for case in cases
        if set(case.failure_branches) == {"legacy", "bundled"}
    } == {
        "missing_trajectory",
        "invalid_chunk_size",
        "invalid_vds_value",
        "invalid_repair_policy",
    }
    sidecar_cases = [
        case
        for case in cases
        if case.contract_ids == ("failure.sidecar_table",)
    ]
    assert {case.failure_mutation for case in sidecar_cases} == {
        "unsupported_sidecar_key",
        "sidecar_length_mismatch",
        "sidecar_path_conflict",
    }
    assert all(case.failure_branches == ("bundled",) for case in sidecar_cases)
    metadata_cases = [
        case for case in cases if case.contract_ids == ("failure.h5_metadata",)
    ]
    assert {
        case.failure_mutation: case.expected_error_category
        for case in metadata_cases
    } == {
        "h5_topology_atom_count_mismatch": "spongeErrorValueErrorCommand",
        "h5_topology_mass_shape": "spongeErrorBadFileFormat",
        "h5_topology_mass_dtype": "spongeErrorBadFileFormat",
        "h5_topology_schema_version": "spongeErrorValueErrorCommand",
        "h5_nb14_dual_root": "spongeErrorBadFileFormat",
        "h5_nb14_extra_param_shape": "spongeErrorBadFileFormat",
        "h5_eam_unknown_format": "spongeErrorBadFileFormat",
        "h5_eam_embed_shape": "spongeErrorBadFileFormat",
        "h5_positional_restraint_dual_owner": "spongeErrorConflictingCommand",
        "h5_positional_restraint_weight_shape": "spongeErrorBadFileFormat",
        "h5_soft_wall_dual_owner": "spongeErrorConflictingCommand",
        "h5_soft_wall_count_shape": "spongeErrorBadFileFormat",
        "h5_soft_wall_name_shape": "spongeErrorBadFileFormat",
        "h5_soft_wall_potential_shape": "spongeErrorBadFileFormat",
        "h5_cv_restraint_dual_owner": "spongeErrorConflictingCommand",
        "h5_cv_restraint_partial_owner": "spongeErrorBadFileFormat",
        "h5_cv_restraint_offset_mismatch": "spongeErrorBadFileFormat",
        "h5_cv_restraint_definition_conflict": "spongeErrorBadFileFormat",
    }
    assert all(case.failure_branches == ("bundled",) for case in metadata_cases)
    contracts = load_contract_registry()
    metadata_contract = contracts["failure.h5_metadata"]
    assert metadata_contract.status == "supported"
    assert set(metadata_contract.case_ids) == {
        case.name for case in metadata_cases
    }
    assert metadata_contract.assertion_ids == ("stable_failure_semantics",)
    assert metadata_contract.reason == ""
    assert "failure.h5_metadata.runtime_rejections" not in contracts
    assert SUPPORTED_TOPOLOGY_SCHEMA_VERSIONS == (
        "0",
        "1",
        "xponge.legacy_to_bundle.v1",
    )
    restart_owner_cases = [
        case
        for case in cases
        if case.contract_ids == ("failure.restart_owner_state",)
    ]
    assert {
        case.failure_mutation: case.restart_load_policy
        for case in restart_owner_cases
    } == {
        "restart_dynamic_without_owner": "dynamic",
        "restart_protocol_without_owner": "protocol",
        "restart_full_without_owner": "full",
    }
    assert all(
        case.failure_branches == ("bundled",) for case in restart_owner_cases
    )


def test_h5_bundle_runner_defaults_to_current_pixi_build_dir():
    source = H5_BUNDLE_RUNNER.read_text(encoding="utf-8")

    assert "SPONGE_H5_BUNDLE_TEST_BUILD_DIR" in source
    assert 'elif [[ -n "${PIXI_ENVIRONMENT_NAME:-}" ]]' in source
    assert 'BUILD_DIR="${ROOT_DIR}/build-${PIXI_ENVIRONMENT_NAME}"' in source
    assert 'BUILD_DIR="${ROOT_DIR}/build-h5-tests"' in source


def test_output_behavior_cases_cover_particle_fields_and_both_vds_modes():
    contracts = load_contract_registry()
    cases = {case.name: case for case in _cases_for_profile()}

    assert cases["normal_core_h5_output"].vds is False
    assert cases["normal_sits_ff19sb_cmap_peptide"].vds is True
    for case_name in (
        "normal_core_h5_output",
        "normal_sits_ff19sb_cmap_peptide",
    ):
        case = cases[case_name]
        assert {
            "output.legacy.crd",
            "output.legacy.box",
            "output.legacy.velocity",
            "output.legacy.force",
        } <= set(case.contract_ids)
        assert "particle_legacy_coexistence" in case.assertion_ids

    for contract_id in (
        "output.legacy.crd",
        "output.legacy.box",
        "output.legacy.velocity",
        "output.legacy.force",
    ):
        assert contracts[contract_id].status == "supported"
        assert contracts[contract_id].minimum_evidence == "E3"


def test_mdinfo_parser_uses_declared_structured_keys(tmp_path):
    mdinfo = tmp_path / "mdinfo.txt"
    mdinfo.write_text(
        "Working Directory: /dynamic/path\n"
        + "\n".join(f"{key} is value for {key}" for key in MDINFO_CONTRACT_KEYS)
        + "\nStart Wall Time: dynamic\n",
        encoding="utf-8",
    )

    parsed = _parse_mdinfo_key_values(mdinfo)

    assert set(parsed) == MDINFO_CONTRACT_KEYS
    assert all(len(values) == 1 for values in parsed.values())


def test_vds_chunk_boundary_gate_enumerates_all_required_edges():
    source = VDS_WRITER_TEST.read_text(encoding="utf-8")

    assert "Test_Vds_Chunk_Boundary_Frame_Counts" in source
    for boundary in ("{1, {1}}", "{2, {2}}", "{3, {2, 1}}", "{5, {2, 2, 1}}"):
        assert boundary in source


def test_output_behavior_closure_keeps_family_module_restart_and_repair_gates():
    required_functions = {
        OUTPUT_PLAN_TEST: {
            "Test_H5_Output_Path_Keys_Enable_Only_Their_Bundle",
            "Test_All_H5_Output_Bundles_Can_Be_Enabled_Together",
            "Test_Legacy_Output_Plan_All_Keys",
            "Test_Resolve_Legacy_Output_Plan_Matrix",
        },
        WRITER_TEST: {
            "Test_Trajectory_Optional_Velocity_And_Force_Paths",
            "Test_Observable_Only_Writer",
            "Test_Restart_Module_State_And_Legacy_Provenance",
            "Test_Legacy_Provenance_On_Trajectory_And_Observable",
        },
        MODULE_TEST: {
            "Test_Nhc_And_Sits_Mappings",
            "Test_Metadynamics_And_Diagnostics",
            "Test_Reaxff_Mappings",
        },
        HIGHFIVE_TEST: {
            "Test_Trajectory_Optional_Particle_Fields_With_Real_Backend",
            "Test_Restart_Writer_With_Real_Backend",
            "Test_Module_Metad_And_Reaxff_With_Real_Backend",
            "Test_Vds_Complete_Prefix_Repair_With_Real_Backend",
        },
    }
    for path, functions in required_functions.items():
        source = path.read_text(encoding="utf-8")
        for function in functions:
            assert function in source, (
                f"{path.name} no longer covers {function}"
            )


def test_input_semantic_gate_requires_nontrivial_owned_result():
    result = assert_module_semantics(
        "bond",
        [{"bond": 2.0}, {"bond": 3.0}],
        [{"bond": 2.0}, {"bond": 3.0}],
        InputSemanticSpec("input.topology.bond", ("bond",)),
        deterministic=True,
    )

    assert result["legacy_nontrivial"] is True
    assert result["bundled_nontrivial"] is True


def test_focused_edip_case_requires_pure_h5_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_edip_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_EDIP_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.manybody.edip",
    )
    assert spec == (
        InputSemanticSpec("input.manybody.edip", ("EDIP",), 1.0e-6),
    )
    assert contracts["input.manybody.edip"].status == "supported"
    assert contracts["input.manybody.edip"].case_ids == (case.name,)
    assert contracts["input.manybody.edip"].assertion_ids == (
        "input_semantic_equivalence",
    )


def test_focused_sw_case_requires_sidecar_pair_and_three_body_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_sw_sidecar_pair_three_body"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_SW_SIDECAR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.manybody.sw.sidecar",
    )
    assert spec == (
        InputSemanticSpec("input.manybody.sw.sidecar", ("SW",), 1.0e-6),
    )
    sidecar = contracts["input.manybody.sw.sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (case.name,)
    assert sidecar.assertion_ids == ("input_semantic_equivalence",)


def test_focused_typed_sw_case_requires_pair_three_body_payload_response():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_sw_typed_pair_three_body"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_SW_TYPED_FIXTURE
    assert case.fixture_case == "focused_sw_typed_three_atom"
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.manybody.sw",
    )
    assert spec == (InputSemanticSpec("input.manybody.sw", ("SW",), 1.0e-6),)
    native = contracts["input.manybody.sw"]
    assert native.status == "supported"
    assert native.case_ids == (case.name,)
    assert native.assertion_ids == ("input_semantic_equivalence",)


def test_focused_sw_gate_rejects_pair_only_energy_and_force_mutations():
    full_force = (
        -352.62115,
        -352.62115,
        0.0,
        404.2786,
        -51.657455,
        0.0,
        -51.657455,
        404.2786,
        0.0,
    )
    pair_only_force = (
        -340.4832,
        -340.4832,
        0.0,
        343.52106,
        -3.037885,
        0.0,
        -3.037885,
        343.52106,
        0.0,
    )
    result = _assert_sw_pair_three_body_oracle(
        "SW", [{"SW": 194.50}], full_force
    )
    assert result["three_body_energy_contribution"] == pytest.approx(35.71)
    assert result["maximum_force_delta_from_pair_only"] > 60.0

    with pytest.raises(AssertionError, match="pair\\+three-body SW energy"):
        _assert_sw_pair_three_body_oracle(
            "SW lambda=0", [{"SW": 158.79}], pair_only_force
        )
    with pytest.raises(AssertionError, match="pair\\+three-body SW force"):
        _assert_sw_pair_three_body_oracle(
            "SW pair-only force", [{"SW": 194.50}], pair_only_force
        )


def test_focused_tersoff_case_requires_isolated_angular_sidecar_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_tersoff_sidecar_angular"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_TERSOFF_SIDECAR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.manybody.tersoff.sidecar",
    )
    assert spec == (
        InputSemanticSpec(
            "input.manybody.tersoff.sidecar", ("potential",), 1.0e-6
        ),
    )
    sidecar = contracts["input.manybody.tersoff.sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (case.name,)
    assert sidecar.assertion_ids == ("input_semantic_equivalence",)


def test_focused_typed_tersoff_case_requires_angular_payload_response():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_tersoff_typed_angular"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_TERSOFF_TYPED_FIXTURE
    assert case.fixture_case == "focused_tersoff_typed_three_atom"
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.manybody.tersoff",
    )
    assert spec == (
        InputSemanticSpec("input.manybody.tersoff", ("potential",), 1.0e-6),
    )
    native = contracts["input.manybody.tersoff"]
    assert native.status == "supported"
    assert native.case_ids == (case.name,)
    assert native.assertion_ids == ("input_semantic_equivalence",)


def test_focused_tersoff_gate_rejects_gamma_zero_mutations():
    rows = [
        {
            "potential": -173.23,
            "eff_pot": -173.23468,
            "PM": 0.0,
            "temperature": 0.0,
        }
    ]
    full_force = (
        135.94907,
        135.94907,
        0.0,
        -119.686844,
        -16.262218,
        0.0,
        -16.262218,
        -119.686844,
        0.0,
    )
    gamma_zero_force = (
        144.78313,
        144.78313,
        0.0,
        -144.78313,
        0.0,
        0.0,
        0.0,
        -144.78313,
        0.0,
    )
    result = _assert_tersoff_angular_oracle("Tersoff", rows, full_force)
    assert result["angular_energy_contribution"] == pytest.approx(22.83)
    assert result["maximum_force_delta_from_gamma_zero"] > 25.0

    gamma_zero_rows = [
        {
            "potential": -196.06,
            "eff_pot": -196.05984,
            "PM": 0.0,
            "temperature": 0.0,
        }
    ]
    with pytest.raises(AssertionError, match="angular Tersoff potential"):
        _assert_tersoff_angular_oracle(
            "Tersoff gamma=0", gamma_zero_rows, gamma_zero_force
        )
    with pytest.raises(AssertionError, match="angular Tersoff force"):
        _assert_tersoff_angular_oracle(
            "Tersoff gamma=0 force", rows, gamma_zero_force
        )


def test_focused_custom_pair_case_requires_pure_h5_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_custom_pair_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_CUSTOM_PAIR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.custom.pairwise",
    )
    assert spec == (
        InputSemanticSpec("input.custom.pairwise", ("custom_pair",), 1.0e-6),
    )
    assert contracts["input.custom.pairwise"].status == "supported"
    assert contracts["input.custom.pairwise"].case_ids == (case.name,)
    assert contracts["input.custom.pairwise"].assertion_ids == (
        "input_semantic_equivalence",
    )


def test_focused_exclusions_case_requires_native_payload_and_coulomb_oracle():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_exclusions_coulomb_oracle"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_EXCLUSIONS_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.exclusions",
    )
    assert spec == (
        InputSemanticSpec("input.topology.exclusions", ("Coulomb",), 1.0e-6),
    )
    assert contracts["input.topology.exclusions"].status == "supported"
    assert contracts["input.topology.exclusions"].case_ids == (case.name,)
    assert contracts["input.topology.exclusions"].assertion_ids == (
        "input_semantic_equivalence",
    )


def test_focused_residue_case_retains_runtime_partition_and_pbc_mapping():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_residue_sidecar_pbc_mapping"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_RESIDUE_SIDECAR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.residue.sidecar",
    )
    assert spec == (
        InputSemanticSpec("input.topology.residue.sidecar", ("bond",), 1.0e-6),
    )
    sidecar = contracts["input.topology.residue.sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (
        "normal_residue_sidecar_pbc_mapping",
        "normal_residue_sidecar_com_res_virial",
    )
    assert sidecar.assertion_ids == ("input_semantic_equivalence",)


def test_focused_residue_case_requires_com_res_membership_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_residue_sidecar_com_res_virial"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_RESIDUE_COM_RES_FIXTURE
    assert case.fixture_case == "focused_residue_sidecar_com_res_four_atom"
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.001
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.residue.sidecar",
    )
    assert spec == (
        InputSemanticSpec(
            "input.topology.residue.sidecar",
            ("bond", "restrain", "pressure", "Pxx"),
            1.0e-6,
        ),
    )
    sidecar = contracts["input.topology.residue.sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (
        "normal_residue_sidecar_pbc_mapping",
        "normal_residue_sidecar_com_res_virial",
    )
    assert sidecar.assertion_ids == ("input_semantic_equivalence",)


def test_focused_typed_residue_cases_require_both_runtime_consumers():
    contracts = load_contract_registry()
    pbc_case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_residue_typed_pbc_mapping"
    )
    com_res_case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_residue_typed_com_res_virial"
    )
    pbc_spec = INPUT_SEMANTIC_SPECS_BY_CASE[pbc_case.name]
    com_res_spec = INPUT_SEMANTIC_SPECS_BY_CASE[com_res_case.name]

    assert pbc_case.fixture_case == FOCUSED_RESIDUE_TYPED_PBC_FIXTURE
    assert com_res_case.fixture_case == FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE
    assert pbc_case.statistical_md is False
    assert com_res_case.statistical_md is False
    assert pbc_case.normal_step_limit == 1
    assert com_res_case.normal_step_limit == 1
    assert pbc_case.normal_dt == 0.0
    assert com_res_case.normal_dt == 0.001
    assert pbc_case.input_behavior_only is True
    assert com_res_case.input_behavior_only is True
    expected_contracts = (
        "output.legacy.mdout",
        "input.topology.residue",
    )
    assert pbc_case.contract_ids == expected_contracts
    assert com_res_case.contract_ids == expected_contracts
    assert pbc_spec == (
        InputSemanticSpec("input.topology.residue", ("bond",), 1.0e-6),
    )
    assert com_res_spec == (
        InputSemanticSpec(
            "input.topology.residue",
            ("bond", "restrain", "pressure", "Pxx"),
            1.0e-6,
        ),
    )
    typed = contracts["input.topology.residue"]
    assert typed.status == "supported"
    assert typed.case_ids == (pbc_case.name, com_res_case.name)
    assert typed.assertion_ids == ("input_semantic_equivalence",)


def test_focused_core_topology_case_requires_payload_sensitive_consumers():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_core_topology_payload_sensitivity"
    )

    assert case.fixture_case == FOCUSED_CORE_TOPOLOGY_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.mass",
        "input.topology.charge",
        "input.topology.lj",
    )
    assert INPUT_SEMANTIC_SPECS_BY_CASE[case.name] == (
        InputSemanticSpec("input.topology.mass", ("temperature",), 1.0e-6),
        InputSemanticSpec("input.topology.charge", ("Coulomb",), 1.0e-6),
        InputSemanticSpec("input.topology.lj", ("LJ",), 1.0e-6),
    )
    for contract_id in case.contract_ids[1:]:
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert case.name in contract.case_ids
        assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_nb14_scaled_and_extra_surfaces_have_independent_behavior_cases():
    contracts = load_contract_registry()
    cases = {case.name: case for case in _cases_for_profile()}
    expected = {
        "normal_nb14_scaled_nonzero": (
            FOCUSED_NB14_SCALED_FIXTURE,
            "input.topology.nb14",
        ),
        "normal_nb14_extra_nonzero": (
            FOCUSED_NB14_EXTRA_FIXTURE,
            "input.topology.nb14_extra",
        ),
    }
    for case_name, (fixture, contract_id) in expected.items():
        case = cases[case_name]
        assert case.fixture_case == fixture
        assert case.statistical_md is False
        assert case.normal_step_limit == 1
        assert case.normal_dt == 0.0
        assert case.input_behavior_only is True
        assert case.contract_ids == ("output.legacy.mdout", contract_id)
        assert INPUT_SEMANTIC_SPECS_BY_CASE[case_name] == (
            InputSemanticSpec(contract_id, ("nb14_LJ", "nb14_EE"), 1.0e-6),
        )
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert case_name in contract.case_ids
        assert contract.assertion_ids == ("input_semantic_equivalence",)

    scaled = contracts["input.topology.nb14"]
    extra = contracts["input.topology.nb14_extra"]
    assert scaled.legacy_surface == "nb14_in_file"
    assert scaled.bundled_surface == "/forcefield/nb14"
    assert scaled.inventory_refs == ("topology_sidecar_keys:nb14_in_file",)
    assert extra.legacy_surface == "nb14_extra_in_file"
    assert extra.bundled_surface == "/forcefield/nb14_extra"
    assert extra.inventory_refs == ("topology_sidecar_keys:nb14_extra_in_file",)
    assert any(
        spec.contract_id == "input.topology.nb14_extra"
        for spec in RERUN_INPUT_SEMANTIC_SPECS
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"correct_value": math.nan}, "has no finite temperature"),
        ({"correct_value": 0.0}, "correct temperature is trivial"),
        ({"control_value": 26.205}, "payload did not change temperature"),
        ({"control_force": (0.0,) * 3}, "force control has wrong shape"),
        (
            {"control_force": (math.nan, 0.0, 0.0, 0.0, 0.0, 0.0)},
            "force control is non-finite",
        ),
        (
            {"force_must_change": True},
            "payload did not change force",
        ),
        (
            {"control_force": (1.0, 0.0, 0.0, -1.0, 0.0, 0.0)},
            "mass-only control unexpectedly changed force",
        ),
    ],
)
def test_core_topology_payload_oracle_rejects_insensitive_or_invalid_response(
    kwargs, message
):
    arguments = {
        "label": "core topology",
        "observable": "temperature",
        "correct_value": 26.21,
        "control_value": 52.42,
        "minimum_observable_delta": 1.0e-2,
        "correct_force": (0.0,) * 6,
        "control_force": (0.0,) * 6,
        "force_must_change": False,
    }
    arguments.update(kwargs)
    with pytest.raises(AssertionError, match=message):
        _assert_core_topology_payload_response(**arguments)


def test_focused_gb_case_requires_native_state_and_sidecar_activation_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_gb_hybrid_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_GB_HYBRID_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.gb.hybrid_activation",
    )
    assert spec == (
        InputSemanticSpec(
            "input.topology.gb.hybrid_activation", ("gb",), 1.0e-6
        ),
    )
    hybrid = contracts["input.topology.gb.hybrid_activation"]
    assert hybrid.status == "supported"
    assert hybrid.case_ids == (case.name,)
    assert hybrid.assertion_ids == ("input_semantic_equivalence",)


def test_focused_gb_native_case_requires_pure_typed_nonzero_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_gb_native_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_GB_NATIVE_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.gb",
    )
    assert spec == (InputSemanticSpec("input.topology.gb", ("gb",), 1.0e-6),)
    native = contracts["input.topology.gb"]
    assert native.status == "supported"
    assert native.case_ids == (case.name,)
    assert native.assertion_ids == ("input_semantic_equivalence",)
    assert native.reason == ""


def test_focused_gb_gate_rejects_activation_only_or_coulomb_only_behavior():
    spec = InputSemanticSpec(
        "input.topology.gb.hybrid_activation", ("gb",), 1.0e-6
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "GB",
            [{"gb": 0.0}],
            [{"gb": 0.0}],
            spec,
            deterministic=True,
        )
    with pytest.raises(AssertionError, match=r"GB\+Coulomb force oracle"):
        _assert_gb_force_oracle(
            "GB mutation",
            [{"Coulomb": -0.50, "gb": -0.25, "potential": -0.75}],
            (0.25, 0.0, 0.0, -0.25, 0.0, 0.0),
        )
    with pytest.raises(AssertionError, match="gb oracle mismatch"):
        _assert_gb_force_oracle(
            "GB mutation",
            [{"Coulomb": -0.50, "gb": 0.0, "potential": -0.50}],
            (0.10313021, 0.0, 0.0, -0.10313021, 0.0, 0.0),
        )


def test_focused_improper_cases_cover_native_and_unmodified_conversion_routes():
    contracts = load_contract_registry()
    cases = {case.name: case for case in _cases_for_profile()}
    native_case = cases["normal_improper_native_nonzero"]
    native_spec = INPUT_SEMANTIC_SPECS_BY_CASE[native_case.name]

    assert native_case.fixture_case == FOCUSED_IMPROPER_NATIVE_FIXTURE
    assert native_case.statistical_md is False
    assert native_case.normal_step_limit == 1
    assert native_case.normal_dt == 0.0
    assert native_case.input_behavior_only is True
    assert native_case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.improper.native_runtime",
    )
    assert native_spec == (
        InputSemanticSpec(
            "input.topology.improper.native_runtime",
            ("improper_dihedral",),
            1.0e-6,
        ),
    )
    native = contracts["input.topology.improper.native_runtime"]
    assert native.status == "supported"
    assert native.case_ids == (native_case.name,)
    assert native.assertion_ids == ("input_semantic_equivalence",)

    converted_names = (
        "normal_improper_converted_canonical_nonzero",
        "normal_improper_converted_alias_nonzero",
    )
    converted_fixtures = (
        FOCUSED_IMPROPER_CONVERTED_CANONICAL_FIXTURE,
        FOCUSED_IMPROPER_CONVERTED_ALIAS_FIXTURE,
    )
    for case_name, fixture in zip(
        converted_names, converted_fixtures, strict=True
    ):
        case = cases[case_name]
        assert case.fixture_case == fixture
        assert case.statistical_md is False
        assert case.normal_step_limit == 1
        assert case.normal_dt == 0.0
        assert case.input_behavior_only is True
        assert case.contract_ids == (
            "output.legacy.mdout",
            "input.topology.improper",
        )
        assert INPUT_SEMANTIC_SPECS_BY_CASE[case.name] == (
            InputSemanticSpec(
                "input.topology.improper",
                ("improper_dihedral",),
                1.0e-6,
            ),
        )

    conversion = contracts["input.topology.improper"]
    assert conversion.status == "supported"
    assert conversion.case_ids == converted_names
    assert conversion.assertion_ids == ("input_semantic_equivalence",)
    assert conversion.reason == ""


def test_focused_improper_gate_rejects_activation_only_evidence():
    spec = InputSemanticSpec(
        "input.topology.improper.native_runtime",
        ("improper_dihedral",),
        1.0e-6,
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "improper",
            [{"improper_dihedral": 0.0}],
            [{"improper_dihedral": 0.0}],
            spec,
            deterministic=True,
        )
    with pytest.raises(AssertionError, match="improper energy changed"):
        _assert_focused_improper_oracle(
            "improper mutation",
            [{"improper_dihedral": 0.0}],
            [1.0] * 12,
        )
    with pytest.raises(AssertionError, match="maximum force changed"):
        _assert_focused_improper_oracle(
            "improper mutation",
            [{"improper_dihedral": 31.36}],
            [1.0] * 12,
        )
    with pytest.raises(AssertionError, match="bundled force is all trivial"):
        _assert_nontrivial_equivalent_forces(
            "improper force", [1.0, -1.0], [0.0, 0.0]
        )


def test_focused_lj_soft_core_case_requires_pure_h5_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_lj_soft_core_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_LJ_SOFT_CORE_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.lj_soft_core",
    )
    assert spec == (
        InputSemanticSpec("input.topology.lj_soft_core", ("LJ_soft",), 1.0e-6),
    )
    contract = contracts["input.topology.lj_soft_core"]
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_focused_lj_soft_core_gate_rejects_trivial_energy_and_force():
    spec = InputSemanticSpec(
        "input.topology.lj_soft_core", ("LJ_soft",), 1.0e-6
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "LJ soft-core",
            [{"LJ_soft": 0.0}],
            [{"LJ_soft": 0.0}],
            spec,
            deterministic=True,
        )
    with pytest.raises(AssertionError, match="bundled force is all trivial"):
        _assert_nontrivial_equivalent_forces(
            "LJ soft-core force", [1.0, -1.0], [0.0, 0.0]
        )


def test_focused_subsystem_division_case_requires_partition_sensitive_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_subsystem_division_partition"
    )
    assert case.fixture_case == FOCUSED_SUBSYSTEM_DIVISION_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.subsystem_division",
    )
    assert INPUT_SEMANTIC_SPECS_BY_CASE[case.name] == (
        InputSemanticSpec(
            "input.topology.subsystem_division",
            ("LJ_soft_inter", "LJ_soft_intra"),
            1.0e-6,
        ),
    )
    contract = contracts["input.topology.subsystem_division"]
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_focused_subsystem_division_gate_rejects_weak_partition_oracles():
    valid = {
        "baseline_inter": -0.06,
        "baseline_intra": 0.0,
        "baseline_total": -0.06,
        "control_inter": 0.0,
        "control_intra": -0.06,
        "control_total": -0.06,
    }
    response = _assert_subsystem_partition_response("subsystem", valid)
    assert response == {
        "inter_response": 0.06,
        "intra_response": 0.06,
        "total_delta": 0.0,
    }

    mutations = (
        ("non-finite", {"baseline_inter": math.nan}, "non-finite"),
        ("trivial inter", {"baseline_inter": 0.0}, "energy is trivial"),
        ("baseline mixed", {"baseline_intra": -0.02}, "not empty"),
        ("trivial intra", {"control_intra": 0.0}, "energy is trivial"),
        ("control mixed", {"control_inter": -0.02}, "retained inter"),
        (
            "baseline nonconserving",
            {"baseline_total": -0.10},
            "does not conserve",
        ),
        (
            "control nonconserving",
            {"control_total": -0.10},
            "does not conserve",
        ),
        (
            "total changed",
            {"control_intra": -0.08, "control_total": -0.08},
            "mask changed total",
        ),
    )
    for label, changes, message in mutations:
        with pytest.raises(AssertionError, match=message):
            _assert_subsystem_partition_response(label, {**valid, **changes})


@pytest.mark.parametrize(
    ("case_name", "fixture", "contract_id", "observable"),
    [
        (
            "normal_virtual_atoms_all_types",
            FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
            "input.topology.virtual_atoms",
            "PM",
        ),
        (
            "normal_virtual_atoms_pbc_boundary",
            FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
            "input.topology.virtual_atoms_pbc",
            "PM",
        ),
        (
            "normal_virtual_atoms_plural_alias",
            FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
            "input.topology.virtual_atoms_alias",
            "PM",
        ),
    ],
)
def test_focused_virtual_atom_cases_require_native_runtime_oracles(
    case_name: str, fixture: str, contract_id: str, observable: str
):
    contracts = load_contract_registry()
    case = next(case for case in _cases_for_profile() if case.name == case_name)

    assert case.fixture_case == fixture
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == ("output.legacy.mdout", contract_id)
    assert INPUT_SEMANTIC_SPECS_BY_CASE[case.name] == (
        InputSemanticSpec(contract_id, (observable,), 1.0e-6),
    )
    contract = contracts[contract_id]
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_focused_virtual_atom_gate_rejects_semantic_mutations():
    expected_coordinates = (
        9.5,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        9.75,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
    )
    redistributed_forces = (
        0.25,
        0.0,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    )
    wrong_pbc_coordinates = list(expected_coordinates)
    wrong_pbc_coordinates[6] = 7.25
    with pytest.raises(AssertionError, match="coordinate oracle"):
        _assert_virtual_atom_oracle(
            "PBC mutation",
            expected_coordinates,
            (2,),
            wrong_pbc_coordinates,
            redistributed_forces,
        )

    with pytest.raises(AssertionError, match="real-atom force is all trivial"):
        _assert_virtual_atom_oracle(
            "force redistribution mutation",
            expected_coordinates,
            (2,),
            expected_coordinates,
            (0.0,) * len(expected_coordinates),
        )


def test_focused_constraint_sidecar_case_requires_projected_runtime_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_constraint_sidecar_projection"
    )

    assert case.fixture_case == FOCUSED_CONSTRAINT_SIDECAR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 4
    assert case.normal_interval == 1
    assert case.normal_dt == 0.001
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.constraint.sidecar",
    )
    assert case.assertion_ids == (
        "mdout_deterministic_equivalence",
        "constraint_geometry_equivalence",
    )

    sidecar = contracts["input.protocol.constraint.sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (case.name,)
    assert sidecar.assertion_ids == ("constraint_geometry_equivalence",)


def test_focused_typed_constraint_case_requires_payload_projection():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_constraint_typed_projection"
    )

    assert case.fixture_case == FOCUSED_CONSTRAINT_TYPED_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 4
    assert case.normal_interval == 1
    assert case.normal_dt == 0.001
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.constraint",
    )
    assert case.assertion_ids == (
        "mdout_deterministic_equivalence",
        "constraint_geometry_equivalence",
    )
    typed = contracts["input.protocol.constraint"]
    assert typed.status == "supported"
    assert typed.case_ids == (case.name,)
    assert typed.assertion_ids == ("constraint_geometry_equivalence",)


@pytest.mark.parametrize(
    ("positions", "velocities", "message"),
    [
        (
            (0.0, 0.0, 0.0, 1.6, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "distance residual exceeds tolerance",
        ),
        (
            (0.0, 0.0, 0.0, 1.5, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            "radial velocity residual exceeds tolerance",
        ),
    ],
)
def test_focused_constraint_gate_rejects_ignored_or_wrong_constraint(
    positions, velocities, message
):
    with pytest.raises(AssertionError, match=message):
        _assert_constraint_projection_oracle(
            "constraint mutation", positions, velocities
        )


def test_focused_sits_case_requires_typed_nk_bias_and_scaled_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_sits_nk_typed_restart_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_SITS_NK_TYPED_RESTART_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.restart_load_policy == "protocol"
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.sits.nk_typed_restart",
    )
    assert spec == (
        InputSemanticSpec(
            "input.protocol.sits.nk_typed_restart",
            ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
            1.0e-4,
        ),
    )
    typed_restart = contracts["input.protocol.sits.nk_typed_restart"]
    assert typed_restart.status == "supported"
    assert typed_restart.component == "restart_protocol_state"
    assert typed_restart.case_ids == (
        case.name,
        "normal_sits_typed_configuration_nonzero",
    )
    assert typed_restart.assertion_ids == ("input_semantic_equivalence",)
    typed_config = contracts["input.protocol.sits"]
    assert typed_config.status == "supported"
    assert typed_config.case_ids == (
        "normal_sits_typed_configuration_nonzero",
        "normal_sits_typed_inactive_configuration",
    )


def test_focused_typed_sits_case_requires_config_atoms_and_restart_state():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_sits_typed_configuration_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_SITS_TYPED_CONFIG_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.restart_load_policy == "protocol"
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.sits",
        "input.protocol.sits.nk_typed_restart",
    )
    assert spec == (
        InputSemanticSpec(
            "input.protocol.sits",
            ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
            1.0e-4,
        ),
        InputSemanticSpec(
            "input.protocol.sits.nk_typed_restart",
            ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
            1.0e-4,
        ),
    )
    typed_config = contracts["input.protocol.sits"]
    assert typed_config.status == "supported"
    assert typed_config.minimum_evidence == "E3"
    assert typed_config.case_ids == (
        case.name,
        "normal_sits_typed_inactive_configuration",
    )
    assert typed_config.assertion_ids == (
        "input_semantic_equivalence",
        "mdout_deterministic_equivalence",
    )
    assert "/sits/config" in typed_config.bundled_surface
    assert "/sits/atom_indices" in typed_config.bundled_surface


def test_focused_typed_sits_inactive_case_preserves_missing_mode_semantics():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_sits_typed_inactive_configuration"
    )

    assert case.fixture_case == FOCUSED_SITS_TYPED_INACTIVE_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.restart_load_policy == "structural"
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.sits",
    )
    assert case.assertion_ids == ("mdout_deterministic_equivalence",)
    assert case.name not in INPUT_SEMANTIC_SPECS_BY_CASE

    typed_config = contracts["input.protocol.sits"]
    assert case.name in typed_config.case_ids
    assert "mdout_deterministic_equivalence" in typed_config.assertion_ids


def test_focused_typed_sits_controls_reject_ignored_payloads():
    pe_a_result = _assert_sits_typed_control_response(
        "typed SITS",
        "pe_a_half",
        {"SITS_bias": -1.4591, "SITS_fb": 0.6471},
        0.012950390577316284,
    )
    assert pe_a_result["maximum_force_delta"] > 0.01
    atom_result = _assert_sits_typed_control_response(
        "typed SITS",
        "single_selected_atom",
        {
            "SITS_AA_kAB": -0.61,
            "SITS_bias": -0.7295,
            "SITS_fb": 0.6471,
        },
        0.02654215693473816,
    )
    assert atom_result["maximum_force_delta"] > 0.02
    precedence_result = _assert_sits_typed_control_response(
        "typed SITS",
        "explicit_config_precedence",
        {"SITS_bias": -1.4591, "SITS_fb": 0.6471},
        0.012950390577316284,
    )
    assert precedence_result["maximum_force_delta"] > 0.01

    with pytest.raises(AssertionError, match="SITS_bias oracle"):
        _assert_sits_typed_control_response(
            "ignored pe_a",
            "pe_a_half",
            {"SITS_bias": -0.5317, "SITS_fb": 0.7049},
            0.012950390577316284,
        )
    with pytest.raises(AssertionError, match="SITS_AA_kAB oracle"):
        _assert_sits_typed_control_response(
            "ignored atom indices",
            "single_selected_atom",
            {
                "SITS_AA_kAB": -1.22,
                "SITS_bias": -0.5317,
                "SITS_fb": 0.7049,
            },
            0.02654215693473816,
        )
    with pytest.raises(AssertionError, match="force response is too small"):
        _assert_sits_typed_control_response(
            "ignored force response",
            "single_selected_atom",
            {
                "SITS_AA_kAB": -0.61,
                "SITS_bias": -0.7295,
                "SITS_fb": 0.6471,
            },
            0.0,
        )


def test_focused_sits_gate_rejects_initialization_only_and_unscaled_force():
    rows = [
        {
            "SITS_AA_kAB": -1.22,
            "SITS_bias": -0.5317,
            "SITS_fb": 0.7049,
            "LJ_short": -1.0,
            "LJ": -1.0,
            "PM": -0.5,
            "potential": -1.28,
            "eff_pot": -1.2829471,
        }
    ]
    scaled_force = (
        0.1828715056180954,
        1.096548518653151e-09,
        -1.2317579178855453e-09,
        -0.2489061951637268,
        -7.1972111603813e-10,
        1.3064306303434137e-09,
    )
    result = _assert_sits_nk_typed_restart_oracle("SITS", rows, scaled_force)
    assert result["nk"] == [1.0, 4.0]
    assert result["maximum_abs_force"] > 0.2

    initialization_only = [
        {
            "SITS_AA_kAB": 0.0,
            "SITS_bias": 0.0,
            "SITS_fb": 1.0,
            "LJ_short": -1.0,
            "LJ": -1.0,
            "PM": -0.5,
            "potential": -1.28,
            "eff_pot": -1.28,
        }
    ]
    with pytest.raises(AssertionError, match="SITS_AA_kAB oracle"):
        _assert_sits_nk_typed_restart_oracle(
            "initialization only",
            initialization_only,
            scaled_force,
        )
    with pytest.raises(AssertionError, match="SITS force oracle"):
        _assert_sits_nk_typed_restart_oracle(
            "unscaled force",
            rows,
            (
                0.21928882598876953,
                1.096548518653151e-09,
                -1.2317579178855453e-09,
                -0.2489061951637268,
                -7.1972111603813e-10,
                1.3064306303434137e-09,
            ),
        )

    nk_control = [dict(rows[0], SITS_bias=-0.1833, SITS_fb=0.8677)]
    with pytest.raises(AssertionError, match="SITS_bias oracle"):
        _assert_sits_nk_typed_restart_oracle(
            "symmetric Nk control", nk_control, scaled_force
        )


def test_focused_steering_case_requires_cv_sidecar_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_steering_cv_sidecar_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_STEERING_CV_SIDECAR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.steering.cv_sidecar",
    )
    assert spec == (
        InputSemanticSpec(
            "input.protocol.steering.cv_sidecar", ("steer_cv",), 1.0e-6
        ),
    )
    sidecar = contracts["input.protocol.steering.cv_sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (case.name,)
    assert sidecar.assertion_ids == ("input_semantic_equivalence",)
    typed = contracts["input.protocol.steering"]
    assert typed.status == "supported"
    assert typed.case_ids == ("normal_steering_cv_typed_nonzero",)


def test_focused_typed_steering_case_requires_cv_config_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_steering_cv_typed_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_STEERING_CV_TYPED_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.steering",
        "input.protocol.cv",
    )
    assert spec == (
        InputSemanticSpec("input.protocol.steering", ("steer_cv",), 1.0e-6),
        InputSemanticSpec("input.protocol.cv", ("steer_cv",), 1.0e-6),
    )
    typed = contracts["input.protocol.steering"]
    assert typed.status == "supported"
    assert typed.legacy_surface == (
        "steer block and referenced CV definitions in cv_in_file"
    )
    assert typed.bundled_surface == "/cv/config"
    assert typed.case_ids == (case.name,)
    assert typed.assertion_ids == ("input_semantic_equivalence",)
    assert typed.reason == ""


def test_focused_steering_gate_rejects_zero_weight_energy_and_force():
    rows = [
        {
            "steer_cv": 3.0,
            "potential": 3.0,
            "eff_pot": 3.0,
            "PM": 0.0,
            "temperature": 0.0,
        }
    ]
    force = (2.0, 0.0, 0.0, -2.0, 0.0, 0.0)
    result = _assert_steering_cv_oracle("steering", rows, force)
    assert result["steering_energy"] == 3.0
    assert result["maximum_abs_force"] == 2.0

    zero_rows = [
        {
            "steer_cv": 0.0,
            "potential": 0.0,
            "eff_pot": 0.0,
            "PM": 0.0,
            "temperature": 0.0,
        }
    ]
    with pytest.raises(AssertionError, match="steering steer_cv oracle"):
        _assert_steering_cv_oracle("weight=0", zero_rows, (0.0,) * 6)
    with pytest.raises(AssertionError, match="steering force oracle"):
        _assert_steering_cv_oracle("zero force", rows, (0.0,) * 6)


@pytest.mark.parametrize(
    ("rows", "forces", "message"),
    [
        (
            [{"eff_pot": -13.0 / 12.0}],
            (
                -1.0 / 16.0,
                0.0,
                0.0,
                1.0 / 9.0,
                0.0,
                0.0,
                1.0 / 16.0 - 1.0 / 9.0,
                0.0,
                0.0,
            ),
            "exclusion energy mismatch",
        ),
        (
            [{"eff_pot": -1.0 / 12.0}],
            (0.0,) * 9,
            "exclusion force oracle mismatch",
        ),
    ],
)
def test_focused_exclusions_oracle_rejects_ignored_or_wrong_payload(
    rows, forces, message
):
    with pytest.raises(AssertionError, match=message):
        _assert_exclusion_coulomb_oracle("exclusions", rows, forces)


def test_focused_residue_gate_retains_partition_and_pbc_mutations():
    rows = [{"bond": 2.0}]
    forces = (
        4.0,
        0.0,
        0.0,
        -4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    positions = (
        19.0,
        0.0,
        0.0,
        21.0,
        0.0,
        0.0,
        25.0,
        0.0,
        0.0,
        28.0,
        0.0,
        0.0,
    )
    result = _assert_residue_pbc_mapping_oracle(
        "residue",
        rows,
        forces,
        residue_numbers=2,
        runtime_residue_numbers=2,
        positions=positions,
    )
    assert result["runtime_residue_numbers"] == 2
    assert result["maximum_abs_force"] == 4.0

    with pytest.raises(
        AssertionError, match="runtime residue partition mismatch"
    ):
        _assert_residue_pbc_mapping_oracle(
            "split-to-singletons",
            rows,
            forces,
            residue_numbers=2,
            runtime_residue_numbers=4,
            positions=positions,
        )
    wrong_positions = (*positions[:6], 5.0, *positions[7:])
    with pytest.raises(AssertionError, match="PBC mapping oracle mismatch"):
        _assert_residue_pbc_mapping_oracle(
            "wrong mapping",
            rows,
            forces,
            residue_numbers=2,
            runtime_residue_numbers=2,
            positions=wrong_positions,
        )


def test_focused_residue_gate_rejects_wrong_virial_mapping_or_force():
    rows = [{"bond": 2.0, "restrain": 2.0, "pressure": 0.04, "Pxx": 0.11}]
    forces = (
        0.0033416748,
        0.0,
        0.0,
        -4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    positions = (
        19.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        5.0,
        0.0,
        0.0,
        8.0,
        0.0,
        0.0,
    )
    result = _assert_residue_com_res_virial_oracle(
        "residue",
        rows,
        forces,
        residue_numbers=2,
        positions=positions,
    )
    assert result["residue_numbers"] == 2
    assert result["maximum_abs_force"] == 4.0

    with pytest.raises(AssertionError, match="residue state mismatch"):
        _assert_residue_com_res_virial_oracle(
            "wrong-count",
            rows,
            forces,
            residue_numbers=4,
            positions=positions,
        )

    with pytest.raises(AssertionError, match="com_res pressure mismatch"):
        _assert_residue_com_res_virial_oracle(
            "wrong-pressure",
            [{"bond": 2.0, "restrain": 2.0, "pressure": 0.0, "Pxx": 0.11}],
            forces,
            residue_numbers=2,
            positions=positions,
        )

    with pytest.raises(AssertionError, match="com_res Pxx mismatch"):
        _assert_residue_com_res_virial_oracle(
            "wrong-Pxx",
            [{"bond": 2.0, "restrain": 2.0, "pressure": 0.04, "Pxx": 0.0}],
            forces,
            residue_numbers=2,
            positions=positions,
        )

    with pytest.raises(AssertionError, match="force payload is trivial"):
        _assert_residue_com_res_virial_oracle(
            "trivial-force",
            rows,
            (0.0,) * 12,
            residue_numbers=2,
            positions=positions,
        )


@pytest.mark.parametrize(
    ("legacy", "bundled", "message"),
    [
        ([0.0, 0.0], [0.0, 0.0], "legacy force is all trivial"),
        ([1.0, -1.0], [1.0, 0.0], "mismatch at index"),
    ],
)
def test_focused_edip_force_gate_rejects_trivial_or_mismatched_force(
    legacy, bundled, message
):
    with pytest.raises(AssertionError, match=message):
        _assert_nontrivial_equivalent_forces("EDIP force", legacy, bundled)


def test_input_semantic_registry_uses_owned_observables_not_initialization_logs():
    specs = [
        *RERUN_INPUT_SEMANTIC_SPECS,
        *(
            spec
            for case_specs in INPUT_SEMANTIC_SPECS_BY_CASE.values()
            for spec in case_specs
        ),
    ]

    assert specs
    assert all(spec.observables for spec in specs)
    assert all(
        "initial" not in observable.lower()
        for spec in specs
        for observable in spec.observables
    )


def test_h5_string_reader_preserves_multiline_diagnostic_text(tmp_path):
    path = tmp_path / "meta.obs.spg.h5md"
    dataset = "/parameters/sponge/metadynamics/meta0/hills"
    expected = "Step 0\n  indented continuation\nStep 1\n"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            dataset,
            data=expected,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    assert _h5_string_values(path, dataset) == [expected]


def test_input_semantic_contract_inventory_is_explicit_and_evidence_gated():
    contracts = load_contract_registry()
    specialized_semantic_assertions = {
        "input.bias.nhc": "restart_dynamic_continuation_equivalence",
        "input.bias.metadynamics": (
            "restart_protocol_full_continuation_equivalence"
        ),
        "input.protocol.constraint.sidecar": (
            "constraint_geometry_equivalence"
        ),
        "input.protocol.constraint": "constraint_geometry_equivalence",
    }
    runtime_spec_ids = {
        spec.contract_id
        for specs in INPUT_SEMANTIC_SPECS_BY_CASE.values()
        for spec in specs
    } | {spec.contract_id for spec in RERUN_INPUT_SEMANTIC_SPECS}

    assert REQUIRED_INPUT_SEMANTIC_CONTRACTS <= set(contracts)
    assert runtime_spec_ids | set(specialized_semantic_assertions) == {
        contract_id
        for contract_id in REQUIRED_INPUT_SEMANTIC_CONTRACTS
        if contracts[contract_id].status == "supported"
    }
    for contract_id in REQUIRED_INPUT_SEMANTIC_CONTRACTS:
        contract = contracts[contract_id]
        assert contract.minimum_evidence == "E3"
        if contract.status == "supported":
            expected_assertion = specialized_semantic_assertions.get(
                contract_id, "input_semantic_equivalence"
            )
            assert expected_assertion in contract.assertion_ids
        else:
            assert contract.status == "deferred"
            assert contract.reason


@pytest.mark.parametrize(
    ("legacy", "bundled", "message"),
    [
        ([{"other": 1.0}], [{"bond": 1.0}], "missing module-owned observable"),
        ([{"bond": 0.0}], [{"bond": 0.0}], "all trivial"),
        ([{"bond": 1.0}], [{"bond": 2.0}], "mismatch at row"),
    ],
)
def test_input_semantic_gate_rejects_activation_only_evidence(
    legacy, bundled, message
):
    with pytest.raises(AssertionError, match=message):
        assert_module_semantics(
            "bond",
            legacy,
            bundled,
            InputSemanticSpec("input.topology.bond", ("bond",)),
            deterministic=True,
        )


def test_rerun_overrides_remain_root_keys_before_module_tables():
    updated = _insert_root_toml_keys(
        'mode = "rerun"\n[REAXFF]\nin_file = "reaxff.txt"\n',
        ["rerun_start = 1", 'crd = "traj.dat"'],
    )

    parsed = tomllib.loads(updated)
    assert parsed["rerun_start"] == 1
    assert parsed["crd"] == "traj.dat"
    assert parsed["REAXFF"] == {"in_file": "reaxff.txt"}


def _synthetic_matrix_sample_plan(scenario) -> dict[str, object]:
    runtime_case = next(
        case
        for case in MATRIX_RUNTIME_CASES
        if case.scenario_id == scenario.scenario_id
    )
    return _matrix_sample_plan(
        runtime_case, profile="production", fast_mode=False
    )


def _ready_promotion_fixture():
    matrix = load_execution_matrix()
    contract_id = "runtime.synthetic_matrix"
    scenarios = []
    report_cases = {}
    for index, scenario in enumerate(matrix.scenarios):
        case_id = f"synthetic_matrix_case_{index}"
        scenarios.append(
            replace(
                scenario,
                status="executable",
                case_ids=(case_id,),
                reason="",
            )
        )
        report_cases[case_id] = {
            "metadata": {
                **scenario.axis_values(),
                **(
                    {
                        "gpu_device_map": "0",
                        "gpu_model": "synthetic GPU",
                        "cuda_driver_version": "999.0",
                        "cuda_runtime_version": "13.0",
                    }
                    if scenario.backend == "gpu"
                    else {}
                ),
                "omp_num_threads": scenario.omp_threads,
                "mpi_rank_count": scenario.mpi_ranks,
                "rank0_output_owner": True,
                "sample_plan": _synthetic_matrix_sample_plan(scenario),
            },
            "records": [
                {
                    "case_id": case_id,
                    "contract_id": contract_id,
                    "assertion_id": "synthetic_equivalence",
                    "evidence_level": "E3",
                    "status": "passed",
                    "details": {"method": "synthetic_runtime_comparison"},
                }
            ],
        }
    matrix = replace(
        matrix,
        promotion_state="candidate",
        scenarios=tuple(scenarios),
    )
    contracts = {
        contract_id: ContractSpec(
            contract_id=contract_id,
            direction="runtime",
            component="execution_matrix",
            status="supported",
            minimum_evidence="E3",
            legacy_surface="synthetic legacy",
            bundled_surface="synthetic bundled",
            case_ids=tuple(report_cases),
            assertion_ids=("synthetic_equivalence",),
            inventory_refs=(),
        ),
        VDS_CROSS_PROCESS_RESUME_CONTRACT: ContractSpec(
            contract_id=VDS_CROSS_PROCESS_RESUME_CONTRACT,
            direction="output",
            component="trajectory_vds",
            status="unsupported",
            minimum_evidence="E4",
            legacy_surface="legacy process restart and append",
            bundled_surface="VDS reopen-and-append",
            case_ids=(),
            assertion_ids=(),
            inventory_refs=(),
            reason=(
                "The VDS writer does not implement an append/resume open mode."
            ),
        ),
    }
    scope_exclusions = required_scope_exclusions(contracts)
    runs = tuple(
        ProductionRun(
            run_id=f"run-{index}",
            passed=True,
            retry_count=0,
            runtime_ratio=1.0,
            finalize_fraction=0.1,
            output_bytes_ratio=1.0,
            comparator_mutations_rejected=True,
            scope_exclusions=scope_exclusions,
            scope_statement=SCOPED_EQUIVALENCE_STATEMENT,
        )
        for index in range(matrix.required_consecutive_production_runs)
    )
    return (
        matrix,
        contracts,
        {
            "cases": report_cases,
            "scope_exclusions": list(scope_exclusions),
            "scope_statement": SCOPED_EQUIVALENCE_STATEMENT,
        },
        runs,
    )


def test_execution_matrix_enumerates_every_required_axis_and_risk_pair():
    matrix = load_execution_matrix()
    runtime_cases = {case.scenario_id: case for case in MATRIX_RUNTIME_CASES}
    case_ids = {case.name for case in _cases_for_profile()} | set(runtime_cases)

    validate_execution_matrix(matrix, case_ids)

    assert set(matrix.required_axes) == {
        "ensemble",
        "thermostat",
        "box_geometry",
        "constraint",
        "backend",
        "omp_threads",
        "mpi_ranks",
        "comparison",
        "feature_family",
        "vds",
    }
    assert matrix.required_consecutive_production_runs == 3
    assert matrix.promotion_state == "candidate"
    assert set(matrix.required_axes["feature_family"]) == {
        "core",
        "manybody",
        "sits",
        "metadynamics",
        "positional_restraint",
        "cv_restraint",
        "soft_wall",
        "virtual_atom",
        "constraint",
    }
    assert set(matrix.required_axes["vds"]) == {False, True}
    executable = [
        scenario
        for scenario in matrix.scenarios
        if scenario.status == "executable"
    ]
    assert len(executable) == 39
    for scenario in executable:
        assert scenario.case_ids[0] == scenario.scenario_id
        runtime_case = runtime_cases[scenario.scenario_id]
        assert scenario.axis_values() == runtime_case.axis_values()
        assert scenario.barostat == runtime_case.barostat
        assert scenario.tier == runtime_case.tier
        if runtime_case.source_case_id is None:
            assert scenario.case_ids == (scenario.scenario_id,)
        else:
            assert scenario.case_ids == (
                scenario.scenario_id,
                runtime_case.source_case_id,
            )


def test_production_statistical_matrix_has_predeclared_power_floor():
    statistical_cases = [
        case for case in MATRIX_RUNTIME_CASES if case.statistical
    ]
    assert statistical_cases
    for case in statistical_cases:
        plan = _matrix_sample_plan(case, profile="production", fast_mode=False)
        assert plan["step_limit"] == 10_000
        assert plan["replica_count"] == MINIMUM_PRODUCTION_STATISTICAL_REPLICAS
        assert plan["inference_unit"] == "replica"
        assert plan["blocks_per_replica_lower_bound"] == 8
        assert plan["paired_block_count_lower_bound"] == 384
        assert plan["paired_block_count_lower_bound"] >= (
            MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS
        )

        fast_plan = _matrix_sample_plan(
            case, profile="production", fast_mode=True
        )
        assert fast_plan["paired_block_count_lower_bound"] < (
            MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS
        )

    assert PROFILE_LIMITS["production"]["normal_relative_margin"] == 1.0e-2
    assert PROFILE_LIMITS["production"]["normal_absolute_margin"] == 5.0e-2


def test_statistical_matrix_compares_dynamic_h5_families_not_restart_snapshot():
    statistical_cases = [
        case for case in MATRIX_RUNTIME_CASES if case.statistical
    ]

    assert statistical_cases
    for case in statistical_cases:
        assert _matrix_h5_families(case) == {"trajectory", "observable"}


def test_execution_matrix_fixture_hashes_are_reviewed_and_pinned():
    manifest = json.loads(
        (MATRIX_FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["generator_commit"] == "2c0dc3e"
    assert manifest["initial_velocity_seed"] == 20260709
    assert manifest["nonorthogonal_box_angles_degrees"] == [80.0, 100.0, 110.0]
    assert set(manifest["sha256"]) == {
        "common/mdin.bundled.spg.toml",
        "common/legacy_sidecars/LJ_in_file/tip3p_LJ.txt",
        "common/legacy_sidecars/bond_in_file/tip3p_bond.txt",
        "common/legacy_sidecars/charge_in_file/tip3p_charge.txt",
        "common/legacy_sidecars/exclude_in_file/tip3p_exclude.txt",
        "common/legacy_sidecars/mass_in_file/tip3p_mass.txt",
        "common/topology.spgt.h5",
        "common/protocol.spgp.h5",
        "orthogonal/restart.spgr.h5",
        "nonorthogonal/restart.spgr.h5",
        "initial_velocity.txt",
    }
    for relative_path, expected in manifest["sha256"].items():
        actual = hashlib.sha256(
            (MATRIX_FIXTURE_ROOT / relative_path).read_bytes()
        ).hexdigest()
        assert actual == expected, relative_path

    topology_path = MATRIX_FIXTURE_ROOT / "common/topology.spgt.h5"
    residue_sidecar = (
        MATRIX_FIXTURE_ROOT
        / "common/legacy_sidecars/residue_in_file/tip3p_residue.txt"
    )
    assert not residue_sidecar.exists()
    with h5py.File(topology_path, "r") as topology:
        assert "/atoms/residue_index" in topology
        assert "/residues/atom_offset" in topology
        sidecar_keys = {
            value.decode() if isinstance(value, bytes) else str(value)
            for value in topology[
                "/parameters/sponge/files/legacy_sidecars/key"
            ][()]
        }
        assert "residue_in_file" not in sidecar_keys


def test_full_contract_fixtures_have_one_typed_residue_owner():
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        bundle = FULL_CONTRACT_FIXTURE_ROOT / family / "bundle"
        topology_path = bundle / "topology.spgt.h5"
        with h5py.File(topology_path, "r") as topology:
            assert "/atoms/residue_index" in topology, family
            assert "/residues/atom_offset" in topology, family
            sidecar_root = "/parameters/sponge/files/legacy_sidecars"
            sidecar_keys = set()
            if sidecar_root in topology:
                sidecar_keys = {
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in topology[f"{sidecar_root}/key"][()]
                }
            assert "residue_in_file" not in sidecar_keys, family
        assert not (bundle / "legacy_sidecars/residue_in_file").exists(), family

    manifest_path = (
        FULL_CONTRACT_FIXTURE_ROOT
        / "bundled_input_with_legacy_sidecar/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    residue_entry = next(
        entry
        for entry in manifest["entries"]
        if entry["contract_id"] == "topology.residue"
    )
    assert residue_entry["status"] == "typed_converted"
    assert "sidecar_key" not in residue_entry
    assert "sidecar_path" not in residue_entry


def test_gpu_rank2_device_mapping_is_explicit_and_configurable(monkeypatch):
    rank2 = next(
        case
        for case in MATRIX_RUNTIME_CASES
        if case.backend == "gpu" and case.mpi_ranks == 2
    )
    rank1 = next(
        case
        for case in MATRIX_RUNTIME_CASES
        if case.backend == "gpu" and case.mpi_ranks == 1
    )

    monkeypatch.delenv("SPONGE_BUNDLED_IO_AB_GPU_DEVICES", raising=False)
    default_keys = _runtime_keys(rank2, 20260709, step_limit=64, interval=1)
    assert 'device = "0"' in default_keys
    assert not any(
        key.startswith("device =")
        for key in _runtime_keys(rank1, 20260709, step_limit=64, interval=1)
    )

    monkeypatch.setenv("SPONGE_BUNDLED_IO_AB_GPU_DEVICES", "0 1")
    explicit_keys = _runtime_keys(rank2, 20260709, step_limit=64, interval=1)
    assert 'device = "0 1"' in explicit_keys


def test_execution_matrix_rejects_removed_axis_and_unmapped_combination():
    matrix = load_execution_matrix()
    axes = dict(matrix.required_axes)
    axes.pop("mpi_ranks")
    with pytest.raises(AssertionError, match="axes differ"):
        validate_execution_matrix(replace(matrix, required_axes=axes))

    impossible = {
        "ensemble": "npt",
        "thermostat": "nose_hoover_chain",
        "comparison": "deterministic",
    }
    with pytest.raises(AssertionError, match="has no scenario"):
        validate_execution_matrix(
            replace(
                matrix,
                required_combinations=(
                    *matrix.required_combinations,
                    impossible,
                ),
            )
        )

    without_rank2_manybody = tuple(
        combination
        for combination in matrix.required_combinations
        if combination
        != {
            "backend": "cpu",
            "mpi_ranks": 2,
            "feature_family": "manybody",
        }
    )
    with pytest.raises(AssertionError, match="missing required feature"):
        validate_execution_matrix(
            replace(
                matrix,
                required_combinations=without_rank2_manybody,
            )
        )

    unknown_case = replace(
        matrix.scenarios[0], case_ids=("missing_executable_case",)
    )
    with pytest.raises(AssertionError, match="references unknown cases"):
        validate_execution_matrix(
            replace(
                matrix,
                scenarios=(unknown_case, *matrix.scenarios[1:]),
            ),
            available_case_ids={case.name for case in _cases_for_profile()}
            | {case.scenario_id for case in MATRIX_RUNTIME_CASES},
        )


def test_candidate_execution_matrix_cannot_be_promoted_by_declarations_only():
    matrix = load_execution_matrix()
    decision = evaluate_promotion_readiness(
        matrix,
        load_contract_registry(),
        evidence_report=None,
        production_runs=(),
    )

    assert decision.ready is False
    assert not any(
        "promotion_state is shadow" in item for item in decision.blockers
    )
    assert any(
        "does not prove environment" in item for item in decision.blockers
    )
    assert any("contracts lack evidence" in item for item in decision.blockers)
    assert any("consecutive retry-free" in item for item in decision.blockers)


def test_promotion_requires_environment_metadata_for_every_scenario():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    assert evaluate_promotion_readiness(matrix, contracts, report, runs).ready
    contract_id = next(iter(contracts))
    contract_case_id = contracts[contract_id].case_ids[0]
    contract_report = {
        "cases": {
            contract_case_id: {
                "records": [
                    {
                        "case_id": contract_case_id,
                        "contract_id": contract_id,
                        "assertion_id": "synthetic_equivalence",
                        "evidence_level": "E3",
                        "status": "passed",
                        "details": {"method": "synthetic_runtime_comparison"},
                    }
                ]
            }
        },
        "scope_exclusions": list(required_scope_exclusions(contracts)),
        "scope_statement": SCOPED_EQUIVALENCE_STATEMENT,
    }
    assert evaluate_promotion_readiness(
        matrix,
        contracts,
        contract_report,
        runs,
        scenario_evidence_report=report,
    ).ready

    first = matrix.scenarios[0]
    case_payload = report["cases"][first.case_ids[0]]
    case_payload["metadata"]["omp_num_threads"] = first.omp_threads + 1
    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        first.scenario_id in blocker and "does not prove environment" in blocker
        for blocker in decision.blockers
    )


def test_promotion_metadata_must_prove_feature_family_and_vds():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    feature_scenario = next(
        scenario
        for scenario in matrix.scenarios
        if scenario.feature_family != "core"
    )
    case_payload = report["cases"][feature_scenario.case_ids[0]]
    case_payload["metadata"]["feature_family"] = "core"
    case_payload["metadata"]["vds"] = not feature_scenario.vds

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        feature_scenario.scenario_id in blocker
        and "does not prove environment" in blocker
        for blocker in decision.blockers
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "replica_count",
            MINIMUM_PRODUCTION_STATISTICAL_REPLICAS - 1,
        ),
        (
            "paired_block_count_lower_bound",
            MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS - 1,
        ),
        ("fast_mode", True),
        ("inference_unit", "block"),
    ],
)
def test_promotion_rejects_underpowered_statistical_matrix(field, value):
    matrix, contracts, report, runs = _ready_promotion_fixture()
    scenario = next(
        item for item in matrix.scenarios if item.comparison == "statistical"
    )
    sample_plan = report["cases"][scenario.case_ids[0]]["metadata"][
        "sample_plan"
    ]
    sample_plan[field] = value

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        scenario.scenario_id in blocker
        and "does not prove environment" in blocker
        for blocker in decision.blockers
    )


def test_promotion_metadata_must_prove_gpu_hardware():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    gpu_scenario = next(
        scenario for scenario in matrix.scenarios if scenario.backend == "gpu"
    )
    case_payload = report["cases"][gpu_scenario.case_ids[0]]
    case_payload["metadata"].pop("gpu_model")

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        gpu_scenario.scenario_id in blocker
        and "does not prove environment" in blocker
        for blocker in decision.blockers
    )


def test_promotion_rejects_a_handwritten_pass_record_without_details():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    first_case = next(iter(report["cases"].values()))
    first_case["records"][0]["details"] = {}

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        "contract evidence schema is invalid" in blocker
        and "non-empty details" in blocker
        for blocker in decision.blockers
    )


def test_promotion_rejects_e0_presented_as_runtime_behavior():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    first_case = next(iter(report["cases"].values()))
    first_case["records"][0]["evidence_level"] = "E0"

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        "contract evidence schema is invalid" in blocker
        and "requires E3, got E0" in blocker
        for blocker in decision.blockers
    )


def test_promotion_requires_three_consecutive_runs_without_retry():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    runs = (*runs[:-1], replace(runs[-1], retry_count=1))

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any("got 0" in blocker for blocker in decision.blockers)


@pytest.mark.parametrize(
    ("field", "budget_field", "token"),
    [
        ("runtime_ratio", "maximum_runtime_ratio", "runtime_ratio"),
        (
            "finalize_fraction",
            "maximum_finalize_fraction",
            "finalize_fraction",
        ),
        (
            "output_bytes_ratio",
            "maximum_output_bytes_ratio",
            "output_bytes_ratio",
        ),
    ],
)
def test_promotion_rejects_each_performance_budget_violation(
    field, budget_field, token
):
    matrix, contracts, report, runs = _ready_promotion_fixture()
    limit = getattr(matrix.performance_budgets, budget_field)
    runs = (*runs[:-1], replace(runs[-1], **{field: limit + 0.01}))

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(token in blocker for blocker in decision.blockers)


def test_promotion_requires_all_comparator_mutations_to_be_rejected():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    runs = (
        *runs[:-1],
        replace(runs[-1], comparator_mutations_rejected=False),
    )

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        "mutations were not all rejected" in item for item in decision.blockers
    )


def test_promotion_requires_scoped_evidence_report():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    report = dict(report)
    report.pop("scope_exclusions")

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        "scope boundary is invalid" in item for item in decision.blockers
    )


@pytest.mark.parametrize(
    ("field", "value", "token"),
    [
        ("scope_exclusions", (), "scope exclusions"),
        ("scope_statement", "Full equivalence.", "scope statement"),
    ],
)
def test_promotion_requires_scoped_production_runs(field, value, token):
    matrix, contracts, report, runs = _ready_promotion_fixture()
    runs = (*runs[:-1], replace(runs[-1], **{field: value}))

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(token in item for item in decision.blockers)


def test_shadow_workflow_runs_tiers_without_becoming_a_release_gate():
    workflow = AB_SHADOW_WORKFLOW.read_text(encoding="utf-8")
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Bundled I/O A/B Shadow" in workflow
    assert "python -m benchmarks.bundled_io.execution_matrix" in workflow
    assert "smoke-bundled-io-contract" in workflow
    assert "ab-bundled-io-medium" in workflow
    assert "ab-bundled-io-production" in workflow
    assert "test_bundled_io_ab_execution_matrix.py" in workflow
    assert (
        workflow.count("SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS: cpu-rank1") == 2
    )
    assert "SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS: cpu-rank2" in workflow
    assert "SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS: gpu-rank1" in workflow
    assert "SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS: gpu-rank2" in workflow
    assert "pixi install -e dev-cpu-mpi" in workflow
    assert "pixi install -e dev-cuda13" in workflow
    assert "pixi install -e dev-cuda13-mpi" in workflow
    assert "pixi run -e dev-cuda13-mpi configure-cuda-mpi" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu, cuda13]" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu, cuda13, mpi]" in workflow
    assert "nvidia-smi --query-gpu=name,driver_version" in workflow
    assert (
        "SPONGE_XPONGE_ROOT: ${{ github.workspace }}/.xponge-source" in workflow
    )
    assert workflow.count("repository: SPONGEMM/XPONGE") == 6
    assert workflow.count("ref: d607b8965d18c4ac3c307d00aba974da7e81fa89") == 6
    assert workflow.count("path: .xponge-source") == 6
    assert workflow.count("Apply reviewed XPONGE converter runtime patch") == 6
    assert workflow.count("xponge_converter_runtime.patch") == 6
    assert (
        hashlib.sha256(XPONGE_RUNTIME_PATCH.read_bytes()).hexdigest()
        == "9f5730ab18aa2946d47e3a40c0f9934e86e16b17b7ce64588d6e867af870d1f4"
    )
    assert (
        workflow.count(
            "SPONGE_BUNDLED_IO_AB_RUN_ID: "
            "${{ github.run_id }}-${{ github.run_attempt }}"
        )
        == 4
    )
    assert "SPONGE_BUNDLED_IO_AB_COMPARATOR_EVIDENCE:" in workflow
    assert "ab_comparator_evidence.json" in workflow
    assert workflow.count("continue-on-error: true") == 5
    assert "bundled-io-ab-gpu-${{ github.run_id }}" in workflow
    assert "bundled-io-ab-gpu-mpi-${{ github.run_id }}" in workflow
    assert "schedule:" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "bundled-io-ab-shadow" not in release
    assert "ab-bundled-io-production" not in release


PROMOTION_RUN_ID = "production-20260714-1"
PROMOTION_SOURCE_COMMIT = "a" * 40


def _write_promotion_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _promotion_artifacts(tmp_path: Path):
    matrix = load_execution_matrix()
    contracts = load_contract_registry()
    declared_cases = {case.name: case for case in _cases_for_profile()}
    evidence_path = tmp_path / "ab_evidence.json"
    evidence_cases = {}
    for contract_id, contract in contracts.items():
        if contract.status != "supported":
            continue
        case_id = contract.case_ids[0]
        assertion_id = next(
            assertion_id
            for assertion_id in contract.assertion_ids
            if assertion_id in declared_cases[case_id].assertion_ids
        )
        details = {"method": "promotion_fixture_contract_evidence"}
        if assertion_id == "input_semantic_equivalence":
            details["contract_oracle"] = {
                "oracle_contract_id": contract_id,
                "control_mutation": {"kind": "fixture_surface_substitution"},
                "expected_delta": {"relation": "equivalent"},
                "actual_delta": {"result": "equivalent"},
                "compared_payload": ["fixture_observable"],
            }
        evidence_cases.setdefault(
            case_id,
            {"metadata": {"profile": "production"}, "records": []},
        )["records"].append(
            {
                "case_id": case_id,
                "contract_id": contract_id,
                "assertion_id": assertion_id,
                "evidence_level": contract.minimum_evidence,
                "status": "passed",
                "details": details,
            }
        )
    _write_promotion_json(
        evidence_path,
        {
            "schema_version": "1",
            "run_id": PROMOTION_RUN_ID,
            "registry_summary": registry_summary(contracts),
            "scope_exclusions": list(required_scope_exclusions(contracts)),
            "scope_statement": SCOPED_EQUIVALENCE_STATEMENT,
            "cases": evidence_cases,
        },
    )

    matrix_cases = {}
    for index, scenario in enumerate(matrix.scenarios):
        matrix_cases[scenario.case_ids[0]] = {
            "metadata": {
                **scenario.axis_values(),
                **(
                    {
                        "gpu_device_map": "0",
                        "gpu_model": "synthetic GPU",
                        "cuda_driver_version": "999.0",
                        "cuda_runtime_version": "13.0",
                    }
                    if scenario.backend == "gpu"
                    else {}
                ),
                "omp_num_threads": scenario.omp_threads,
                "mpi_rank_count": scenario.mpi_ranks,
                "rank0_output_owner": True,
                "sample_plan": _synthetic_matrix_sample_plan(scenario),
                "performance": {
                    "runtime_ratio": 1.0 + index * 0.001,
                    "finalize_fraction": 0.1 + index * 0.001,
                    "output_bytes_ratio": 1.0 + index * 0.001,
                },
            },
            "records": [
                {
                    "contract_id": "runtime.execution_matrix",
                    "evidence_level": "E3",
                    "status": "passed",
                }
            ],
        }
    midpoint = len(matrix_cases) // 2
    items = list(matrix_cases.items())
    matrix_paths = []
    for part, selected in enumerate((items[:midpoint], items[midpoint:])):
        path = tmp_path / f"ab_matrix_evidence_{part}.json"
        _write_promotion_json(
            path,
            {
                "schema_version": 1,
                "run_id": PROMOTION_RUN_ID,
                "source_commit": PROMOTION_SOURCE_COMMIT,
                "source_tree_state": "clean",
                "cases": dict(selected),
            },
        )
        matrix_paths.append(path)

    comparator_path = tmp_path / "ab_comparator_evidence.json"
    write_comparator_mutation_report(
        comparator_path,
        PROMOTION_RUN_ID,
        {
            test_name: tuple(
                (
                    f"statistics.py::{test_name}[{index}]",
                    "passed",
                )
                for index in range(
                    REQUIRED_COMPARATOR_MUTATION_NODE_COUNTS[test_name]
                )
            )
            for test_name in REQUIRED_COMPARATOR_MUTATION_TESTS
        },
        pytest_exit_status=0,
    )
    return matrix, contracts, evidence_path, matrix_paths, comparator_path


def _derive_promotion_run(artifacts, source_tree_state="clean"):
    matrix, contracts, evidence, matrix_evidence, comparator = artifacts
    return derive_production_run(
        run_id=PROMOTION_RUN_ID,
        source_commit=PROMOTION_SOURCE_COMMIT,
        source_tree_state=source_tree_state,
        retry_count=0,
        evidence_path=evidence,
        matrix_evidence_paths=matrix_evidence,
        comparator_evidence_path=comparator,
        matrix=matrix,
        contracts=contracts,
    )


def test_production_run_is_derived_from_complete_hashed_evidence(tmp_path):
    run, provenance = _derive_promotion_run(_promotion_artifacts(tmp_path))

    assert run.run_id == PROMOTION_RUN_ID
    assert run.passed is True
    assert run.comparator_mutations_rejected is True
    assert run.runtime_ratio == pytest.approx(1.038)
    assert run.finalize_fraction == pytest.approx(0.138)
    assert run.output_bytes_ratio == pytest.approx(1.038)
    assert run.scope_exclusions == required_scope_exclusions(
        load_contract_registry()
    )
    assert run.scope_statement == SCOPED_EQUIVALENCE_STATEMENT
    assert provenance["source_commit"] == PROMOTION_SOURCE_COMMIT
    assert provenance["source_tree_state"] == "clean"
    assert all(
        len(value) == 64
        for key, value in provenance.items()
        if key.endswith("_sha256")
    )

    history_path = tmp_path / "production_history.json"
    append_production_run_history(history_path, run, provenance)
    assert load_production_run_history(history_path) == (run,)
    with pytest.raises(AssertionError, match="duplicate production run ID"):
        append_production_run_history(history_path, run, provenance)


def test_production_run_rejects_partial_comparator_mutation_evidence(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    comparator = artifacts[-1]
    report = json.loads(comparator.read_text(encoding="utf-8"))
    partial = "test_nonfinite_kind_and_sign_mutations_are_rejected"
    report["tests"][partial] = {
        "status": "passed",
        "expected_node_count": 3,
        "nodeids": [f"statistics.py::{partial}[0]"],
    }
    _write_promotion_json(comparator, report)

    with pytest.raises(AssertionError, match="lack passing rejection evidence"):
        _derive_promotion_run(artifacts)


def test_production_run_rejects_missing_scope_exclusion(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    evidence = artifacts[2]
    report = json.loads(evidence.read_text(encoding="utf-8"))
    report["scope_exclusions"].remove(VDS_CROSS_PROCESS_RESUME_CONTRACT)
    _write_promotion_json(evidence, report)

    with pytest.raises(AssertionError, match="scope_exclusions"):
        _derive_promotion_run(artifacts)


def test_history_loader_rejects_missing_scope_boundary(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    run, provenance = _derive_promotion_run(artifacts)
    history_path = tmp_path / "production_history.json"
    append_production_run_history(history_path, run, provenance)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["runs"][0].pop("scope_exclusions")
    _write_promotion_json(history_path, history)

    with pytest.raises(AssertionError, match="scope_exclusions"):
        load_production_run_history(history_path)


def test_production_run_rejects_failed_comparator_pytest_session(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    comparator = artifacts[-1]
    report = json.loads(comparator.read_text(encoding="utf-8"))
    report["pytest_exit_status"] = 1
    _write_promotion_json(comparator, report)

    with pytest.raises(AssertionError, match="pytest_exit_status must be 0"):
        _derive_promotion_run(artifacts)


def test_production_run_rejects_dirty_source_attestation(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)

    with pytest.raises(AssertionError, match="source_tree_state must be clean"):
        _derive_promotion_run(artifacts, source_tree_state="dirty")


def test_production_run_rejects_unproven_matrix_environment(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    matrix_evidence = artifacts[-2]
    report = json.loads(matrix_evidence[0].read_text(encoding="utf-8"))
    first = next(iter(report["cases"].values()))
    first["metadata"]["rank0_output_owner"] = False
    _write_promotion_json(matrix_evidence[0], report)

    with pytest.raises(AssertionError, match="does not prove environment"):
        _derive_promotion_run(artifacts)


def test_production_run_rejects_missing_gpu_matrix_artifacts(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    matrix_evidence = artifacts[-2]
    for path in matrix_evidence:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["cases"] = {
            case_id: payload
            for case_id, payload in report["cases"].items()
            if payload["metadata"]["backend"] != "gpu"
        }
        _write_promotion_json(path, report)

    with pytest.raises(AssertionError, match="does not prove environment"):
        _derive_promotion_run(artifacts)


def test_production_run_rejects_below_minimum_contract_evidence(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    contracts = artifacts[1]
    evidence = artifacts[2]
    report = json.loads(evidence.read_text(encoding="utf-8"))
    e4_contract = next(
        contract_id
        for contract_id, contract in contracts.items()
        if contract.status == "supported" and contract.minimum_evidence == "E4"
    )
    record = next(
        item
        for case_payload in report["cases"].values()
        for item in case_payload["records"]
        if item["contract_id"] == e4_contract
    )
    record["evidence_level"] = "E3"
    _write_promotion_json(evidence, report)

    with pytest.raises(AssertionError, match="requires E4, got E3"):
        _derive_promotion_run(artifacts)


def test_production_run_rejects_nonfinite_performance_evidence(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    matrix_evidence = artifacts[-2]
    report = json.loads(matrix_evidence[0].read_text(encoding="utf-8"))
    first = next(iter(report["cases"].values()))
    first["metadata"]["performance"]["runtime_ratio"] = float("nan")
    _write_promotion_json(matrix_evidence[0], report)

    with pytest.raises(AssertionError, match="finite performance evidence"):
        _derive_promotion_run(artifacts)


def test_matrix_evidence_merge_rejects_stale_and_conflicting_runs(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    matrix_evidence = artifacts[-2]
    stale = json.loads(matrix_evidence[0].read_text(encoding="utf-8"))
    stale["run_id"] = "stale-run"
    _write_promotion_json(matrix_evidence[0], stale)
    with pytest.raises(AssertionError, match="run_id mismatch"):
        merge_matrix_evidence(matrix_evidence, PROMOTION_RUN_ID)

    stale["run_id"] = PROMOTION_RUN_ID
    duplicate_case_id = next(iter(stale["cases"]))
    other = json.loads(matrix_evidence[1].read_text(encoding="utf-8"))
    other["cases"][duplicate_case_id] = {
        "metadata": {"backend": "wrong"},
        "records": [],
    }
    _write_promotion_json(matrix_evidence[0], stale)
    _write_promotion_json(matrix_evidence[1], other)
    with pytest.raises(AssertionError, match="conflicting matrix evidence"):
        merge_matrix_evidence(matrix_evidence, PROMOTION_RUN_ID)


def test_matrix_evidence_merge_rejects_mixed_source_commits(tmp_path):
    artifacts = _promotion_artifacts(tmp_path)
    matrix_evidence = artifacts[-2]
    report = json.loads(matrix_evidence[1].read_text(encoding="utf-8"))
    report["source_commit"] = "b" * 40
    _write_promotion_json(matrix_evidence[1], report)

    with pytest.raises(AssertionError, match="source commits differ"):
        merge_matrix_evidence(matrix_evidence, PROMOTION_RUN_ID)


def test_history_loader_rejects_unattested_handwritten_run(tmp_path):
    history = tmp_path / "production_history.json"
    _write_promotion_json(
        history,
        {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": PROMOTION_RUN_ID,
                    "source_commit": PROMOTION_SOURCE_COMMIT,
                    "passed": True,
                    "retry_count": 0,
                    "runtime_ratio": 1.0,
                    "finalize_fraction": 0.1,
                    "output_bytes_ratio": 1.0,
                    "comparator_mutations_rejected": True,
                }
            ],
        },
    )

    with pytest.raises(AssertionError, match="source_tree_state"):
        load_production_run_history(history)


def test_source_tree_attestation_rejects_commit_and_worktree_drift(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(repo)],
        text=True,
        capture_output=True,
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "tracked.txt"],
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Bundled IO Gate",
            "-c",
            "user.email=bundled-io@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    assert inspect_clean_source_tree(repo, source_commit) == "clean"
    with pytest.raises(AssertionError, match="source commit mismatch"):
        inspect_clean_source_tree(repo, "0" * 40)

    tracked.write_text("modified\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="source tree is dirty"):
        inspect_clean_source_tree(repo, source_commit)
    tracked.write_text("baseline\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="source tree is dirty"):
        inspect_clean_source_tree(repo, source_commit)
