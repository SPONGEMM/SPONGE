from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.bundled_io.ab_contracts import (
    DYNAMIC_RESTART_CONTRACT_STATUSES,
    AssertionEvidence,
    build_case_evidence,
    load_contract_registry,
    load_implementation_inventory,
    update_evidence_report,
    validate_complete_evidence_report,
    validate_contract_registry,
    validate_implementation_inventory,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    _cases_for_profile,
)


def _synthetic_evidence_level(assertion_id: str) -> str:
    if assertion_id == "full_contract_input_inventory":
        return "E0"
    if assertion_id in {
        "restart_continuation_equivalence",
        "restart_dynamic_continuation_equivalence",
        "restart_bussi_continuation_equivalence",
        "restart_pressure_barostat_continuation_equivalence",
        "restart_middle_langevin_continuation_equivalence",
        "restart_andersen_continuation_equivalence",
        "restart_monte_carlo_barostat_continuation_equivalence",
        "restart_protocol_full_continuation_equivalence",
    }:
        return "E4"
    if assertion_id in {
        "stable_failure_semantics",
        "writer_failure_isolation",
    }:
        return "F1"
    return "E3"


def _case(case_name: str):
    return next(case for case in _cases_for_profile() if case.name == case_name)


def _synthetic_details(case, assertion_id, contracts):
    details = {"method": "unit_test_derived_from_declared_case"}
    if assertion_id != "input_semantic_equivalence":
        return details
    details["contract_oracles"] = {
        contract_id: {
            "oracle_contract_id": contract_id,
            "control_mutation": {"kind": "synthetic_surface_substitution"},
            "expected_delta": {"relation": "equivalent"},
            "actual_delta": {"result": "equivalent"},
            "compared_payload": ["synthetic_observable"],
        }
        for contract_id in case.contract_ids
        if assertion_id in contracts[contract_id].assertion_ids
    }
    details["results"] = [
        {
            "contract_id": contract_id,
            "observables": ["synthetic_observable"],
            "replicas": [
                {"legacy_nontrivial": True, "bundled_nontrivial": True}
            ],
        }
        for contract_id in details["contract_oracles"]
    ]
    return details


def test_real_registry_and_case_matrix_are_symmetric():
    contracts = load_contract_registry()
    summary = validate_contract_registry(contracts, _cases_for_profile())
    validate_implementation_inventory(
        contracts, load_implementation_inventory()
    )

    assert summary["contract_count"] == len(contracts)
    assert summary["status_counts"] == {
        "deferred": 0,
        "supported": 91,
        "unsupported": 1,
    }
    assert summary["evidence_class_counts"] == {
        "inventory": 1,
        "conversion": 0,
        "runtime_behavior": 72,
        "continuation": 14,
        "failure_semantics": 5,
    }


def test_dynamic_restart_inventory_rejects_a_missing_module_contract():
    contracts = load_contract_registry()
    removed = "input.restart.dynamic.andersen_rng"
    broken = dict(contracts)
    broken.pop(removed)

    assert removed in DYNAMIC_RESTART_CONTRACT_STATUSES
    with pytest.raises(
        AssertionError,
        match="dynamic restart contract inventory differs",
    ):
        validate_contract_registry(broken, _cases_for_profile())


def test_removing_a_registered_case_fails_the_gate():
    contracts = load_contract_registry()
    cases = _cases_for_profile()

    with pytest.raises(AssertionError, match="missing case"):
        validate_contract_registry(contracts, cases[:-1])


def test_unknown_contract_reference_fails_the_gate():
    contracts = load_contract_registry()
    cases = _cases_for_profile()
    broken = replace(
        cases[0], contract_ids=cases[0].contract_ids + ("unknown.contract",)
    )

    with pytest.raises(AssertionError, match="unknown contracts"):
        validate_contract_registry(contracts, [broken, *cases[1:]])


def test_missing_runtime_assertion_evidence_fails_the_gate():
    contracts = load_contract_registry()
    case = _case("normal_core_h5_output")
    assertions = [
        AssertionEvidence(
            assertion_id="mdout_statistical_equivalence",
            evidence_level="E3",
            details={"method": "test"},
        )
    ]

    with pytest.raises(AssertionError, match="assertion evidence mismatch"):
        build_case_evidence(contracts, case, assertions)


def test_e0_cannot_satisfy_an_e3_contract():
    contracts = load_contract_registry()
    case = _case("normal_core_h5_output")
    assertions = [
        AssertionEvidence(
            assertion_id=assertion_id,
            evidence_level="E0",
            details={"method": "path_presence_only"},
        )
        for assertion_id in case.assertion_ids
    ]

    with pytest.raises(AssertionError, match=r"requires E3, got \['E0'\]"):
        build_case_evidence(contracts, case, assertions)


def test_empty_assertion_details_are_not_behavior_evidence():
    contracts = load_contract_registry()
    case = _case("normal_core_h5_output")
    assertions = [
        AssertionEvidence(
            assertion_id=assertion_id,
            evidence_level=_synthetic_evidence_level(assertion_id),
            details=(
                {}
                if index == 0
                else _synthetic_details(case, assertion_id, contracts)
            ),
        )
        for index, assertion_id in enumerate(case.assertion_ids)
    ]

    with pytest.raises(AssertionError, match="non-empty evidence details"):
        build_case_evidence(contracts, case, assertions)


def test_forged_assertion_id_is_not_contract_evidence():
    contracts = load_contract_registry()
    case = _case("normal_core_h5_output")
    assertions = [
        AssertionEvidence(
            assertion_id=(
                "forged_behavior_assertion" if index == 0 else assertion_id
            ),
            evidence_level=_synthetic_evidence_level(assertion_id),
            details=_synthetic_details(case, assertion_id, contracts),
        )
        for index, assertion_id in enumerate(case.assertion_ids)
    ]

    with pytest.raises(AssertionError, match="assertion evidence mismatch"):
        build_case_evidence(contracts, case, assertions)


def test_shared_input_assertion_requires_the_matching_contract_oracle():
    contracts = load_contract_registry()
    case = _case("normal_core_topology_payload_sensitivity")
    assertions = []
    for assertion_id in case.assertion_ids:
        details = _synthetic_details(case, assertion_id, contracts)
        if assertion_id == "input_semantic_equivalence":
            contract_id = next(iter(details["contract_oracles"]))
            details["contract_oracles"][contract_id]["oracle_contract_id"] = (
                "input.topology.wrong"
            )
        assertions.append(
            AssertionEvidence(
                assertion_id=assertion_id,
                evidence_level=_synthetic_evidence_level(assertion_id),
                details=details,
            )
        )

    with pytest.raises(AssertionError, match="oracle contract mismatch"):
        build_case_evidence(contracts, case, assertions)


def test_evidence_report_merges_cases_and_recomputes_coverage(tmp_path):
    contracts = load_contract_registry()
    cases = _cases_for_profile()[:2]
    report_path = tmp_path / "ab_evidence.json"

    for case in cases:
        assertions = [
            AssertionEvidence(
                assertion_id=assertion_id,
                evidence_level=_synthetic_evidence_level(assertion_id),
                details=_synthetic_details(case, assertion_id, contracts),
            )
            for assertion_id in case.assertion_ids
        ]
        records = build_case_evidence(contracts, case, assertions)
        report = update_evidence_report(
            report_path,
            contracts,
            case,
            records,
            {"profile": "unit"},
            "unit-run",
        )

    assert set(report["cases"]) == {case.name for case in cases}
    assert report["registry_summary"]["status_counts"] == {
        "deferred": 0,
        "supported": 91,
        "unsupported": 1,
    }
    coverage = report["coverage"]
    assert coverage["covered_supported_contract_count"] > 0
    assert coverage["missing_supported_contracts"]
    assert 0.0 < coverage["supported_coverage_fraction"] < 1.0
    assert coverage["status_coverage"]["supported"]["contract_count"] == 91
    assert coverage["status_coverage"]["deferred"]["contract_count"] == 0
    assert coverage["status_coverage"]["unsupported"]["contract_count"] == 1
    assert set(coverage["evidence_class_coverage"]) == {
        "inventory",
        "conversion",
        "runtime_behavior",
        "continuation",
        "failure_semantics",
    }
    with pytest.raises(AssertionError, match="missing supported contracts"):
        validate_complete_evidence_report(report_path, contracts, "unit-run")


def test_failure_evidence_cannot_satisfy_behavior_evidence():
    contracts = load_contract_registry()
    case = _case("normal_core_h5_output")
    assertions = [
        AssertionEvidence(
            assertion_id=assertion_id,
            evidence_level="F1",
            details={"method": "failure_contract"},
        )
        for assertion_id in case.assertion_ids
    ]

    with pytest.raises(AssertionError, match=r"requires E3, got \['F1'\]"):
        build_case_evidence(contracts, case, assertions)


def test_complete_report_requires_and_accepts_every_supported_contract(
    tmp_path,
):
    contracts = load_contract_registry()
    report_path = tmp_path / "complete_ab_evidence.json"

    for case in _cases_for_profile():
        assertions = [
            AssertionEvidence(
                assertion_id=assertion_id,
                evidence_level=_synthetic_evidence_level(assertion_id),
                details=_synthetic_details(case, assertion_id, contracts),
            )
            for assertion_id in case.assertion_ids
        ]
        records = build_case_evidence(contracts, case, assertions)
        update_evidence_report(
            report_path,
            contracts,
            case,
            records,
            {"profile": "unit"},
            "complete-unit-run",
        )

    report = validate_complete_evidence_report(
        report_path, contracts, "complete-unit-run"
    )
    coverage = report["coverage"]
    assert coverage["supported_coverage_fraction"] == 1.0
    assert coverage["missing_supported_contracts"] == []
    assert (
        coverage["evidence_class_coverage"]["inventory"]["coverage_fraction"]
        == 1.0
    )
    assert coverage["evidence_class_coverage"]["conversion"] == {
        "contract_count": 0,
        "contract_ids": [],
        "covered_contract_count": 0,
        "covered_contracts": [],
        "missing_contracts": [],
        "coverage_fraction": 1.0,
    }
    assert all(
        payload["coverage_fraction"] == 1.0
        for payload in coverage["evidence_class_coverage"].values()
    )
    assert coverage["status_coverage"]["supported"] == {
        "contract_count": 91,
        "contract_ids": sorted(
            contract_id
            for contract_id, contract in contracts.items()
            if contract.status == "supported"
        ),
        "evidenced_contract_count": 91,
        "evidenced_contracts": sorted(
            contract_id
            for contract_id, contract in contracts.items()
            if contract.status == "supported"
        ),
    }
    assert (
        coverage["status_coverage"]["deferred"]["evidenced_contract_count"] == 0
    )
    assert (
        coverage["status_coverage"]["unsupported"]["evidenced_contract_count"]
        == 0
    )
