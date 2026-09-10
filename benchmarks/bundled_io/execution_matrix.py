from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from benchmarks.bundled_io.ab_contracts import (
    REGISTRY_PATH,
    SCOPED_EQUIVALENCE_STATEMENT,
    VDS_CROSS_PROCESS_RESUME_CONTRACT,
    ContractSpec,
    load_contract_registry,
    validate_scope_boundary,
    validated_passed_contract_levels,
)

MATRIX_PATH = Path(__file__).with_name("contracts") / "ab_execution_matrix.json"
REQUIRED_AXIS_NAMES = {
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
VALID_PROMOTION_STATES = {"shadow", "candidate", "promoted"}
VALID_SCENARIO_STATUSES = {
    "executable",
    "evidenced",
    "deferred",
    "unsupported",
}
VALID_TIERS = {"medium", "production"}
EVIDENCE_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
REQUIRED_FEATURE_FAMILIES = {
    "manybody",
    "sits",
    "metadynamics",
    "positional_restraint",
    "cv_restraint",
    "soft_wall",
    "virtual_atom",
    "constraint",
}
REQUIRED_RANK2_FEATURE_FAMILIES = {
    "manybody",
    "sits",
    "virtual_atom",
    "constraint",
}
REQUIRED_GPU_RANK2_FEATURE_FAMILIES = {
    "manybody",
    "virtual_atom",
    "constraint",
}
MINIMUM_PRODUCTION_STATISTICAL_REPLICAS = 48
MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS = 384


@dataclass(frozen=True)
class PerformanceBudgets:
    maximum_runtime_ratio: float
    maximum_finalize_fraction: float
    maximum_output_bytes_ratio: float


@dataclass(frozen=True)
class ExecutionScenario:
    scenario_id: str
    ensemble: str
    thermostat: str
    barostat: str
    box_geometry: str
    constraint: str
    backend: str
    omp_threads: int
    mpi_ranks: int
    comparison: str
    feature_family: str
    vds: bool
    tier: str
    status: str
    case_ids: tuple[str, ...]
    reason: str

    def axis_values(self) -> dict[str, object]:
        return {
            "ensemble": self.ensemble,
            "thermostat": self.thermostat,
            "box_geometry": self.box_geometry,
            "constraint": self.constraint,
            "backend": self.backend,
            "omp_threads": self.omp_threads,
            "mpi_ranks": self.mpi_ranks,
            "comparison": self.comparison,
            "feature_family": self.feature_family,
            "vds": self.vds,
        }


@dataclass(frozen=True)
class ExecutionMatrix:
    schema_version: int
    promotion_state: str
    required_consecutive_production_runs: int
    performance_budgets: PerformanceBudgets
    required_axes: Mapping[str, tuple[object, ...]]
    required_combinations: tuple[Mapping[str, object], ...]
    scenarios: tuple[ExecutionScenario, ...]


@dataclass(frozen=True)
class ProductionRun:
    run_id: str
    passed: bool
    retry_count: int
    runtime_ratio: float
    finalize_fraction: float
    output_bytes_ratio: float
    comparator_mutations_rejected: bool
    scope_exclusions: tuple[str, ...] = ()
    scope_statement: str = ""


@dataclass(frozen=True)
class PromotionDecision:
    ready: bool
    blockers: tuple[str, ...]


def load_execution_matrix(path: Path = MATRIX_PATH) -> ExecutionMatrix:
    payload = _load_json_object(path, "A/B execution matrix")
    if payload.get("schema_version") != 1:
        raise AssertionError("A/B execution matrix schema_version must be 1")

    promotion_state = _required_string(payload, "promotion_state", "matrix")
    consecutive_runs = _required_int(
        payload,
        "required_consecutive_production_runs",
        "matrix",
        minimum=1,
    )
    raw_budgets = _required_object(payload, "performance_budgets", "matrix")
    budgets = PerformanceBudgets(
        maximum_runtime_ratio=_required_number(
            raw_budgets, "maximum_runtime_ratio", "performance_budgets"
        ),
        maximum_finalize_fraction=_required_number(
            raw_budgets, "maximum_finalize_fraction", "performance_budgets"
        ),
        maximum_output_bytes_ratio=_required_number(
            raw_budgets, "maximum_output_bytes_ratio", "performance_budgets"
        ),
    )

    raw_axes = _required_object(payload, "required_axes", "matrix")
    required_axes: dict[str, tuple[object, ...]] = {}
    for name, values in raw_axes.items():
        if not isinstance(values, list) or not values:
            raise AssertionError(
                f"execution axis {name} must be a non-empty list"
            )
        if any(not isinstance(value, (str, int)) for value in values):
            raise AssertionError(f"execution axis {name} has an invalid value")
        if len(values) != len(set(values)):
            raise AssertionError(f"execution axis {name} contains duplicates")
        required_axes[name] = tuple(values)

    raw_combinations = payload.get("required_combinations")
    if not isinstance(raw_combinations, list) or not raw_combinations:
        raise AssertionError("required_combinations must be a non-empty list")
    combinations: list[Mapping[str, object]] = []
    for index, item in enumerate(raw_combinations):
        if not isinstance(item, dict) or not item:
            raise AssertionError(
                f"required_combinations[{index}] must be a non-empty object"
            )
        combinations.append(dict(item))

    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise AssertionError(
            "execution matrix scenarios must be a non-empty list"
        )
    scenarios = tuple(
        _parse_scenario(raw, index) for index, raw in enumerate(raw_scenarios)
    )
    matrix = ExecutionMatrix(
        schema_version=1,
        promotion_state=promotion_state,
        required_consecutive_production_runs=consecutive_runs,
        performance_budgets=budgets,
        required_axes=required_axes,
        required_combinations=tuple(combinations),
        scenarios=scenarios,
    )
    validate_execution_matrix(matrix)
    return matrix


def validate_execution_matrix(
    matrix: ExecutionMatrix,
    available_case_ids: Sequence[str] | None = None,
) -> None:
    if matrix.promotion_state not in VALID_PROMOTION_STATES:
        raise AssertionError(
            f"invalid execution matrix promotion_state: {matrix.promotion_state}"
        )
    if set(matrix.required_axes) != REQUIRED_AXIS_NAMES:
        raise AssertionError(
            "execution matrix axes differ: "
            f"expected={sorted(REQUIRED_AXIS_NAMES)}, "
            f"actual={sorted(matrix.required_axes)}"
        )
    _validate_budgets(matrix.performance_budgets)

    scenario_ids = [scenario.scenario_id for scenario in matrix.scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise AssertionError("execution matrix scenario IDs must be unique")

    known_cases = (
        set(available_case_ids) if available_case_ids is not None else None
    )
    observed_axes = {name: set() for name in REQUIRED_AXIS_NAMES}
    for scenario in matrix.scenarios:
        if scenario.status not in VALID_SCENARIO_STATUSES:
            raise AssertionError(
                f"{scenario.scenario_id} has invalid status {scenario.status}"
            )
        if scenario.tier not in VALID_TIERS:
            raise AssertionError(
                f"{scenario.scenario_id} has invalid tier {scenario.tier}"
            )
        if (
            scenario.status in {"executable", "evidenced"}
            and not scenario.case_ids
        ):
            raise AssertionError(
                f"{scenario.status} scenario {scenario.scenario_id} has no case IDs"
            )
        if (
            scenario.status in {"deferred", "unsupported"}
            and not scenario.reason
        ):
            raise AssertionError(
                f"{scenario.status} scenario {scenario.scenario_id} requires a reason"
            )
        if len(scenario.case_ids) != len(set(scenario.case_ids)):
            raise AssertionError(
                f"{scenario.scenario_id} contains duplicate case IDs"
            )
        if known_cases is not None:
            unknown = sorted(set(scenario.case_ids) - known_cases)
            if unknown:
                raise AssertionError(
                    f"{scenario.scenario_id} references unknown cases: {unknown}"
                )

        for name, value in scenario.axis_values().items():
            if value not in matrix.required_axes[name]:
                raise AssertionError(
                    f"{scenario.scenario_id} has undeclared {name}={value!r}"
                )
            observed_axes[name].add(value)

    for name, required_values in matrix.required_axes.items():
        missing = sorted(set(required_values) - observed_axes[name], key=str)
        if missing:
            raise AssertionError(
                f"execution matrix scenarios do not cover {name}: {missing}"
            )

    canonical_combinations: set[tuple[tuple[str, object], ...]] = set()
    for combination in matrix.required_combinations:
        unknown_axes = sorted(set(combination) - REQUIRED_AXIS_NAMES)
        if unknown_axes:
            raise AssertionError(
                f"required combination references unknown axes: {unknown_axes}"
            )
        for name, value in combination.items():
            if value not in matrix.required_axes[name]:
                raise AssertionError(
                    f"required combination has undeclared {name}={value!r}"
                )
        canonical = tuple(sorted(combination.items()))
        if canonical in canonical_combinations:
            raise AssertionError(
                f"duplicate required combination: {dict(combination)}"
            )
        canonical_combinations.add(canonical)
        if not any(
            _scenario_matches(scenario, combination)
            for scenario in matrix.scenarios
        ):
            raise AssertionError(
                f"required execution combination has no scenario: {dict(combination)}"
            )
    required_feature_combinations = {
        tuple(
            sorted(
                {
                    "backend": "cpu",
                    "mpi_ranks": 1,
                    "feature_family": feature_family,
                }.items()
            )
        )
        for feature_family in REQUIRED_FEATURE_FAMILIES
    }
    required_feature_combinations.update(
        tuple(
            sorted(
                {
                    "backend": "cpu",
                    "mpi_ranks": 2,
                    "feature_family": feature_family,
                }.items()
            )
        )
        for feature_family in REQUIRED_RANK2_FEATURE_FAMILIES
    )
    required_feature_combinations.update(
        tuple(
            sorted(
                {
                    "backend": "gpu",
                    "mpi_ranks": 1,
                    "feature_family": feature_family,
                }.items()
            )
        )
        for feature_family in REQUIRED_FEATURE_FAMILIES
    )
    required_feature_combinations.update(
        tuple(
            sorted(
                {
                    "backend": "gpu",
                    "mpi_ranks": 2,
                    "feature_family": feature_family,
                }.items()
            )
        )
        for feature_family in REQUIRED_GPU_RANK2_FEATURE_FAMILIES
    )
    missing_feature_combinations = sorted(
        required_feature_combinations - canonical_combinations, key=str
    )
    if missing_feature_combinations:
        raise AssertionError(
            "execution matrix is missing required feature/environment "
            f"combinations: {missing_feature_combinations}"
        )


def load_production_run_history(path: Path) -> tuple[ProductionRun, ...]:
    payload = _load_json_object(path, "A/B production run history")
    if payload.get("schema_version") != 1:
        raise AssertionError("production run history schema_version must be 1")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        raise AssertionError("production run history runs must be a list")
    runs = tuple(
        _parse_production_run(raw, index) for index, raw in enumerate(raw_runs)
    )
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise AssertionError("production run IDs must be unique")
    return runs


def evaluate_promotion_readiness(
    matrix: ExecutionMatrix,
    contracts: Mapping[str, ContractSpec],
    evidence_report: Mapping[str, object] | None,
    production_runs: Sequence[ProductionRun],
    scenario_evidence_report: Mapping[str, object] | None = None,
) -> PromotionDecision:
    validate_execution_matrix(matrix)
    blockers: list[str] = []
    if matrix.promotion_state == "shadow":
        blockers.append("execution matrix promotion_state is shadow")

    unresolved = sorted(
        scenario.scenario_id
        for scenario in matrix.scenarios
        if scenario.status in {"deferred", "unsupported"}
    )
    if unresolved:
        blockers.append(f"execution scenarios lack evidence: {unresolved}")

    report = evidence_report or {}
    try:
        expected_scope_exclusions = validate_scope_boundary(contracts, report)
    except AssertionError as error:
        blockers.append(f"scope boundary is invalid: {error}")
        expected_scope_exclusions = ()
    report_cases = report.get("cases")
    if not isinstance(report_cases, dict):
        report_cases = {}
    scenario_report = scenario_evidence_report or report
    scenario_cases = scenario_report.get("cases")
    if not isinstance(scenario_cases, dict):
        scenario_cases = {}
    try:
        contract_levels = validated_passed_contract_levels(
            contracts, report_cases
        )
    except AssertionError as error:
        blockers.append(f"contract evidence schema is invalid: {error}")
        contract_levels = {}
    missing_contracts = []
    insufficient_contracts = []
    for contract_id, contract in contracts.items():
        if contract.status != "supported":
            continue
        actual = contract_levels.get(contract_id, ())
        if not actual:
            missing_contracts.append(contract_id)
        elif not any(
            _evidence_satisfies(level, contract.minimum_evidence)
            for level in actual
        ):
            insufficient_contracts.append(
                f"{contract_id}:{sorted(actual)}<{contract.minimum_evidence}"
            )
    if missing_contracts:
        blockers.append(
            f"supported contracts lack evidence: {sorted(missing_contracts)}"
        )
    if insufficient_contracts:
        blockers.append(
            f"supported contracts have insufficient evidence: {insufficient_contracts}"
        )

    for scenario in matrix.scenarios:
        if scenario.status not in {"executable", "evidenced"}:
            continue
        if not any(
            _case_proves_scenario(scenario_cases.get(case_id), scenario)
            for case_id in scenario.case_ids
        ):
            blockers.append(
                f"scenario metadata does not prove environment: {scenario.scenario_id}"
            )

    required = matrix.required_consecutive_production_runs
    qualifying: list[ProductionRun] = []
    for run in reversed(production_runs):
        if not run.passed or run.retry_count != 0:
            break
        qualifying.append(run)
        if len(qualifying) == required:
            break
    if len(qualifying) < required:
        blockers.append(
            f"need {required} consecutive retry-free production runs, got {len(qualifying)}"
        )
    else:
        for run in qualifying:
            if run.scope_exclusions != expected_scope_exclusions:
                blockers.append(
                    f"scope exclusions are missing or stale in run {run.run_id}"
                )
            if run.scope_statement != SCOPED_EQUIVALENCE_STATEMENT:
                blockers.append(
                    f"scope statement is invalid in run {run.run_id}"
                )
            if not run.comparator_mutations_rejected:
                blockers.append(
                    f"comparator mutations were not all rejected in run {run.run_id}"
                )
            _append_budget_blockers(blockers, matrix.performance_budgets, run)

    return PromotionDecision(ready=not blockers, blockers=tuple(blockers))


def _parse_scenario(raw: object, index: int) -> ExecutionScenario:
    if not isinstance(raw, dict):
        raise AssertionError(f"scenarios[{index}] must be an object")
    label = f"scenarios[{index}]"
    case_ids = raw.get("case_ids")
    if not isinstance(case_ids, list) or not all(
        isinstance(case_id, str) and case_id for case_id in case_ids
    ):
        raise AssertionError(f"{label} case_ids must be a string list")
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        raise AssertionError(f"{label} reason must be a string")
    return ExecutionScenario(
        scenario_id=_required_string(raw, "scenario_id", label),
        ensemble=_required_string(raw, "ensemble", label),
        thermostat=_required_string(raw, "thermostat", label),
        barostat=_required_string(raw, "barostat", label),
        box_geometry=_required_string(raw, "box_geometry", label),
        constraint=_required_string(raw, "constraint", label),
        backend=_required_string(raw, "backend", label),
        omp_threads=_required_int(raw, "omp_threads", label, minimum=1),
        mpi_ranks=_required_int(raw, "mpi_ranks", label, minimum=1),
        comparison=_required_string(raw, "comparison", label),
        feature_family=_required_string(raw, "feature_family", label),
        vds=_required_bool(raw, "vds", label),
        tier=_required_string(raw, "tier", label),
        status=_required_string(raw, "status", label),
        case_ids=tuple(case_ids),
        reason=reason,
    )


def _parse_production_run(raw: object, index: int) -> ProductionRun:
    if not isinstance(raw, dict):
        raise AssertionError(f"runs[{index}] must be an object")
    label = f"runs[{index}]"
    passed = raw.get("passed")
    mutations = raw.get("comparator_mutations_rejected")
    if not isinstance(passed, bool) or not isinstance(mutations, bool):
        raise AssertionError(f"{label} boolean fields are invalid")
    _required_commit(raw, "source_commit", label)
    if raw.get("source_tree_state") != "clean":
        raise AssertionError(f"{label} source_tree_state must be clean")
    for key in (
        "contract_evidence_sha256",
        "matrix_evidence_sha256",
        "comparator_evidence_sha256",
    ):
        _required_sha256(raw, key, label)
    scope_exclusions = raw.get("scope_exclusions")
    if (
        not isinstance(scope_exclusions, list)
        or not all(
            isinstance(contract_id, str) and contract_id
            for contract_id in scope_exclusions
        )
        or len(scope_exclusions) != len(set(scope_exclusions))
        or VDS_CROSS_PROCESS_RESUME_CONTRACT not in scope_exclusions
    ):
        raise AssertionError(
            f"{label} scope_exclusions must uniquely include "
            f"{VDS_CROSS_PROCESS_RESUME_CONTRACT}"
        )
    scope_statement = _required_string(raw, "scope_statement", label)
    if scope_statement != SCOPED_EQUIVALENCE_STATEMENT:
        raise AssertionError(f"{label} scope_statement is invalid")
    return ProductionRun(
        run_id=_required_string(raw, "run_id", label),
        passed=passed,
        retry_count=_required_int(raw, "retry_count", label, minimum=0),
        runtime_ratio=_required_number(
            raw, "runtime_ratio", label, minimum=0.0
        ),
        finalize_fraction=_required_number(
            raw, "finalize_fraction", label, minimum=0.0
        ),
        output_bytes_ratio=_required_number(
            raw, "output_bytes_ratio", label, minimum=0.0
        ),
        comparator_mutations_rejected=mutations,
        scope_exclusions=tuple(scope_exclusions),
        scope_statement=scope_statement,
    )


def _case_proves_scenario(
    case_payload: object, scenario: ExecutionScenario
) -> bool:
    if not isinstance(case_payload, dict):
        return False
    metadata = case_payload.get("metadata")
    records = case_payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        return False
    if not records or any(
        not isinstance(record, dict) or record.get("status") != "passed"
        for record in records
    ):
        return False
    expected = {
        "ensemble": scenario.ensemble,
        "thermostat": scenario.thermostat,
        "box_geometry": scenario.box_geometry,
        "constraint": scenario.constraint,
        "backend": scenario.backend,
        "comparison": scenario.comparison,
        "feature_family": scenario.feature_family,
        "vds": scenario.vds,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        return False
    try:
        omp_threads = int(metadata.get("omp_num_threads"))
        mpi_ranks = int(metadata.get("mpi_rank_count"))
    except (TypeError, ValueError):
        return False
    gpu_metadata_valid = scenario.backend != "gpu" or all(
        isinstance(metadata.get(name), str) and metadata.get(name)
        for name in (
            "gpu_device_map",
            "gpu_model",
            "cuda_driver_version",
            "cuda_runtime_version",
        )
    )
    statistical_metadata_valid = scenario.comparison != "statistical" or (
        isinstance(metadata.get("sample_plan"), dict)
        and metadata["sample_plan"].get("profile") == "production"
        and metadata["sample_plan"].get("fast_mode") is False
        and metadata["sample_plan"].get("inference_unit") == "replica"
        and isinstance(metadata["sample_plan"].get("replica_count"), int)
        and metadata["sample_plan"]["replica_count"]
        >= MINIMUM_PRODUCTION_STATISTICAL_REPLICAS
        and isinstance(
            metadata["sample_plan"].get("paired_block_count_lower_bound"),
            int,
        )
        and metadata["sample_plan"]["paired_block_count_lower_bound"]
        >= MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS
    )
    return (
        omp_threads == scenario.omp_threads
        and mpi_ranks == scenario.mpi_ranks
        and metadata.get("rank0_output_owner") is True
        and gpu_metadata_valid
        and statistical_metadata_valid
    )


def _passed_contract_levels(
    report_cases: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    levels: dict[str, list[str]] = {}
    for case_payload in report_cases.values():
        if not isinstance(case_payload, dict):
            continue
        records = case_payload.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or record.get("status") != "passed":
                continue
            contract_id = record.get("contract_id")
            level = record.get("evidence_level")
            if isinstance(contract_id, str) and isinstance(level, str):
                levels.setdefault(contract_id, []).append(level)
    return {name: tuple(values) for name, values in levels.items()}


def _evidence_satisfies(actual: str, required: str) -> bool:
    if actual == "F1" or required == "F1":
        return actual == required
    return (
        actual in EVIDENCE_RANK
        and required in EVIDENCE_RANK
        and (EVIDENCE_RANK[actual] >= EVIDENCE_RANK[required])
    )


def _append_budget_blockers(
    blockers: list[str], budgets: PerformanceBudgets, run: ProductionRun
) -> None:
    comparisons = (
        ("runtime_ratio", run.runtime_ratio, budgets.maximum_runtime_ratio),
        (
            "finalize_fraction",
            run.finalize_fraction,
            budgets.maximum_finalize_fraction,
        ),
        (
            "output_bytes_ratio",
            run.output_bytes_ratio,
            budgets.maximum_output_bytes_ratio,
        ),
    )
    for name, actual, maximum in comparisons:
        if actual > maximum:
            blockers.append(
                f"production run {run.run_id} exceeds {name} budget: "
                f"{actual}>{maximum}"
            )


def _validate_budgets(budgets: PerformanceBudgets) -> None:
    values = {
        "maximum_runtime_ratio": budgets.maximum_runtime_ratio,
        "maximum_finalize_fraction": budgets.maximum_finalize_fraction,
        "maximum_output_bytes_ratio": budgets.maximum_output_bytes_ratio,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise AssertionError(
                f"performance budget {name} must be finite and positive"
            )


def _scenario_matches(
    scenario: ExecutionScenario, combination: Mapping[str, object]
) -> bool:
    values = scenario.axis_values()
    return all(values[name] == value for name, value in combination.items())


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} root must be an object")
    return payload


def _required_object(
    payload: Mapping[str, object], key: str, label: str
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise AssertionError(f"{label} requires object {key}")
    return dict(value)


def _required_string(
    payload: Mapping[str, object], key: str, label: str
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{label} requires non-empty string {key}")
    return value


def _required_bool(payload: Mapping[str, object], key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise AssertionError(f"{label} requires boolean {key}")
    return value


def _required_commit(
    payload: Mapping[str, object], key: str, label: str
) -> str:
    value = _required_string(payload, key, label)
    if re.fullmatch(r"[0-9a-f]{7,64}", value) is None:
        raise AssertionError(f"{label} has invalid commit ID {key}")
    return value


def _required_sha256(
    payload: Mapping[str, object], key: str, label: str
) -> str:
    value = _required_string(payload, key, label)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AssertionError(f"{label} has invalid SHA-256 {key}")
    return value


def _required_int(
    payload: Mapping[str, object], key: str, label: str, *, minimum: int
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AssertionError(f"{label} requires integer {key} >= {minimum}")
    return value


def _required_number(
    payload: Mapping[str, object],
    key: str,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AssertionError(f"{label} requires numeric {key}")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise AssertionError(f"{label} has invalid numeric {key}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate bundled I/O A/B matrix"
    )
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--contracts", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--matrix-evidence", type=Path)
    parser.add_argument("--history", type=Path)
    args = parser.parse_args(argv)
    matrix = load_execution_matrix(args.matrix)
    if args.evaluate:
        if args.evidence is None or args.history is None:
            parser.error("--evaluate requires --evidence and --history")
        evidence = _load_json_object(args.evidence, "A/B evidence report")
        matrix_evidence = (
            _load_json_object(
                args.matrix_evidence, "A/B matrix evidence report"
            )
            if args.matrix_evidence is not None
            else None
        )
        runs = load_production_run_history(args.history)
        decision = evaluate_promotion_readiness(
            matrix,
            load_contract_registry(args.contracts),
            evidence,
            runs,
            matrix_evidence,
        )
        print(
            json.dumps(
                {"ready": decision.ready, "blockers": decision.blockers},
                sort_keys=True,
            )
        )
        return 0 if decision.ready else 1
    print(
        json.dumps(
            {
                "promotion_state": matrix.promotion_state,
                "scenario_count": len(matrix.scenarios),
                "executable_scenario_count": sum(
                    scenario.status in {"executable", "evidenced"}
                    for scenario in matrix.scenarios
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
