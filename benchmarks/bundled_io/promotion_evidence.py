from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from benchmarks.bundled_io.ab_contracts import (
    SCOPED_EQUIVALENCE_STATEMENT,
    ContractSpec,
    load_contract_registry,
    required_scope_exclusions,
    validate_complete_evidence_report,
)
from benchmarks.bundled_io.execution_matrix import (
    MINIMUM_PRODUCTION_STATISTICAL_PAIRED_BLOCKS,
    MINIMUM_PRODUCTION_STATISTICAL_REPLICAS,
    ExecutionMatrix,
    ExecutionScenario,
    ProductionRun,
    evaluate_promotion_readiness,
    load_execution_matrix,
)

REQUIRED_COMPARATOR_MUTATION_TESTS = (
    "test_energy_offset_mutation_is_rejected",
    "test_variance_increase_mutation_is_rejected",
    "test_atom_permutation_mutation_is_rejected",
    "test_local_force_offset_mutation_is_rejected",
    "test_missing_nonfinite_mutation_is_rejected",
    "test_nonfinite_kind_and_sign_mutations_are_rejected",
    "test_frame_schedule_mutation_is_rejected",
    "test_chunk_boundary_layout_rejects_frame_and_shard_mutations",
)
REQUIRED_COMPARATOR_MUTATION_NODE_COUNTS = {
    test_name: (
        3
        if test_name == "test_nonfinite_kind_and_sign_mutations_are_rejected"
        else 1
    )
    for test_name in REQUIRED_COMPARATOR_MUTATION_TESTS
}


def write_comparator_mutation_report(
    path: Path,
    run_id: str,
    outcomes: Mapping[str, Sequence[tuple[str, str]]],
    pytest_exit_status: int,
) -> dict[str, object]:
    if not run_id:
        raise AssertionError("comparator evidence requires a non-empty run_id")
    tests = {}
    for test_name in REQUIRED_COMPARATOR_MUTATION_TESTS:
        records = tuple(outcomes.get(test_name, ()))
        statuses = [status for _, status in records]
        expected_count = REQUIRED_COMPARATOR_MUTATION_NODE_COUNTS[test_name]
        if len(statuses) != expected_count:
            status = "missing"
        elif all(item == "passed" for item in statuses):
            status = "passed"
        else:
            status = "failed"
        tests[test_name] = {
            "status": status,
            "expected_node_count": expected_count,
            "nodeids": sorted(nodeid for nodeid, _ in records),
        }
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "pytest_exit_status": pytest_exit_status,
        "required_tests": list(REQUIRED_COMPARATOR_MUTATION_TESTS),
        "tests": tests,
    }
    _write_json_atomically(path, report)
    return report


def merge_matrix_evidence(
    paths: Sequence[Path], run_id: str, source_commit: str | None = None
) -> dict[str, object]:
    if not paths:
        raise AssertionError("at least one matrix evidence report is required")
    merged: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "cases": {},
    }
    merged_cases = merged["cases"]
    assert isinstance(merged_cases, dict)
    observed_source_commit = None
    for path in paths:
        report = _load_json_object(path, "matrix evidence")
        if report.get("schema_version") not in {1, "1"}:
            raise AssertionError(f"invalid matrix evidence schema: {path}")
        if report.get("run_id") != run_id:
            raise AssertionError(
                "matrix evidence run_id mismatch: "
                f"expected={run_id!r}, actual={report.get('run_id')!r}, "
                f"path={path}"
            )
        report_source_commit = report.get("source_commit")
        if (
            not isinstance(report_source_commit, str)
            or re.fullmatch(r"[0-9a-f]{7,64}", report_source_commit) is None
        ):
            raise AssertionError(
                f"matrix evidence has invalid source commit: {path}"
            )
        if source_commit is not None and report_source_commit != source_commit:
            raise AssertionError(
                "matrix evidence source commit mismatch: "
                f"expected={source_commit}, actual={report_source_commit}, "
                f"path={path}"
            )
        if (
            observed_source_commit is not None
            and report_source_commit != observed_source_commit
        ):
            raise AssertionError(
                f"matrix evidence source commits differ: {path}"
            )
        if report.get("source_tree_state") != "clean":
            raise AssertionError(
                f"matrix evidence source tree is not clean: {path}"
            )
        observed_source_commit = report_source_commit
        cases = report.get("cases")
        if not isinstance(cases, dict):
            raise AssertionError(
                f"matrix evidence cases must be an object: {path}"
            )
        for case_id, payload in cases.items():
            existing = merged_cases.get(case_id)
            if existing is not None and existing != payload:
                raise AssertionError(
                    f"conflicting matrix evidence for case {case_id}: {path}"
                )
            merged_cases[case_id] = payload
    merged["source_commit"] = observed_source_commit
    merged["source_tree_state"] = "clean"
    return merged


def derive_production_run(
    *,
    run_id: str,
    source_commit: str,
    source_tree_state: str,
    retry_count: int,
    evidence_path: Path,
    matrix_evidence_paths: Sequence[Path],
    comparator_evidence_path: Path,
    matrix: ExecutionMatrix,
    contracts: Mapping[str, ContractSpec],
) -> tuple[ProductionRun, dict[str, str]]:
    if not run_id:
        raise AssertionError("production run requires a non-empty run_id")
    if re.fullmatch(r"[0-9a-f]{7,64}", source_commit) is None:
        raise AssertionError(
            "source_commit must be a 7-64 character lowercase hex ID"
        )
    if source_tree_state != "clean":
        raise AssertionError("production run source_tree_state must be clean")
    if retry_count < 0:
        raise AssertionError("retry_count must be non-negative")

    evidence = validate_complete_evidence_report(
        evidence_path, contracts, run_id
    )
    matrix_evidence = merge_matrix_evidence(
        matrix_evidence_paths, run_id, source_commit
    )
    comparator_evidence = _load_json_object(
        comparator_evidence_path, "comparator mutation evidence"
    )
    _validate_comparator_evidence(comparator_evidence, run_id)
    _validate_behavior_evidence(matrix, contracts, evidence, matrix_evidence)
    performance = _derive_performance(matrix, matrix_evidence)

    run = ProductionRun(
        run_id=run_id,
        passed=True,
        retry_count=retry_count,
        runtime_ratio=performance["runtime_ratio"],
        finalize_fraction=performance["finalize_fraction"],
        output_bytes_ratio=performance["output_bytes_ratio"],
        comparator_mutations_rejected=True,
        scope_exclusions=required_scope_exclusions(contracts),
        scope_statement=SCOPED_EQUIVALENCE_STATEMENT,
    )
    provenance = {
        "source_commit": source_commit,
        "source_tree_state": source_tree_state,
        "contract_evidence_sha256": _file_sha256(evidence_path),
        "matrix_evidence_sha256": _payload_sha256(matrix_evidence),
        "comparator_evidence_sha256": _file_sha256(comparator_evidence_path),
    }
    return run, provenance


def append_production_run_history(
    path: Path, run: ProductionRun, provenance: Mapping[str, str]
) -> dict[str, object]:
    required_provenance = {
        "source_commit",
        "source_tree_state",
        "contract_evidence_sha256",
        "matrix_evidence_sha256",
        "comparator_evidence_sha256",
    }
    if set(provenance) != required_provenance:
        raise AssertionError(
            "production run provenance fields differ: "
            f"expected={sorted(required_provenance)}, "
            f"actual={sorted(provenance)}"
        )
    if path.exists():
        history = _load_json_object(path, "production run history")
    else:
        history = {"schema_version": 1, "runs": []}
    if history.get("schema_version") != 1:
        raise AssertionError("production run history schema_version must be 1")
    runs = history.get("runs")
    if not isinstance(runs, list):
        raise AssertionError("production run history runs must be a list")
    if any(
        isinstance(item, dict) and item.get("run_id") == run.run_id
        for item in runs
    ):
        raise AssertionError(f"duplicate production run ID: {run.run_id}")
    runs.append({**asdict(run), **dict(provenance)})
    _write_json_atomically(path, history)
    return history


def inspect_clean_source_tree(repo_root: Path, expected_commit: str) -> str:
    actual_commit = _run_git(repo_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise AssertionError(
            "source commit mismatch: "
            f"expected={expected_commit!r}, actual={actual_commit!r}"
        )
    status = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        paths = [
            line[3:] if len(line) > 3 else line for line in status.splitlines()
        ]
        preview = paths[:10]
        suffix = (
            f" (+{len(paths) - len(preview)} more)" if len(paths) > 10 else ""
        )
        raise AssertionError(
            f"production source tree is dirty: {preview}{suffix}"
        )
    return "clean"


def _validate_comparator_evidence(
    report: Mapping[str, object], run_id: str
) -> None:
    if report.get("schema_version") != 1:
        raise AssertionError("comparator evidence schema_version must be 1")
    if report.get("run_id") != run_id:
        raise AssertionError(
            "comparator evidence run_id mismatch: "
            f"expected={run_id!r}, actual={report.get('run_id')!r}"
        )
    exit_status = report.get("pytest_exit_status")
    if (
        not isinstance(exit_status, int)
        or isinstance(exit_status, bool)
        or exit_status != 0
    ):
        raise AssertionError("comparator evidence pytest_exit_status must be 0")
    required = report.get("required_tests")
    if required != list(REQUIRED_COMPARATOR_MUTATION_TESTS):
        raise AssertionError("comparator evidence required_tests differ")
    tests = report.get("tests")
    if not isinstance(tests, dict):
        raise AssertionError("comparator evidence tests must be an object")
    if set(tests) != set(REQUIRED_COMPARATOR_MUTATION_TESTS):
        raise AssertionError("comparator evidence test inventory differs")
    rejected = []
    for test_name in REQUIRED_COMPARATOR_MUTATION_TESTS:
        payload = tests[test_name]
        if not isinstance(payload, dict):
            continue
        nodeids = payload.get("nodeids")
        expected_count = REQUIRED_COMPARATOR_MUTATION_NODE_COUNTS[test_name]
        if (
            payload.get("status") == "passed"
            and payload.get("expected_node_count") == expected_count
            and isinstance(nodeids, list)
            and len(nodeids) == expected_count
            and len(nodeids) == len(set(nodeids))
        ):
            rejected.append(test_name)
    if len(rejected) != len(REQUIRED_COMPARATOR_MUTATION_TESTS):
        missing = sorted(
            set(REQUIRED_COMPARATOR_MUTATION_TESTS) - set(rejected)
        )
        raise AssertionError(
            f"comparator mutations lack passing rejection evidence: {missing}"
        )


def _validate_behavior_evidence(
    matrix: ExecutionMatrix,
    contracts: Mapping[str, ContractSpec],
    evidence: Mapping[str, object],
    matrix_evidence: Mapping[str, object],
) -> None:
    validation_run = ProductionRun(
        run_id="evidence-validation",
        passed=True,
        retry_count=0,
        runtime_ratio=matrix.performance_budgets.maximum_runtime_ratio * 0.5,
        finalize_fraction=matrix.performance_budgets.maximum_finalize_fraction
        * 0.5,
        output_bytes_ratio=matrix.performance_budgets.maximum_output_bytes_ratio
        * 0.5,
        comparator_mutations_rejected=True,
        scope_exclusions=required_scope_exclusions(contracts),
        scope_statement=SCOPED_EQUIVALENCE_STATEMENT,
    )
    candidate = replace(matrix, promotion_state="candidate")
    decision = evaluate_promotion_readiness(
        candidate,
        contracts,
        evidence,
        (validation_run,) * matrix.required_consecutive_production_runs,
        matrix_evidence,
    )
    if not decision.ready:
        raise AssertionError(
            "production evidence is incomplete: " + "; ".join(decision.blockers)
        )


def _derive_performance(
    matrix: ExecutionMatrix, matrix_evidence: Mapping[str, object]
) -> dict[str, float]:
    cases = matrix_evidence.get("cases")
    assert isinstance(cases, dict)
    values = {
        "runtime_ratio": [],
        "finalize_fraction": [],
        "output_bytes_ratio": [],
    }
    for scenario in matrix.scenarios:
        matched = False
        for case_id in scenario.case_ids:
            payload = cases.get(case_id)
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            if not _metadata_proves_scenario(metadata, payload, scenario):
                continue
            assert isinstance(metadata, dict)
            performance = metadata.get("performance")
            if not isinstance(performance, dict):
                continue
            parsed = {}
            for name in values:
                value = performance.get(name)
                if not isinstance(value, (int, float)) or isinstance(
                    value, bool
                ):
                    break
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    break
                parsed[name] = number
            else:
                for name, number in parsed.items():
                    values[name].append(number)
                matched = True
                break
        if not matched:
            raise AssertionError(
                f"scenario lacks finite performance evidence: {scenario.scenario_id}"
            )
    return {name: max(samples) for name, samples in values.items()}


def _metadata_proves_scenario(
    metadata: object,
    payload: Mapping[str, object],
    scenario: ExecutionScenario,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return False
    if any(
        not isinstance(record, dict) or record.get("status") != "passed"
        for record in records
    ):
        return False
    expected = scenario.axis_values()
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


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} root must be an object")
    return payload


def _run_git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.rstrip("\n")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive a bundled I/O production run from real evidence"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--retry-count", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--matrix-evidence", type=Path, action="append", required=True
    )
    parser.add_argument("--comparator-evidence", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args(argv)

    source_tree_state = inspect_clean_source_tree(
        args.repo_root, args.source_commit
    )
    run, provenance = derive_production_run(
        run_id=args.run_id,
        source_commit=args.source_commit,
        source_tree_state=source_tree_state,
        retry_count=args.retry_count,
        evidence_path=args.evidence,
        matrix_evidence_paths=args.matrix_evidence,
        comparator_evidence_path=args.comparator_evidence,
        matrix=load_execution_matrix(),
        contracts=load_contract_registry(),
    )
    append_production_run_history(args.history, run, provenance)
    print(json.dumps({**asdict(run), **provenance}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
