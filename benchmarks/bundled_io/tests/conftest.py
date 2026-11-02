from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks.bundled_io.promotion_evidence import (
    REQUIRED_COMPARATOR_MUTATION_TESTS,
    write_comparator_mutation_report,
)

COMPARATOR_EVIDENCE_ENV = "SPONGE_BUNDLED_IO_AB_COMPARATOR_EVIDENCE"
RUN_ID_ENV = "SPONGE_BUNDLED_IO_AB_RUN_ID"
COMPARATOR_MODULE = (
    "benchmarks/bundled_io/tests/test_bundled_io_ab_statistics.py"
)
_OUTCOMES: dict[str, dict[str, str]] = {}


def pytest_configure(config):
    if os.environ.get(COMPARATOR_EVIDENCE_ENV) and not os.environ.get(
        RUN_ID_ENV
    ):
        raise pytest.UsageError(
            f"{COMPARATOR_EVIDENCE_ENV} requires {RUN_ID_ENV}"
        )


def pytest_runtest_logreport(report):
    module_name = report.nodeid.split("::", 1)[0]
    if not module_name.endswith(COMPARATOR_MODULE):
        return
    test_name = report.nodeid.rsplit("::", 1)[-1].split("[", 1)[0]
    if test_name not in REQUIRED_COMPARATOR_MUTATION_TESTS:
        return
    outcomes = _OUTCOMES.setdefault(test_name, {})
    existing = outcomes.get(report.nodeid)
    if report.failed:
        outcomes[report.nodeid] = "failed"
    elif report.skipped and existing != "failed":
        outcomes[report.nodeid] = "skipped"
    elif report.when == "call" and existing not in {"failed", "skipped"}:
        outcomes[report.nodeid] = "passed"


def pytest_sessionfinish(session, exitstatus):
    evidence_path = os.environ.get(COMPARATOR_EVIDENCE_ENV)
    if not evidence_path:
        return
    outcomes = {
        test_name: tuple(nodeids.items())
        for test_name, nodeids in _OUTCOMES.items()
    }
    write_comparator_mutation_report(
        Path(evidence_path),
        os.environ[RUN_ID_ENV],
        outcomes,
        pytest_exit_status=exitstatus,
    )
