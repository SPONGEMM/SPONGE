from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("mace")
    group.addoption(
        "--steps",
        action="store",
        type=int,
        default=5000,
        help="Step limit for the long perf-mace dynamics benchmark.",
    )


@pytest.fixture(scope="module")
def statics_path():
    return Path(__file__).resolve().parents[1] / "statics"


@pytest.fixture(scope="session")
def mace_steps(request):
    value = int(request.config.getoption("--steps"))
    if value <= 0:
        raise ValueError("--steps must be positive")
    return value
