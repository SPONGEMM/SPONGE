import pytest


def pytest_addoption(parser):
    group = parser.getgroup("reaxff")
    group.addoption(
        "--steps",
        action="store",
        type=int,
        default=100,
        help="Step limit for the ReaxFF throughput performance benchmark.",
    )


@pytest.fixture(scope="session")
def reaxff_steps(request):
    value = int(request.config.getoption("--steps"))
    if value <= 0:
        raise ValueError("--steps must be positive")
    return value
