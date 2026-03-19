import pytest


def pytest_addoption(parser):
    group = parser.getgroup("sinkmeta")
    group.addoption(
        "--steps",
        action="store",
        type=int,
        default=50000,
        help="NVT step_limit for the sinkmeta performance benchmark.",
    )


@pytest.fixture(scope="session")
def sinkmeta_steps(request):
    value = int(request.config.getoption("--steps"))
    if value <= 0:
        raise ValueError("--steps must be positive")
    return value
