import pytest


def pytest_addoption(parser):
    group = parser.getgroup("sits")
    group.addoption(
        "--iter-steps",
        action="store",
        type=int,
        default=1000000,
        help="Iteration step_limit for SITS benchmark test.",
    )
    group.addoption(
        "--prod-steps",
        action="store",
        type=int,
        default=20000000,
        help="Production step_limit for SITS benchmark test.",
    )


@pytest.fixture(scope="session")
def sits_iter_steps(request):
    value = int(request.config.getoption("--iter-steps"))
    if value <= 0:
        raise ValueError("--iter-steps must be positive")
    return value


@pytest.fixture(scope="session")
def sits_prod_steps(request):
    value = int(request.config.getoption("--prod-steps"))
    if value <= 0:
        raise ValueError("--prod-steps must be positive")
    return value
