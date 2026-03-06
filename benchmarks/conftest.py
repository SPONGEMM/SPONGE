from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mpi",
        action="store",
        default=None,
        metavar="N",
        type=int,
        help=(
            "Launch SPONGE with `mpirun -np N SPONGE`. "
            "Omit this option to run SPONGE directly."
        ),
    )


def _validate_mpi_np(value):
    if value is not None and value < 1:
        raise pytest.UsageError("--mpi must be a positive integer")


def pytest_configure(config):
    _validate_mpi_np(config.getoption("mpi"))


def _request_path(request):
    path = getattr(request, "path", None)
    if path is not None:
        return Path(path).resolve()
    return Path(str(request.node.fspath)).resolve()


def _suite_root_from_request(request):
    return _request_path(request).parent.parent


@pytest.fixture(scope="module")
def statics_path(request):
    return _suite_root_from_request(request) / "statics"


@pytest.fixture(scope="module")
def outputs_path(request):
    path = _suite_root_from_request(request) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def mpi_np(pytestconfig):
    value = pytestconfig.getoption("mpi")
    _validate_mpi_np(value)
    return value
