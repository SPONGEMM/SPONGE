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


@pytest.fixture(scope="session")
def mpi_np(pytestconfig):
    value = pytestconfig.getoption("mpi")
    _validate_mpi_np(value)
    return value


@pytest.fixture(scope="session")
def mpi_run_tag(mpi_np):
    if mpi_np is None:
        return "direct"
    return f"mpi_np{mpi_np}"
