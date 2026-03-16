from benchmarks.utils import Runner


def run_sponge_barostat(case_dir, timeout=2400, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        mpi_np=mpi_np,
        timeout=timeout,
    )
