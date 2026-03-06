import os
import shlex


def build_sponge_command(
    base_cmd, *, mpi_np=None, mpirun_env_var="SPONGE_MPIRUN"
):
    cmd = list(base_cmd)
    if mpi_np is None:
        return cmd

    mpirun_value = (
        os.environ.get(mpirun_env_var)
        or os.environ.get("SPONGE_MPIRUN")
        or "mpirun --oversubscribe"
    )
    return shlex.split(mpirun_value) + ["-np", str(mpi_np)] + cmd
