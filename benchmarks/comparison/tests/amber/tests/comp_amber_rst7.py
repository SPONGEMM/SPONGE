"""AMBER restart time and velocity blocks are independent."""

import os
import subprocess

import pytest
from comp_amber_nb14 import (
    _extract_mdout_term,
    _write_mdin,
    _write_prmtop,
    _write_rst7,
)


def _run_restart(
    tmp_path, *, atom_count=4, time=None, velocities=False, corrupt=None
):
    prmtop = tmp_path / "system.parm7"
    rst7 = tmp_path / "system.rst7"
    mdin = tmp_path / "mdin.spg.toml"
    mdout = tmp_path / "mdout.txt"
    coordinates = [(float(i), 0.0, 0.0) for i in range(atom_count)]
    _write_prmtop(
        prmtop,
        atom_types=[1] * atom_count,
        nonbonded_parm_index=[1],
        normal_a=[0.0],
        normal_b=[0.0],
    )
    _write_rst7(rst7, coordinates)
    lines = rst7.read_text().splitlines()
    if time is not None:
        lines[1] += f" {time}"
    if velocities:
        values = [
            v for i in range(atom_count) for v in (0.1 * (-1) ** i, 0.0, 0.0)
        ]
        block = [
            "".join(f"{v:12.7f}" for v in values[i : i + 6])
            for i in range(0, len(values), 6)
        ]
        lines[-2:-2] = block
    if corrupt == "truncated":
        lines.pop()
    elif corrupt == "extra":
        lines.append("1.0")
    elif corrupt == "text":
        lines.append("not-a-number")
    elif corrupt == "header":
        lines = [lines[0]]
    elif corrupt == "atom_count":
        lines[1] = "0"
    rst7.write_text("\n".join(lines) + "\n")
    _write_mdin(mdin, prmtop, rst7, mdout)
    result = subprocess.run(
        [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", str(mdin)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result, mdout


@pytest.mark.parametrize("atom_count", [3, 4])
@pytest.mark.parametrize("time", [None, 0.0, 12.5])
@pytest.mark.parametrize("velocities", [False, True])
def test_restart_velocity_presence_is_independent_of_time(
    tmp_path, atom_count, time, velocities
):
    result, mdout = _run_restart(
        tmp_path, atom_count=atom_count, time=time, velocities=velocities
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    temperature = _extract_mdout_term(mdout, "temperature")
    assert temperature > 0.0 if velocities else temperature == 0.0
    assert _extract_mdout_term(mdout, "time") == pytest.approx(time or 0.0)


@pytest.mark.parametrize(
    "corrupt", ["truncated", "extra", "text", "header", "atom_count"]
)
def test_restart_rejects_incomplete_or_extra_data(tmp_path, corrupt):
    result, _ = _run_restart(tmp_path, corrupt=corrupt)
    assert result.returncode != 0
    assert "spongeErrorBadFileFormat" in result.stdout + result.stderr
