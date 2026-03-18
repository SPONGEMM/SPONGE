import os
import re
import subprocess
import sys
from pathlib import Path

from benchmarks.utils import Extractor, Outputer
from benchmarks.utils import Runner


def _prips_plugin_path():
    result = subprocess.run(
        [sys.executable, "-m", "prips"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to resolve prips plugin path from `python -m prips`\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    output = result.stdout + "\n" + result.stderr
    match = re.search(r"Plugin Path:\s*(.+)", output)
    if match is None:
        raise Exception("failed to resolve Plugin Path from `python -m prips`")

    plugin_path = Path(match.group(1).strip())
    if not plugin_path.exists():
        raise Exception(f"prips plugin does not exist: {plugin_path}")
    return plugin_path


def _write_prips_script(case_dir, backend):
    script = (
        "from prips import Sponge\n"
        "\n"
        f"_requested_backend = {backend!r}\n"
        "if _requested_backend == 'auto':\n"
        "    _requested_backend = 'numpy' if Sponge._backend == 1 else 'pytorch'\n"
        "Sponge.set_backend(_requested_backend)\n"
        "_force_delta = 1.25\n"
        "\n"
        "with open('prips_hook.log', 'w', encoding='utf-8') as f:\n"
        "    f.write('[prips]\\n')\n"
        "    f.write(f'backend device={Sponge._backend} name={Sponge.backend_name}\\n')\n"
        "\n"
        "def After_Initial():\n"
        "    crd = Sponge.md_info.crd\n"
        "    with open('prips_hook.log', 'a', encoding='utf-8') as f:\n"
        "        f.write(\n"
        "            f'after_init atom_numbers={Sponge.md_info.atom_numbers} '\n"
        "            f'neighbor_max={Sponge.neighbor_list.max_neighbor_numbers} '\n"
        "            f'rank={Sponge.controller.MPI_rank} '\n"
        "            f'coord[0,0]={float(crd[0, 0]):.6f}\\n'\n"
        "        )\n"
        "\n"
        "def Calculate_Force():\n"
        "    frc = Sponge.dd.frc\n"
        "    step = Sponge.md_info.sys.steps\n"
        "    before = float(frc[0, 0])\n"
        "    if step == 1:\n"
        "        if Sponge.backend_name != 'jax':\n"
        "            frc[0, 0] += _force_delta\n"
        "    after = float(frc[0, 0])\n"
        "    with open('prips_hook.log', 'a', encoding='utf-8') as f:\n"
        "        f.write(\n"
        "            f'calculate_force step={step} '\n"
        "            f'local_atom_numbers={Sponge.dd.atom_numbers} '\n"
        "            f'force[0,0]_before={before:.6f} '\n"
        "            f'force[0,0]_after={after:.6f} '\n"
        "            f'force[0,0]_delta={after - before:.6f}\\n'\n"
        "        )\n"
        "\n"
        "def Mdout_Print():\n"
        "    frc = Sponge.dd.frc\n"
        "    with open('prips_hook.log', 'a', encoding='utf-8') as f:\n"
        "        f.write(\n"
        "            f'mdout_force step={Sponge.md_info.sys.steps} '\n"
        "            f'force[0,0]_final={float(frc[0, 0]):.6f}\\n'\n"
        "        )\n"
    )
    (Path(case_dir) / "prips_test.py").write_text(script, encoding="utf-8")


def _write_prips_mdin(case_dir, plugin_path=None, *, step_limit=1):
    mdin = (
        'md_name = "validation tip3p prips"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        "dt = 0.0\n"
        "cutoff = 8.0\n"
        'default_in_file_prefix = "tip3p"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n"
        'frc = "frc.dat"\n'
        'thermostat = "middle_langevin"\n'
        "thermostat_tau = 0.01\n"
        "thermostat_seed = 2026\n"
        "target_temperature = 300.0\n"
        "hard_wall_z_low = 5.0\n"
        "hard_wall_z_high = 30.0\n"
    )
    if plugin_path is not None:
        mdin += f'plugin = "{Path(plugin_path).as_posix()}"\n'
        mdin += 'py = "prips_test.py"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin, encoding="utf-8")


def test_tip3p_prips_plugin_hooks_run(statics_path, outputs_path, mpi_np):
    backend = os.environ.get("PRIPS_TEST_BACKEND", "auto")
    assert backend in {"auto", "numpy", "jax", "cupy", "pytorch"}
    expected_backend = backend
    if expected_backend == "auto":
        expected_backend = "pytorch"
    plugin_path = _prips_plugin_path()
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p",
        mpi_np=mpi_np,
        run_name="tip3p_prips",
    )
    _write_prips_script(case_dir, backend)
    _write_prips_mdin(case_dir, plugin_path)
    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    hook_log = case_dir / "prips_hook.log"
    assert hook_log.exists()
    hook_lines = hook_log.read_text(encoding="utf-8").splitlines()

    backend_line = next(
        (line for line in hook_lines if line.startswith("backend ")),
        None,
    )
    after_init_line = next(
        (line for line in hook_lines if line.startswith("after_init ")),
        None,
    )
    force_lines = [
        line for line in hook_lines if line.startswith("calculate_force ")
    ]
    final_force_line = next(
        (
            line
            for line in reversed(hook_lines)
            if line.startswith("mdout_force ")
        ),
        None,
    )
    sponge_forces = Extractor.extract_sponge_forces(case_dir, 1011)

    assert backend_line is not None
    assert f"name={expected_backend}" in backend_line
    assert any(
        backend_line.startswith(f"backend device={value} ")
        for value in (1, 2, 10)
    )
    assert after_init_line is not None
    assert "atom_numbers=1011" in after_init_line
    assert "neighbor_max=1200" in after_init_line
    assert "coord[0,0]=" in after_init_line
    assert len(force_lines) == 2
    assert final_force_line is not None
    assert all("local_atom_numbers=1011" in line for line in force_lines)
    force_matches = [
        re.search(
            r"calculate_force\s+step=(\d+)\s+local_atom_numbers=\d+\s+"
            r"force\[0,0\]_before=([-\d.]+)\s+"
            r"force\[0,0\]_after=([-\d.]+)\s+"
            r"force\[0,0\]_delta=([-\d.]+)",
            line,
        )
        for line in force_lines
    ]
    assert all(match is not None for match in force_matches)
    force_records = {
        int(match.group(1)): (
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )
        for match in force_matches
    }
    assert set(force_records) == {0, 1}
    step0_before, step0_after, step0_delta = force_records[0]
    step1_before, step1_after, step1_delta = force_records[1]

    Outputer.print_table(
        ["Metric", "Value"],
        [
            ["Case", "tip3p_prips"],
            ["PluginPath", str(plugin_path)],
            [
                "Backend",
                backend_line.split("=", 1)[1] if backend_line else "N/A",
            ],
            ["AfterInitial", "PASS" if after_init_line else "MISSING"],
            ["CalculateForce", "PASS" if force_lines else "MISSING"],
            ["MdoutForce", "PASS" if final_force_line else "MISSING"],
            ["F[0,0,0]", f"{step0_after:.6f}"],
            ["F[1,0,0]", f"{step1_after:.6f}"],
            ["ΔF[0,0]", f"{step1_after - step0_after:.6f}"],
        ],
        title="Misc Validation: TIP3P PRIPS",
    )

    assert abs(step0_after - step0_before) < 1e-5
    assert abs(step0_delta) < 1e-5
    if expected_backend == "jax":
        assert abs(step1_after - step1_before) < 1e-5
        assert abs(step1_delta) < 1e-5
    else:
        assert abs(step1_after - step1_before - 1.25) < 1e-5
        assert abs(step1_delta - 1.25) < 1e-5
        assert abs(step1_delta - step0_delta - 1.25) < 1e-5
    final_force_match = re.search(
        r"mdout_force\s+step=\d+\s+force\[0,0\]_final=([-\d.]+)",
        final_force_line,
    )
    assert final_force_match is not None
    final_force = float(final_force_match.group(1))
    if expected_backend != "jax":
        assert abs(sponge_forces[0, 0] - final_force) < 1e-5
