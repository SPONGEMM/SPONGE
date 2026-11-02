import json
import os
import subprocess
import textwrap

import pytest


def _toml_string(value):
    return json.dumps(os.fspath(value))


def _gro_atom(resid, resname, atomname, atomnr, x):
    x, y, z = x if isinstance(x, tuple) else (x, 0.0, 0.0)
    return (
        f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def _minimal_topology(injected="", system_text="section integrity test"):
    return f"""
        [ defaults ]
        1 2 yes 1.0 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.30 0.4184

        {injected}

        [ moleculetype ]
        ONE 0

        [ atoms ]
        1 A 1 ONE A 1 0.0 12.0

        [ system ]
        {system_text}

        [ molecules ]
        ONE 1
    """


def _run_topology(tmp_path, topology, atoms=(("A", 0.0),)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    top_path = tmp_path / "topol.top"
    gro_path = tmp_path / "conf.gro"
    mdin_path = tmp_path / "mdin.spg.toml"
    mdout_path = tmp_path / "mdout.txt"

    top_path.write_text(textwrap.dedent(topology).strip() + "\n")
    atom_lines = [
        _gro_atom(1, "ONE", atom_name, index + 1, x)
        for index, (atom_name, x) in enumerate(atoms)
    ]
    gro_path.write_text(
        "\n".join(
            [
                "section integrity test",
                str(len(atom_lines)),
                *atom_lines,
                "   5.00000   5.00000   5.00000",
            ]
        )
        + "\n"
    )
    mdin_path.write_text(
        textwrap.dedent(
            f"""
            md_name = "direct_gromacs_section_integrity"
            mode = "nve"
            step_limit = 0
            dt = 0
            cutoff = 8.0
            gromacs_top = {_toml_string(top_path)}
            gromacs_gro = {_toml_string(gro_path)}
            mdout = {_toml_string(mdout_path)}
            print_zeroth_frame = 1
            write_mdout_interval = 1
            """
        ).strip()
        + "\n"
    )

    result = subprocess.run(
        [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", str(mdin_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return result, mdout_path


def _extract_mdout_term(mdout_path, term_name):
    lines = mdout_path.read_text().splitlines()
    headers = lines[0].split()
    values = lines[1].split()
    assert len(headers) == len(values)
    return float(dict(zip(headers, values))[term_name])


def test_active_unsupported_section_in_include_reports_source_line(tmp_path):
    include_path = tmp_path / "unsupported.itp"
    include_path.write_text("; metadata\n\n[ virtual_sites2 \\\n]\n1 1 1 0.5\n")

    result, _ = _run_topology(
        tmp_path,
        _minimal_topology('#include "unsupported.itp"'),
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode != 0
    assert "spongeErrorBadFileFormat" in output
    assert "unsupported GROMACS topology section [ virtual_sites2 ]" in output
    assert f"{include_path}:3" in output


def test_unsupported_section_in_inactive_branch_is_not_parsed(tmp_path):
    result, _ = _run_topology(
        tmp_path,
        _minimal_topology(
            """
            #ifdef NEVER_DEFINED
            [ virtual_sites2 ]
            #endif
            """
        ),
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert "unsupported GROMACS topology section" not in output


def test_system_metadata_section_is_explicitly_allowed(tmp_path):
    result, _ = _run_topology(
        tmp_path,
        _minimal_topology(system_text="arbitrary human-readable system name"),
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_explicit_exclusions_are_normalized_and_merged_with_nrexcl(tmp_path):
    topology = """
        [ defaults ]
        1 2 yes 1.0 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.30 4.184

        [ moleculetype ]
        TRIPLE 1

        [ atoms ]
        1 A 1 TRIPLE A1 1 0.0 12.0
        2 A 1 TRIPLE A2 2 0.0 12.0
        3 A 1 TRIPLE A3 3 0.0 12.0

        [ bonds ]
        1 2 1 0.35 0.0

        {exclusions}

        [ system ]
        explicit exclusions

        [ molecules ]
        TRIPLE 1
    """
    atoms = (("A1", 0.0), ("A2", 0.35), ("A3", 0.7))
    without_result, without_mdout = _run_topology(
        tmp_path / "without",
        topology.format(exclusions=""),
        atoms=atoms,
    )
    with_result, with_mdout = _run_topology(
        tmp_path / "with",
        topology.format(
            exclusions="""
            [ exclusions ]
            2 3 3
            3 2
            """
        ),
        atoms=atoms,
    )

    assert without_result.returncode == 0, (
        without_result.stdout + "\n" + without_result.stderr
    )
    assert with_result.returncode == 0, (
        with_result.stdout + "\n" + with_result.stderr
    )
    expected_at_0_35_nm = 4.0 * ((0.30 / 0.35) ** 12 - (0.30 / 0.35) ** 6)
    expected_at_0_7_nm = 4.0 * ((0.30 / 0.7) ** 12 - (0.30 / 0.7) ** 6)
    assert _extract_mdout_term(without_mdout, "LJ_short") == pytest.approx(
        expected_at_0_35_nm + expected_at_0_7_nm, abs=0.011
    )
    assert _extract_mdout_term(with_mdout, "LJ_short") == pytest.approx(
        expected_at_0_7_nm, abs=0.011
    )


def test_explicit_exclusions_override_pairs_14_interaction(tmp_path):
    topology = """
        [ defaults ]
        1 2 yes 0.25 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.20 0.4184
        B B 12.0 0.0 A 0.20 0.4184

        [ nonbond_params ]
        A B 1 0.40 4.184

        [ moleculetype ]
        PAIR 1

        [ atoms ]
        1 A 1 PAIR A 1 0.0 12.0
        2 B 1 PAIR B 2 0.0 12.0

        [ bonds ]
        1 2 1 0.50 0.0

        [ pairs ]
        1 2 1

        {exclusions}

        [ system ]
        explicit exclusions override pairs

        [ molecules ]
        PAIR 1
    """
    atoms = (("A", 0.0), ("B", 0.5))
    without_result, without_mdout = _run_topology(
        tmp_path / "without",
        topology.format(exclusions=""),
        atoms=atoms,
    )
    with_result, with_mdout = _run_topology(
        tmp_path / "with",
        topology.format(
            exclusions="""
            [ exclusions ]
            1 2
            """
        ),
        atoms=atoms,
    )

    assert without_result.returncode == 0, (
        without_result.stdout + "\n" + without_result.stderr
    )
    sigma_over_r = 0.40 / 0.5
    expected_pair_14 = 0.25 * 4.0 * (sigma_over_r**12 - sigma_over_r**6)
    assert _extract_mdout_term(without_mdout, "nb14_LJ") == pytest.approx(
        expected_pair_14, abs=0.011
    )
    assert with_result.returncode == 0, (
        with_result.stdout + "\n" + with_result.stderr
    )
    # The nb14 columns disappear entirely once every pair is overridden.
    with_lines = with_mdout.read_text().splitlines()
    with_terms = dict(zip(with_lines[0].split(), with_lines[1].split()))
    assert float(with_terms.get("nb14_LJ", 0.0)) == pytest.approx(
        0.0, abs=0.011
    )
    output = with_result.stdout + "\n" + with_result.stderr
    assert "overridden by explicit [ exclusions ]" in output
