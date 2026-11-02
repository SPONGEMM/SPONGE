import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def _toml_string(value):
    return json.dumps(os.fspath(value))


def _gro_atom(resid, resname, atomname, atomnr, x):
    return (
        f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}"
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}"
    )


def _run_two_atom_topology(
    tmp_path, topology, *, distance_nm=0.5, atom_names=("A", "B")
):
    top_path = tmp_path / "topol.top"
    gro_path = tmp_path / "conf.gro"
    mdin_path = tmp_path / "mdin.spg.toml"
    mdout_path = tmp_path / "mdout.txt"

    top_path.write_text(textwrap.dedent(topology).strip() + "\n")
    gro_path.write_text(
        "\n".join(
            [
                "nonbond params test",
                "2",
                _gro_atom(1, "PAIR", atom_names[0], 1, 0.0),
                _gro_atom(1, "PAIR", atom_names[1], 2, distance_nm),
                "   5.00000   5.00000   5.00000",
            ]
        )
        + "\n"
    )
    mdin_path.write_text(
        textwrap.dedent(
            f"""
            md_name = "direct_gromacs_nonbond_params"
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
    lines = Path(mdout_path).read_text().splitlines()
    headers = lines[0].split()
    values = lines[1].split()
    assert len(headers) == len(values)
    return float(dict(zip(headers, values))[term_name])


def _lj_energy(sigma_nm, epsilon_kj, distance_nm, scale=1.0):
    sigma_over_r = abs(sigma_nm) / distance_nm
    attractive = 0.0 if sigma_nm < 0.0 else sigma_over_r**6
    return scale * 4.0 * (epsilon_kj / 4.184) * (sigma_over_r**12 - attractive)


def test_nonbond_params_override_main_lj_without_fudge_and_keep_named_types(
    tmp_path,
):
    result, mdout_path = _run_two_atom_topology(
        tmp_path,
        """
        [ defaults ]
        1 2 yes 0.25 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.30 0.4184
        B B 12.0 0.0 A 0.30 0.4184

        [ nonbond_params ]
        A B 1 0.25 0.4184
        B A 1 0.40 4.184

        [ moleculetype ]
        PAIR 0

        [ atoms ]
        1 A 1 PAIR A 1 0.0 12.0
        2 B 1 PAIR B 2 0.0 12.0

        [ system ]
        main nonbond override

        [ molecules ]
        PAIR 1
        """,
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert "atom_LJ_type_number is 2" in output
    assert _extract_mdout_term(mdout_path, "LJ_short") == pytest.approx(
        _lj_energy(0.40, 4.184, 0.5), abs=0.011
    )


def test_nonbond_params_negative_sigma_is_purely_repulsive(tmp_path):
    result, mdout_path = _run_two_atom_topology(
        tmp_path,
        """
        [ defaults ]
        1 2 yes 1.0 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.30 0.4184
        B B 12.0 0.0 A 0.30 0.4184

        [ nonbond_params ]
        A B 1 -0.40 4.184

        [ moleculetype ]
        PAIR 0

        [ atoms ]
        1 A 1 PAIR A 1 0.0 12.0
        2 B 1 PAIR B 2 0.0 12.0

        [ system ]
        negative sigma override

        [ molecules ]
        PAIR 1
        """,
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    observed = _extract_mdout_term(mdout_path, "LJ_short")
    assert observed == pytest.approx(_lj_energy(-0.40, 4.184, 0.5), abs=0.011)
    assert observed > 0.0


def test_nonbond_params_leave_unmatched_type_pair_on_combination_rule(tmp_path):
    result, mdout_path = _run_two_atom_topology(
        tmp_path,
        """
        [ defaults ]
        1 2 yes 0.25 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.20 0.4184
        B B 12.0 0.0 A 0.20 0.4184
        C C 12.0 0.0 A 0.60 1.6736

        [ nonbond_params ]
        A B 1 0.10 0.0

        [ moleculetype ]
        PAIR 0

        [ atoms ]
        1 A 1 PAIR A 1 0.0 12.0
        2 C 1 PAIR C 2 0.0 12.0

        [ system ]
        unmatched nonbond pair

        [ molecules ]
        PAIR 1
        """,
        atom_names=("A", "C"),
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert _extract_mdout_term(mdout_path, "LJ_short") == pytest.approx(
        _lj_energy(0.40, 0.8368, 0.5), abs=0.011
    )


@pytest.mark.parametrize(
    ("pairtypes", "pair_entry", "sigma_nm", "epsilon_kj", "scale"),
    [
        ("", "1 2 1", 0.40, 4.184, 0.25),
        ("[ pairtypes ]\nA B 1 0.30 8.368", "1 2 1", 0.30, 8.368, 1.0),
        (
            "[ pairtypes ]\nA B 1 0.30 8.368",
            "1 2 1 0.45 4.184",
            0.45,
            4.184,
            1.0,
        ),
    ],
    ids=["generated_override_then_fudge", "pairtypes", "inline"],
)
def test_pair_parameter_priority_and_fudge_scope(
    tmp_path,
    pairtypes,
    pair_entry,
    sigma_nm,
    epsilon_kj,
    scale,
):
    result, mdout_path = _run_two_atom_topology(
        tmp_path,
        f"""
        [ defaults ]
        1 2 yes 0.25 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.20 0.4184
        B B 12.0 0.0 A 0.20 0.4184

        [ nonbond_params ]
        A B 1 0.40 4.184

        {pairtypes}

        [ moleculetype ]
        PAIR 1

        [ atoms ]
        1 A 1 PAIR A 1 0.0 12.0
        2 B 1 PAIR B 2 0.0 12.0

        [ bonds ]
        1 2 1 0.50 0.0

        [ pairs ]
        {pair_entry}

        [ system ]
        pair parameter priority

        [ molecules ]
        PAIR 1
        """,
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert _extract_mdout_term(mdout_path, "nb14_LJ") == pytest.approx(
        _lj_energy(sigma_nm, epsilon_kj, 0.5, scale), abs=0.011
    )


@pytest.mark.parametrize(
    ("entry", "error_fragment"),
    [
        ("A B 1 0.40", "expected exactly five fields"),
        ("A B 1 0.40 4.184 extra", "expected exactly five fields"),
        ("A B 2 0.40 4.184", "unsupported function"),
        ("A B 1 invalid 4.184", "invalid numeric field"),
        ("A UNKNOWN 1 0.40 4.184", "undefined GROMACS atom type 'UNKNOWN'"),
    ],
    ids=[
        "too_few",
        "too_many",
        "unsupported_funct",
        "invalid_value",
        "unknown_type",
    ],
)
def test_nonbond_params_reject_malformed_entries(
    tmp_path, entry, error_fragment
):
    result, _ = _run_two_atom_topology(
        tmp_path,
        f"""
        [ defaults ]
        1 2 yes 0.25 1.0

        [ atomtypes ]
        A A 12.0 0.0 A 0.30 0.4184
        B B 12.0 0.0 A 0.30 0.4184

        [ nonbond_params ]
        {entry}

        [ moleculetype ]
        PAIR 0

        [ atoms ]
        1 A 1 PAIR A 1 0.0 12.0
        2 B 1 PAIR B 2 0.0 12.0

        [ system ]
        malformed nonbond override

        [ molecules ]
        PAIR 1
        """,
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode != 0
    assert "spongeErrorBadFileFormat" in output
    assert error_fragment in output
