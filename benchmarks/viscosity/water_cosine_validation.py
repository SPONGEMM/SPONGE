#!/usr/bin/env python3
"""Bounded SPC/E cosine response checks; native LAMMPS is a statistical reference.

No automatic zero-shear, equilibrium or experimental-accuracy acceptance.
Inputs, failures and unresolved response intervals are retained in new directories.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import water_model as water
import water_pilot as pilot
from water_constrained_pair import whole_water
from hexane_rattle import project
from lj_gk_compare import LM_TIME, LM_KB

sys.path.insert(0, str(water.ROOT))
from tools.viscosity.analysis import cosine
from tools.viscosity.workflows.config import dumps


def tangent_geometry(q, mass, box):
    """Independent batched water distance Jacobians in physical units."""
    x = np.asarray(q).reshape(-1, 3, 3)
    m = np.asarray(mass).reshape(-1, 3)
    jac = np.zeros((len(x), 3, 9))
    for row, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        d = x[:, a] - x[:, b]
        d -= box * np.floor(d / box + 0.5)
        d /= np.linalg.norm(d, axis=1)[:, None]
        jac[:, row, 3 * a : 3 * a + 3] = d
        jac[:, row, 3 * b : 3 * b + 3] = -d
    inverse = 1 / np.repeat(m, 3, axis=1)
    metric = (jac * inverse[:, None, :]) @ jac.transpose(0, 2, 1)
    return jac, inverse, metric


def tangent(v, geometry):
    jac, inverse, metric = geometry
    flat = np.asarray(v).reshape(-1, 9)
    rhs = -(jac @ flat[:, :, None])
    multiplier = np.linalg.solve(metric, rhs)
    correction = inverse * (jac.transpose(0, 2, 1) @ multiplier)[:, :, 0]
    return (flat + correction).reshape(-1, 3)


def measure(q, u, mass, box):
    """Mass Fourier/fit and *joint tangent* thermal state, independent of C++."""
    geometry = tangent_geometry(q, mass, box)
    com = np.average(u, axis=0, weights=mass)
    c = np.cos(2 * np.pi * q[:, 2] / box)
    mean = np.average(c, weights=mass)
    variance = np.average((c - mean) ** 2, weights=mass)
    covariance = np.average((u[:, 0] - com[0]) * (c - mean), weights=mass)
    retained = np.zeros_like(u)
    retained[:, 0] = c
    retained = tangent(retained, geometry)
    retained -= np.average(retained, axis=0, weights=mass)
    coefficient = np.sum(mass[:, None] * (u - com) * retained) / np.sum(
        mass[:, None] * retained**2
    )
    thermal = u - com - coefficient * retained
    dof = 2 * len(mass) - 4
    conversion = water.SP_TIME / 10
    return dict(
        velocity_fourier=2 * np.average(u[:, 0] * c, weights=mass) * conversion,
        velocity_com_fourier=2 * covariance * conversion,
        velocity_fit=covariance / variance * conversion,
        mean_cosine=mean,
        variance_cosine=variance,
        thermal_temperature=np.sum(mass[:, None] * thermal**2)
        / (dof * water.KB),
        kinetic_temperature=np.sum(mass[:, None] * (u - com) ** 2)
        / ((dof + 1) * water.KB),
        com_vx=com[0] * conversion,
        com_vy=com[1] * conversion,
        com_vz=com[2] * conversion,
    )


def force_drive(q, mass, model, amplitude):
    f = np.zeros_like(q)
    f[:, 0] = (
        mass
        * amplitude
        * 10
        / water.SP_TIME**2
        * np.cos(2 * np.pi * q[:, 2] / model["box_angstrom"])
    )
    return f


def write_sponge(
    path,
    model_dir,
    model,
    q,
    incoming,
    steps,
    dt,
    stride,
    amplitude,
    seed,
    grid,
):
    path.mkdir()
    water.array_file(path / "coordinate.txt", q, model["box_angstrom"])
    water.array_file(path / "velocity.txt", incoming)
    cfg = dict(
        md_name="spce_cosine_validation",
        mode="nvt",
        pbc=True,
        dt=dt,
        step_limit=steps,
        target_temperature=300.0,
        cutoff=model["cutoff_angstrom"],
        skin=1.0,
        thermostat=dict(mode="middle_langevin", tau=0.1, seed=seed),
        constrain=dict(
            mode="SETTLE", in_file=str(model_dir / "constraint.txt")
        ),
        print_zeroth_frame=True,
        write_mdout_interval=stride,
        write_information_interval=stride,
        write_trajectory_interval=stride,
        write_restart_file_interval=0,
        crd="crd.dat",
        vel="vel.dat",
        frc="force.dat",
        output_h5_observable_path="observables.h5",
        coordinate_in_file=str(path / "coordinate.txt"),
        velocity_in_file=str(path / "velocity.txt"),
        PM=dict(
            MPI_size=1, fftx=grid, ffty=grid, fftz=grid, Direct_Tolerance=1e-10
        ),
        external_force=[
            dict(
                type="acceleration",
                acceleration=[amplitude, 0.0, 0.0],
                profile=dict(type="cosine", axis="z", mode=1),
            )
        ],
        velocity_bias=dict(mode="cosine", flow="x", gradient="z"),
        sampling=dict(
            profile="cosine",
            flow="x",
            gradient="z",
            bins=20,
            interval=stride,
            output="profile.h5",
        ),
    )
    for key, name in [
        ("mass", "mass.txt"),
        ("charge", "charge.txt"),
        ("LJ", "lj.txt"),
        ("exclude", "exclude.txt"),
        ("residue", "residue.txt"),
    ]:
        cfg[key + "_in_file"] = str(model_dir / name)
    (path / "mdin.spg.toml").write_text(dumps(cfg))


def write_lammps(path, model_dir, model, q, u, steps, dt, stride, amplitude):
    path.mkdir()
    prefix, rest = (
        (model_dir / "lammps.data").read_text().split("Atoms # full\n\n")
    )
    atoms, suffix = rest.split("\nBonds\n")
    rows = [line.split() for line in atoms.splitlines() if line.strip()]
    xyz = whole_water(q, model["box_angstrom"])
    atoms = "\n".join(
        " ".join(row[:4]) + " " + " ".join(f"{v:.17g}" for v in r)
        for row, r in zip(rows, xyz)
    )
    velocities = "\n".join(
        str(i + 1) + " " + " ".join(f"{v / LM_TIME:.17g}" for v in r)
        for i, r in enumerate(u)
    )
    (path / "model.data").write_text(
        prefix
        + "Atoms # full\n\n"
        + atoms
        + "\n\nBonds\n"
        + suffix
        + "\nVelocities\n\n"
        + velocities
        + "\n"
    )
    water.lammps_input(path, model_dir, model, "full")
    setup = (
        (path / "in.lammps")
        .read_text()
        .split("compute p all")[0]
        .replace(str(model_dir / "lammps.data"), "model.data")
    )
    h = water.internal_timestep(dt)
    # Align effective Hamiltonian units, while retaining the original native
    # thermostat/constraint algorithm as an independent finite-step reference.
    a_lm = amplitude * 10 / (water.SP_TIME * LM_TIME) ** 2
    target = 300 * water.KB / LM_KB
    setup += f"""timestep {h * LM_TIME:.17g}
fix drive all accelerate/cos {a_lm:.17g}
fix raw all store/force
compute cos all viscosity/cos
fix bath all nvt temp {target:.17g} {target:.17g} {(0.1 * water.SP_TIME * LM_TIME):.17g}
fix_modify bath temp cos
fix constraints all shake 1e-10 100 0 b 1 a 1
variable potential equal pe
variable kinetic equal ke
fix samples all ave/time 1 1 {stride} c_cos c_cos[7] v_potential v_kinetic file thermo.dat format " %.17g"
dump state all custom {stride} state.dump id x y z vx vy vz f_raw[1] f_raw[2] f_raw[3]
dump_modify state sort id format float %.17g
thermo {max(stride, steps // 10)}
thermo_style custom step c_cos c_cos[7] pe ke
thermo_modify norm no format float %.17g
run {steps}
"""
    (path / "in.lammps").write_text(setup)
    return dict(
        dt_fs=h * LM_TIME,
        target_temperature_lammps_k=target,
        acceleration_angstrom_fs2=a_lm,
        thermostat="native NHC + SHAKE",
        com_reset=False,
        native_cosine_compute_is_not_joint_projection=True,
    )


def read_lammps(path, atoms, steps, stride, box):
    frames = []
    with path.open() as stream:
        for step in range(0, steps + 1, stride):
            if (
                stream.readline().strip() != "ITEM: TIMESTEP"
                or int(stream.readline()) != step
            ):
                raise ValueError("LAMMPS dump time mismatch")
            if (
                stream.readline().strip() != "ITEM: NUMBER OF ATOMS"
                or int(stream.readline()) != atoms
            ):
                raise ValueError("LAMMPS dump atom mismatch")
            if stream.readline().strip() != "ITEM: BOX BOUNDS pp pp pp":
                raise ValueError("LAMMPS box header")
            cell = np.array(
                [list(map(float, stream.readline().split())) for _ in range(3)]
            )
            np.testing.assert_allclose(cell, [[0, box]] * 3, atol=1e-12, rtol=0)
            header = stream.readline().strip()
            if (
                header
                != "ITEM: ATOMS id x y z vx vy vz f_raw[1] f_raw[2] f_raw[3]"
            ):
                raise ValueError("LAMMPS dump fields")
            frame = np.array(
                [
                    list(map(float, stream.readline().split()))
                    for _ in range(atoms)
                ]
            )
            if (
                frame.shape != (atoms, 10)
                or not np.all(np.isfinite(frame))
                or not np.array_equal(frame[:, 0], np.arange(1, atoms + 1))
            ):
                raise ValueError("LAMMPS dump values/order")
            frames.append(frame[:, 1:])
        if stream.read().strip():
            raise ValueError("LAMMPS extra dump frames")
    values = np.array(frames)
    return values[:, :, :3], values[:, :, 3:6] * LM_TIME, values[:, :, 6:]


def profile_data(q, u, mass, model, times, steps, amplitude):
    values = {name: [] for name in cosine.UNITS}
    bins = 20
    populations = {
        key: []
        for key in ("count", "mass", "momentum", "cosine_mass", "sine_mass")
    }
    for position, velocity in zip(q, u):
        sample = measure(position, velocity, mass, model["box_angstrom"])
        sample.update(
            density=model["actual_density_kg_m3"] / 1000,
            wave_number=2 * np.pi / (model["box_angstrom"] / 10),
            length=model["box_angstrom"] / 10,
            acceleration=amplitude,
            coupling_work=0.0,
        )
        for name in values:
            values[name].append(sample[name])
        c = np.cos(2 * np.pi * position[:, 2] / model["box_angstrom"])
        s = np.sin(2 * np.pi * position[:, 2] / model["box_angstrom"])
        indices = np.floor(
            (position[:, 2] % model["box_angstrom"])
            / model["box_angstrom"]
            * bins
        ).astype(int)
        for key, weight in dict(
            count=np.ones(len(mass)),
            mass=mass,
            momentum=mass * velocity[:, 0] * water.SP_TIME / 10,
            cosine_mass=mass * c,
            sine_mass=mass * s,
        ).items():
            populations[key].append(
                np.bincount(indices, weights=weight, minlength=bins)
            )
    data = {name: np.array(column) for name, column in values.items()}
    data.update(
        {f"bin_{key}": np.array(column) for key, column in populations.items()}
    )
    data.update(time=times, step=steps)
    # Native LAMMPS does not provide the exact same path-work ledger here.
    # Keep it explicitly unavailable in the public report, not a measured zero.
    return data


def audit_sponge(path, model, state, q0, incoming, steps, dt, stride):
    data, meta = cosine.load_profile(path / "profile.h5")
    expected = np.arange(0, steps + 1, stride)
    np.testing.assert_array_equal(data["step"], expected)
    np.testing.assert_allclose(
        data["time"], expected * dt, rtol=1e-12, atol=1e-12
    )
    ordinary = pilot.read_observables(path / "observables.h5", expected)
    n = model["atoms"]
    mass = state["masses"]
    box = model["box_angstrom"]
    q = np.concatenate(
        (q0[None], pilot.native_frames(path / "crd.dat", n, steps // stride))
    )
    s = np.concatenate(
        (
            incoming[None],
            pilot.native_frames(path / "vel.dat", n, steps // stride),
        )
    )
    force = pilot.native_frames(path / "force.dat", n, len(expected))
    physical = []
    errors = {
        name: []
        for name in ("velocity_fourier", "velocity_fit", "thermal_temperature")
    }
    h = water.internal_timestep(dt)
    for i in range(len(q)):
        u = tangent(
            s[i] + 0.5 * h * force[i] / mass[:, None],
            tangent_geometry(q[i], mass, box),
        )
        physical.append(u)
        sample = measure(q[i], u, mass, box)
        for name in errors:
            errors[name].append(abs(sample[name] - data[name][i]))
    errors = {name: float(max(values)) for name, values in errors.items()}
    if (
        max(errors["velocity_fourier"], errors["velocity_fit"]) > 1e-4
        or errors["thermal_temperature"] > 0.05
    ):
        raise ValueError(
            "independent physical-time profile reconstruction failed: "
            + str(errors)
        )
    if (
        max(data["constraint_length_residual"]) > 1e-10
        or max(data["constraint_tangent_residual"]) > 1e-8
    ):
        raise ValueError("canonical constraint residual failed")
    # Same force-time energy, with COM restored, and continuous LJ cutoff energy.
    kinetic = 0.5 * (2 * n - 3) * water.KB * data["kinetic_temperature"]
    kinetic += (
        0.5
        * mass.sum()
        * sum(
            (data[f"com_v{axis}"] * 10 / water.SP_TIME) ** 2 for axis in "xyz"
        )
    )
    shift = np.array([pilot.energy_shift(frame, state, model) for frame in q])
    energy = ordinary["eff_pot"] - shift + kinetic
    residual = (
        energy
        - energy[0]
        - (data["coupling_work"] - data["coupling_work"][0])
        - (data["thermostat_heat"] - data["thermostat_heat"][0])
    )
    np.savetxt(
        path / "energy-audit.csv",
        np.column_stack(
            (
                data["time"],
                energy,
                data["coupling_work"],
                data["thermostat_heat"],
                residual,
                data["constraint_drift_energy"],
            )
        ),
        delimiter=",",
        header="time_ps,total_shifted_kcal_mol,work_kcal_mol,heat_kcal_mol,balance_residual_kcal_mol,drift_kinetic_change_kcal_mol",
        comments="",
    )
    audit = dict(
        status="passed_short_numerical_profile_gates",
        independent_profile_max_errors=errors,
        mean_thermal_temperature_k=float(np.mean(data["thermal_temperature"])),
        max_constraint_length_residual=float(
            max(data["constraint_length_residual"])
        ),
        max_constraint_tangent_residual_nm_ps=float(
            max(data["constraint_tangent_residual"])
        ),
        energy_balance_residual_endpoint_kcal_mol=float(residual[-1]),
        energy_balance_residual_range_kcal_mol=float(np.ptp(residual)),
        energy_balance_acceptance="diagnostic; timestep/mesh dependence must be evaluated",
        formal_viscosity_acceptance=False,
    )
    return data, audit


def audit_case(campaign, name):
    campaign = Path(campaign).resolve()
    if (campaign / "closure.json").exists():
        raise ValueError("finalized campaign; create a new analysis archive")
    out = campaign / "runs" / name
    model, state = water.load_model(campaign / "model")
    mass = state["masses"]
    plan = json.loads((out / "plan.json").read_text())
    steps, dt, stride, amplitude, discard, block = [
        plan[k]
        for k in (
            "steps",
            "dt_ps",
            "stride",
            "amplitude_nm_ps2",
            "discard_ps",
            "block_ps",
        )
    ]
    initial = np.load(out / "initial-state.npz")
    q = initial["canonical"]
    incoming = initial["staggered"]
    for name_, sha in json.loads(
        (out / "input-identities.json").read_text()
    ).items():
        if water.digest(out / name_) != sha:
            raise ValueError("input mutated: " + name_)
    # Reanalysis may intentionally use repaired analysis code. Preserve the
    # preceding outputs and identify both versions instead of silently replacing
    # the evidence or requiring the execution and analysis sources to coincide.
    if (out / "summary.json").exists():
        history = out / "analysis-history" / water.digest(out / "summary.json")
        if not history.exists():
            history.mkdir(parents=True)
            for filename in (
                "summary.json",
                "status.json",
                "common-profile.npz",
                "sponge/energy-audit.csv",
                "lammps/completion.json",
            ):
                original = out / filename
                if original.exists():
                    target = history / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
    analysis_hash = water.digest(__file__)
    snapshot = out / "analysis-sources" / analysis_hash
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    times = (
        json.loads((out / "execution.json").read_text())
        if (out / "execution.json").exists()
        else None
    )
    started = time.monotonic()
    sp_data, audit = audit_sponge(
        out / "sponge", model, state, q, incoming, steps, dt, stride
    )
    ql, ul, fl = read_lammps(
        out / "lammps/state.dump",
        model["atoms"],
        steps,
        stride,
        model["box_angstrom"],
    )
    lm_data = profile_data(
        ql,
        ul,
        mass,
        model,
        np.arange(0, steps + 1, stride) * dt,
        np.arange(0, steps + 1, stride),
        amplitude,
    )
    table = np.loadtxt(out / "lammps/thermo.dat", comments="#", ndmin=2)
    np.testing.assert_array_equal(table[:, 0], np.arange(0, steps + 1, stride))
    error = float(
        np.max(
            abs(
                table[:, 2] * LM_TIME * water.SP_TIME / 10
                - lm_data["velocity_fourier"]
            )
        )
    )
    if error > 1e-10:
        raise ValueError(
            "LAMMPS native Fourier and independent reconstruction disagree"
        )
    if (
        max(
            abs(
                sp_data["velocity_fourier"][0] - lm_data["velocity_fourier"][0]
            ),
            abs(sp_data["velocity_fit"][0] - lm_data["velocity_fit"][0]),
        )
        > 1e-4
    ):
        raise ValueError("initial cross-engine physical profile mismatch")
    initial_force = pilot.native_frames(
        out / "sponge/force.dat", model["atoms"], len(sp_data["time"])
    )[0]
    initial_physical = tangent(
        incoming
        + 0.5 * water.internal_timestep(dt) * initial_force / mass[:, None],
        tangent_geometry(q, mass, model["box_angstrom"]),
    )
    handoff_error = float(np.max(abs(initial_physical - ul[0])))
    if handoff_error > 5e-7:
        raise ValueError(
            "actual initial physical velocity mismatch: " + str(handoff_error)
        )
    force_error = float(np.max(abs(initial_force - fl[0])))
    if force_error > 0.002:
        raise ValueError("initial total force mismatch: " + str(force_error))
    water.save(
        out / "lammps/completion.json",
        dict(
            status="validated_complete_dump",
            final_step=steps,
            frames=len(ql),
            native_fourier_max_error_nm_ps=error,
        ),
    )
    result = dict(
        status="completed_short_numerical_audit_not_viscosity_acceptance",
        sponge=audit,
        initial_physical_velocity_max_error_native=handoff_error,
        initial_total_force_max_error_native=force_error,
        lammps=dict(
            native_fourier_max_error_nm_ps=error,
            mean_joint_thermal_temperature_k=float(
                np.mean(lm_data["thermal_temperature"])
            ),
        ),
        engine_wall_seconds=times,
        analysis_wall_seconds=time.monotonic() - started,
        formal_viscosity_acceptance=False,
    )
    if steps * dt - discard >= 2 * block:
        result["viscosity"] = {}
        for engine, data in [("sponge", sp_data), ("lammps", lm_data)]:
            result["viscosity"][engine] = {}
            for estimator in ("fourier", "com_fourier", "fit"):
                report, profile = cosine.analyze(
                    data,
                    discard,
                    block,
                    estimator,
                    bootstrap=1000,
                    seed=20261111,
                )
                if engine == "lammps":
                    for key in list(report):
                        if "work" in key:
                            del report[key]
                    report.update(
                        work_available=False,
                        work_in_window_kcal_per_mol=None,
                        note="path work not tallied by this native reference; placeholder is not a measured zero",
                    )
                result["viscosity"][engine][estimator] = report
        result["status"] = (
            "completed_pilot_viscosity_estimate_not_convergence_acceptance"
        )
    lm_data.pop("coupling_work", None)
    np.savez(
        out / "common-profile.npz",
        **{"sponge_" + k: v for k, v in sp_data.items()},
        **{"lammps_" + k: v for k, v in lm_data.items()},
    )
    result["analysis_source_sha256"] = analysis_hash
    result["executed_source_sha256"] = plan["source_sha256"]
    result["reanalysis_with_changed_source"] = (
        result["analysis_source_sha256"] != result["executed_source_sha256"]
    )
    if (out / "failure.json").exists():
        result["retained_prior_failure_sha256"] = water.digest(
            out / "failure.json"
        )
    result["analysis_wall_seconds"] = time.monotonic() - started
    water.save(out / "summary.json", result)
    water.save(
        out / "status.json",
        dict(
            phase="completed", analysis_wall_seconds=time.monotonic() - started
        ),
    )
    return result


def run_case(
    campaign,
    name,
    steps=256,
    dt=0.00025,
    stride=8,
    amplitude=0.2,
    seed=20261111,
    grid=128,
    discard=0.5,
    block=0.5,
):
    campaign = Path(campaign).resolve()
    if (campaign / "closure.json").exists():
        raise ValueError("finalized campaign; create a new campaign")
    plan = json.loads((campaign / "plan.json").read_text())
    gate = json.loads(
        (campaign / "same-state-refined/summary.json").read_text()
    )
    if gate["status"] != "passed_same_state_gates":
        raise ValueError("static gate has not passed")
    if (
        steps < 16
        or steps % stride
        or not 0 < dt <= 0.0005
        or amplitude == 0
        or grid not in (128, 256)
    ):
        raise ValueError("invalid bounded run design")
    source = campaign / "model"
    model, state = water.load_model(source)
    sp = Path(plan["engines"]["sponge"])
    lm = Path(plan["engines"]["lammps"])
    for key, exe in [("sponge", sp), ("lammps", lm)]:
        if water.digest(exe) != plan["engines"][key + "_sha256"]:
            raise ValueError("engine hash changed")
    out = campaign / "runs" / name
    out.mkdir(parents=True, exist_ok=False)
    run_plan = dict(
        steps=steps,
        dt_ps=dt,
        stride=stride,
        amplitude_nm_ps2=amplitude,
        seed=seed,
        grid=grid,
        discard_ps=discard,
        block_ps=block,
        model_sha256=water.digest(source / "model.json"),
        campaign_plan_sha256=water.digest(campaign / "plan.json"),
        source_sha256=water.digest(__file__),
        status="planned",
        reference="native NHC+SHAKE; differs from SPONGE projected Middle",
        formal_viscosity_acceptance=False,
    )
    water.save(out / "plan.json", run_plan)
    started = time.monotonic()
    try:
        initial = np.load(campaign / "physical-seed.npz")
        qraw = initial["positions"]
        u = initial["velocity"]
        mass = state["masses"]
        probe = (
            water.ROOT
            / "build-components-cpu/tests/control/test_constrained_nve"
        )
        q, u = project(
            probe,
            qraw,
            u,
            mass,
            state["pairs"],
            model["box_angstrom"],
            out / "canonical-preparation",
        )
        # Obtain F0 through the identical canonical integrator/shadow-coordinate
        # path. An ordinary unprojected force probe samples a slightly different
        # float coordinate even when both begin from the same text input.
        write_sponge(
            out / "force-probe",
            source,
            model,
            qraw,
            np.zeros_like(u),
            1,
            dt,
            1,
            amplitude,
            seed,
            grid,
        )
        water.execute(
            [sp, "-mdin", "mdin.spg.toml"], out / "force-probe", timeout=1800
        )
        f = pilot.native_frames(
            out / "force-probe/force.dat", model["atoms"], 2
        )[0]
        incoming = np.asarray(
            u - 0.5 * water.internal_timestep(dt) * f / mass[:, None],
            np.float32,
        ).astype(float)
        physical = tangent(
            incoming + 0.5 * water.internal_timestep(dt) * f / mass[:, None],
            tangent_geometry(q, mass, model["box_angstrom"]),
        )
        if np.max(abs(physical - u)) > 5e-6:
            raise ValueError("initial physical velocity handoff error")
        write_sponge(
            out / "sponge",
            source,
            model,
            qraw,
            incoming,
            steps,
            dt,
            stride,
            amplitude,
            seed,
            grid,
        )
        conversion = write_lammps(
            out / "lammps",
            source,
            model,
            q,
            physical,
            steps,
            dt,
            stride,
            amplitude,
        )
        water.save(out / "unit-mapping.json", conversion)
        np.savez(
            out / "initial-state.npz",
            raw=qraw,
            canonical=q,
            staggered=incoming,
            physical=physical,
            force=f,
        )
        identities = {
            str(p.relative_to(out)): water.digest(p)
            for p in out.rglob("*")
            if p.is_file()
        }
        water.save(out / "input-identities.json", identities)
        times = {}
        for engine, exe, filename, flag in [
            ("sponge", sp, "mdin.spg.toml", "-mdin"),
            ("lammps", lm, "in.lammps", "-in"),
        ]:
            water.save(
                out / "status.json",
                dict(phase="running", engine=engine, started_at=time.time()),
            )
            times[engine] = water.execute(
                [exe, flag, filename], out / engine, timeout=1800
            )
        for name_, sha in identities.items():
            if water.digest(out / name_) != sha:
                raise ValueError("input mutated: " + name_)
        water.save(
            out / "status.json",
            dict(phase="postprocessing", engine_wall_seconds=times),
        )
        water.save(out / "execution.json", times)
        result = audit_case(campaign, name)
        result["total_wall_seconds"] = time.monotonic() - started
        water.save(out / "summary.json", result)
        return result
    except Exception as error:
        water.save(
            out / "failure.json",
            dict(error=str(error), wall_seconds=time.monotonic() - started),
        )
        water.save(out / "status.json", dict(phase="failed", error=str(error)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--dt", type=float, default=0.00025)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--amplitude", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--discard", type=float, default=0.5)
    p.add_argument("--block", type=float, default=0.5)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    r = (
        audit_case(a.campaign, a.name)
        if a.audit_only
        else run_case(
            a.campaign,
            a.name,
            a.steps,
            a.dt,
            a.stride,
            a.amplitude,
            a.seed,
            a.grid,
            a.discard,
            a.block,
        )
    )
    print(json.dumps(r, indent=2, allow_nan=False))
