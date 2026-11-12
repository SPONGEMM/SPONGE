#!/usr/bin/env python3
"""Matched one-link NHC+SHAKE water response; no accuracy acceptance by default."""

import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import tomllib
import water_cosine_validation as reference
import water_model as water
import water_pilot as pilot
from lj_gk_compare import LM_KB, LM_TIME

from tools.viscosity.analysis import cosine
from tools.viscosity.workflows.config import dumps


def mapping(model, sp_mass=None):
    dof = 2 * model["atoms"] - 4
    kb_t = float(np.float32(np.float32(water.KB) * np.float32(300.0)))
    tau = float(2 * np.pi * np.sqrt(dof * water.KB) * 0.1)
    h_tau = np.float32(np.float32(tau) * np.float32(water.SP_TIME))
    predicted = np.float32(h_tau * h_tau)
    for factor in (np.float32(300.0),):
        predicted = np.float32(predicted * factor)
    for divisor in (np.float32(4.0), np.float32(np.pi), np.float32(np.pi)):
        predicted = np.float32(predicted / divisor)
    q = float(predicted) if sp_mass is None else float(sp_mass)
    tdamp = LM_TIME * np.sqrt(q / (dof * kb_t))
    return dict(
        thermal_dof=dof,
        chain_length=1,
        sponge_tau_ps=tau,
        sponge_chain_mass_native=q,
        predicted_sponge_mass_native=float(predicted),
        kbt_native=kb_t,
        lammps_target_k=kb_t / LM_KB,
        lammps_tdamp_fs=float(tdamp),
        lammps_chain_mass_native=float(dof * kb_t * tdamp**2 / LM_TIME**2),
        mass_convention="SP: tau_internal^2*T/(4*pi^2); LM: g*kBT*Tdamp^2; actual printed SP float mass used",
        thermal_projection="joint constraint tangent + COM + projected cosine",
        velocity_convention="SP native NHC stored q/v; LM native velocity Verlet q/v; finite-dt mismatch audited",
        finite_step_integrators_identical=False,
    )


def write_sponge(
    path, source, model, q, v, steps, dt, stride, amplitude, grid=128
):
    reference.write_sponge(
        path, source, model, q, v, steps, dt, stride, amplitude, 1, grid
    )
    cfg = tomllib.loads((path / "mdin.spg.toml").read_text())
    cfg["thermostat"] = dict(
        mode="nose_hoover_chain", tau=mapping(model)["sponge_tau_ps"]
    )
    cfg["nose_hoover_chain"] = dict(length=1)
    cfg["constrain"] = dict(
        mode="SHAKE", in_file=str(source / "constraint.txt"), precision="double"
    )
    cfg["settle_disable"] = True
    cfg["SHAKE"] = dict(iteration_numbers=100, tolerance=1e-10, step_length=1.0)
    cfg["sampling"]["velocity_time_layer"] = "stored"
    (path / "mdin.spg.toml").write_text(dumps(cfg))


def write_lammps(
    path, source, model, q, v, steps, dt, stride, amplitude, plugin, masses
):
    reference.write_lammps(
        path, source, model, q, v, steps, dt, stride, amplitude
    )
    text = (path / "in.lammps").read_text()
    text = "plugin load " + str(plugin) + "\n" + text
    text = text.replace(
        "compute cos all viscosity/cos",
        "compute cos all viscosity/cos\ncompute_modify cos extra/dof 4\ncompute joint all temp/water/cos/tangent",
    )
    text = re.sub(
        r"^fix bath all nvt temp .*$",
        f"fix bath all nvt temp {masses['lammps_target_k']:.17g} {masses['lammps_target_k']:.17g} {masses['lammps_tdamp_fs']:.17g} tchain 1 tloop 1",
        text,
        flags=re.M,
    )
    text = text.replace(
        "fix_modify bath temp cos", "fix_modify bath temp joint"
    )
    text = text.replace("c_cos c_cos[7]", "c_joint c_cos c_cos[7]")
    (path / "in.lammps").write_text(text)


def measure_native(q, v, mass, box):
    """Observe raw stored flow, but thermostat only its tangent thermal part."""
    measured = reference.measure(q, v, mass, box)
    tangent_v = reference.tangent(v, reference.tangent_geometry(q, mass, box))
    measured["thermal_temperature"] = reference.measure(
        q, tangent_v, mass, box
    )["thermal_temperature"]
    return measured


def audit_nhc_observables(path, expected):
    """Scalar observables include q0; native chain history starts after step 0."""
    with h5py.File(path) as file:
        journal = file["parameters/sponge/output"]
        if (
            journal["status"][()] != b"finalized"
            or int(journal["last_complete_step"][-1]) != expected[-1]
            or int(journal["frame_count"][-1]) != len(expected)
        ):
            raise ValueError("incomplete observable output")
        group = file["observables/all"]
        np.testing.assert_array_equal(group["step"][()], expected)
        scalar_count = 0
        for item in group.values():
            if isinstance(item, h5py.Group) and "value" in item:
                value = item["value"][()]
                if len(value) != len(expected) or not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError("invalid scalar observable")
                scalar_count += 1
        if scalar_count == 0:
            raise ValueError("missing scalar observables")
        chain = group["thermostat/nose_hoover_chain"]
        np.testing.assert_array_equal(
            chain["coordinate/step"][()], expected[1:]
        )
        for name in ("coordinate", "velocity"):
            value = chain[name + "/value"][()]
            if value.shape != (len(expected) - 1, 1) or not np.all(
                np.isfinite(value)
            ):
                raise ValueError("invalid one-link NHC history")


def audit_native_sponge(path, model, state, q0, v0, steps, dt, stride):
    data, meta = cosine.load_profile(path / "profile.h5")
    expected = np.arange(0, steps + 1, stride)
    np.testing.assert_array_equal(data["step"], expected)
    np.testing.assert_allclose(
        data["time"], expected * dt, atol=1e-12, rtol=1e-12
    )
    if (
        meta.get("profile_velocity_convention") != "native_nhc_stored_v1"
        or int(meta["thermal_dof"]) != 2 * model["atoms"] - 4
    ):
        raise ValueError("wrong NHC velocity/thermal-DOF metadata")
    audit_nhc_observables(path / "observables.h5", expected)
    n = model["atoms"]
    m = state["masses"]
    box = model["box_angstrom"]
    q = np.concatenate(
        (q0[None], pilot.native_frames(path / "crd.dat", n, steps // stride))
    )
    v = np.concatenate(
        (v0[None], pilot.native_frames(path / "vel.dat", n, steps // stride))
    )
    f = pilot.native_frames(path / "force.dat", n, len(expected))
    errors = {
        key: 0.0
        for key in ["velocity_fourier", "velocity_fit", "thermal_temperature"]
    }
    for position, velocity, index in zip(q, v, range(len(q))):
        measured = measure_native(position, velocity, m, box)
        for key in errors:
            errors[key] = max(
                errors[key], float(abs(measured[key] - data[key][index]))
            )
    if (
        max(errors["velocity_fourier"], errors["velocity_fit"]) > 1e-4
        or errors["thermal_temperature"] > 0.05
    ):
        raise ValueError(
            "native NHC independent observer mismatch " + str(errors)
        )
    length = pilot.geometry(q, state, model)
    if length > 2e-5:
        raise ValueError("native float SHAKE coordinate residual too large")
    return (
        data,
        dict(
            status="passed_short_numerical_profile_gates",
            independent_profile_max_errors=errors,
            max_constraint_length_residual=length,
            sampling_convention="native stored q/v; temperature tangent-projected; no force-only half kick",
            energy_balance_acceptance="not evaluated: native SHAKE coordinate/velocity correction is not a canonical drift ledger",
            formal_viscosity_acceptance=False,
        ),
        q,
        v,
        f,
    )


def run_case(
    campaign,
    replica,
    name,
    steps=256,
    dt=0.00025,
    stride=8,
    amplitude=0.2,
    discard=2.0,
    block=1.0,
    analysis_seed=20261112,
):
    campaign = Path(campaign).resolve()
    if (campaign / "closure.json").exists():
        raise ValueError("finalized campaign")
    protocol = json.loads((campaign / "plan.json").read_text())
    base = campaign / "replicas" / replica
    model, state = water.load_model(base / "model")
    sp = Path(protocol["engines"]["sponge"])
    lm = Path(protocol["engines"]["lammps"])
    plugin = Path(protocol["plugin"])
    for path, key in [(sp, "sponge_sha256"), (lm, "lammps_sha256")]:
        if water.digest(path) != protocol["engines"][key]:
            raise ValueError("engine changed")
    if water.digest(plugin) != protocol["plugin_sha256"]:
        raise ValueError("plugin changed")
    if (
        steps < 16
        or steps % stride
        or not 0 < dt <= 0.0005
        or steps // stride > 2000
        or amplitude == 0
    ):
        raise ValueError("invalid bounded case")
    out = base / "runs" / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with np.load(base / "physical-seed.npz") as initial:
            q = initial["positions"].astype(np.float32).astype(float)
            v = initial["velocity"].astype(np.float32).astype(float)
        plan = dict(
            steps=steps,
            dt_ps=dt,
            stride=stride,
            amplitude_nm_ps2=amplitude,
            discard_ps=discard,
            block_ps=block,
            analysis_seed=analysis_seed,
            trajectory_seed=None,
            seed_sha256=water.digest(base / "physical-seed.npz"),
            model_sha256=water.digest(base / "model/model.json"),
            protocol_sha256=water.digest(campaign / "plan.json"),
            source_sha256=water.digest(__file__),
            status="planned",
            initial_state="identical float q/v; native NHC stored initial velocity, no half-kick conversion",
            formal_viscosity_acceptance=False,
        )
        water.save(out / "plan.json", plan)
        write_sponge(
            out / "mass-probe", base / "model", model, q, v, 1, dt, 1, amplitude
        )
        water.execute(
            [sp, "-mdin", "mdin.spg.toml"], out / "mass-probe", timeout=1800
        )
        match = re.search(
            r"chain mass is\s+([-+\d.eE]+)",
            (out / "mass-probe/run.log").read_text(),
        )
        if not match:
            raise ValueError("missing actual SP NHC mass")
        actual_mass = float(np.float32(float(match[1])))
        mapped = mapping(model, actual_mass)
        if abs(actual_mass / mapped["predicted_sponge_mass_native"] - 1) > 1e-6:
            raise ValueError("unexpected NHC mass convention")
        write_sponge(
            out / "sponge",
            base / "model",
            model,
            q,
            v,
            steps,
            dt,
            stride,
            amplitude,
        )
        write_lammps(
            out / "lammps",
            base / "model",
            model,
            q,
            v,
            steps,
            dt,
            stride,
            amplitude,
            plugin,
            mapped,
        )
        water.save(out / "unit-mapping.json", mapped)
        np.savez(out / "initial-state.npz", positions=q, velocity=v)
        identities = {
            str(path.relative_to(out)): water.digest(path)
            for path in out.rglob("*")
            if path.is_file()
        }
        water.save(out / "input-identities.json", identities)
        timings = {}
        for engine, exe, flag, filename in [
            ("sponge", sp, "-mdin", "mdin.spg.toml"),
            ("lammps", lm, "-in", "in.lammps"),
        ]:
            water.save(
                out / "status.json",
                dict(phase="running", engine=engine, started_at=time.time()),
            )
            timings[engine] = water.execute(
                [exe, flag, filename], out / engine, timeout=1800
            )
        water.save(out / "execution.json", dict(engine_wall_seconds=timings))
        return audit_case(campaign, replica, name)
    except Exception as error:
        water.save(
            out / "failure.json",
            dict(error=str(error), wall_seconds=time.monotonic() - started),
        )
        water.save(out / "status.json", dict(phase="failed", error=str(error)))
        raise


def audit_case(campaign, replica, name):
    """Reanalyze finalized outputs without overwriting or rerunning any MD."""
    campaign = Path(campaign).resolve()
    if (campaign / "closure.json").exists():
        raise ValueError("finalized campaign")
    base = campaign / "replicas" / replica
    out = base / "runs" / name
    model, state = water.load_model(base / "model")
    plan = json.loads((out / "plan.json").read_text())
    steps, dt, stride = plan["steps"], plan["dt_ps"], plan["stride"]
    amplitude = plan["amplitude_nm_ps2"]
    discard, block = plan["discard_ps"], plan["block_ps"]
    analysis_seed = plan["analysis_seed"]
    identities = json.loads((out / "input-identities.json").read_text())
    execution = out / "execution.json"
    timings = (
        json.loads(execution.read_text())["engine_wall_seconds"]
        if execution.exists()
        else {}
    )
    with np.load(out / "initial-state.npz") as seed:
        q, v = seed["positions"], seed["velocity"]
    water.save(out / "status.json", dict(phase="postprocessing"))
    spdata, audit, qs, vs, fs = audit_native_sponge(
        out / "sponge", model, state, q, v, steps, dt, stride
    )
    ql, vl, fl = reference.read_lammps(
        out / "lammps/state.dump",
        model["atoms"],
        steps,
        stride,
        model["box_angstrom"],
    )
    times = np.arange(0, steps + 1, stride) * dt
    indices = np.arange(0, steps + 1, stride)
    lmdata = reference.profile_data(
        ql, vl, state["masses"], model, times, indices, amplitude
    )
    lmdata["thermal_temperature"] = np.array(
        [
            measure_native(qi, vi, state["masses"], model["box_angstrom"])[
                "thermal_temperature"
            ]
            for qi, vi in zip(ql, vl)
        ]
    )
    table = np.loadtxt(out / "lammps/thermo.dat", comments="#", ndmin=2)
    np.testing.assert_array_equal(table[:, 0], indices)
    if table.shape != (len(indices), 6):
        raise ValueError("wrong joint/native LAMMPS sample columns")
    t_error = float(
        np.max(
            abs(table[:, 1] * LM_KB / water.KB - lmdata["thermal_temperature"])
        )
    )
    v_error = float(
        np.max(
            abs(
                table[:, 3] * LM_TIME * water.SP_TIME / 10
                - lmdata["velocity_fourier"]
            )
        )
    )
    if t_error > 1e-5 or v_error > 1e-10:
        raise ValueError("LAMMPS independent projection/velocity mismatch")
    force_error = float(np.max(abs(fs[0] - fl[0])))
    velocity_error = float(np.max(abs(vs[0] - vl[0])))
    if force_error > 0.002 or velocity_error > 5e-7:
        raise ValueError("initial force/velocity not aligned")
    result = dict(
        status="completed_short_numerical_audit_not_viscosity_acceptance",
        sponge=audit,
        lammps=dict(
            independent_joint_temperature_max_error_k=t_error,
            native_fourier_max_error_nm_ps=v_error,
        ),
        initial_total_force_max_error_native=force_error,
        initial_stored_velocity_max_error_native=velocity_error,
        endpoint_cross_engine_rms_position_angstrom=float(
            np.sqrt(
                np.mean(
                    (
                        (qs[-1] - ql[-1] + model["box_angstrom"] / 2)
                        % model["box_angstrom"]
                        - model["box_angstrom"] / 2
                    )
                    ** 2
                )
            )
        ),
        endpoint_cross_engine_rms_stored_velocity_native=float(
            np.sqrt(np.mean((vs[-1] - vl[-1]) ** 2))
        ),
        engine_wall_seconds=timings,
        trajectory_seed=None,
        analysis_seed=analysis_seed,
        formal_viscosity_acceptance=False,
    )
    if steps * dt - discard >= 2 * block:
        result["viscosity"] = {}
        for engine, data in [("sponge", spdata), ("lammps", lmdata)]:
            result["viscosity"][engine] = {}
            for estimator in ["fourier", "com_fourier", "fit"]:
                report, _ = cosine.analyze(
                    data,
                    discard,
                    block,
                    estimator,
                    bootstrap=2000,
                    seed=analysis_seed,
                )
                if engine == "lammps":
                    report = {
                        k: v for k, v in report.items() if "work" not in k
                    }
                    report.update(
                        work_available=False,
                        work_in_window_kcal_per_mol=None,
                    )
                result["viscosity"][engine][estimator] = report
        result["status"] = (
            "completed_pilot_viscosity_estimate_not_convergence_acceptance"
        )
    lmdata.pop("coupling_work", None)
    np.savez(
        out / "common-profile.npz",
        **{f"sponge_{k}": v for k, v in spdata.items()},
        **{f"lammps_{k}": v for k, v in lmdata.items()},
    )
    for filename, digest in identities.items():
        if water.digest(out / filename) != digest:
            raise ValueError("input mutated " + filename)
    result.update(
        source_sha256=water.digest(__file__),
        total_engine_wall_seconds=sum(timings.values()),
    )
    water.save(out / "summary.json", result)
    water.save(out / "status.json", dict(phase="completed"))
    return result
