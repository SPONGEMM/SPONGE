#!/usr/bin/env python3
"""Cosine viscosity from an explicit stationary window and ratio of mean amplitude.

Intervals are conditional on independent blocks and linear steady response.
This program does not infer equilibration or prove those assumptions.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

CONTRACT = "sponge.cosine.centered.v1"
CONSTRAINED_CONTRACT = "sponge.cosine.constrained_rattle.v1"
CONSTRAINED_HEAT_METHOD = "joint_tangent_ou_double_v1"
CONSTRAINED_UNITS = {
    **{name: "Pa" for name in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")},
    "constraint_drift_energy": "kcal/mol",
    "constraint_length_residual": "1",
    "constraint_tangent_residual": "nm/ps",
    "kinetic_temperature": "K",
}
ESTIMATORS = ("fourier", "com_fourier", "fit")
HEAT_METHOD = "projected_middle_actual_kinetic_delta_v1"
UNITS = {
    "velocity_fourier": "nm/ps",
    "velocity_com_fourier": "nm/ps",
    "velocity_fit": "nm/ps",
    "mean_cosine": "1",
    "variance_cosine": "1",
    "thermal_temperature": "K",
    "com_vx": "nm/ps",
    "com_vy": "nm/ps",
    "com_vz": "nm/ps",
    "density": "g/cm^3",
    "acceleration": "nm/ps^2",
    "wave_number": "nm^-1",
    "length": "nm",
    "coupling_work": "kcal/mol",
}


def text(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("scalar metadata required")
    scalar = value.reshape(-1)[0]
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def load_profile(path):
    with h5py.File(path, "r") as file:
        provenance = {
            key: text(value[()])
            for key, value in file["parameters/sponge/provenance"].items()
        }
        generic = "observable_contract" in provenance
        if generic:
            if (
                provenance["observable_contract"]
                != "sponge.observables.physical_state.v1"
                or provenance.get("profile_kind") != "cosine"
            ):
                raise ValueError("unsupported observable cosine profile")
            if provenance.get("velocity_time_layer") != "force_centered":
                raise ValueError(
                    "cosine workflow requires force-centered velocity"
                )
            contract = (
                CONSTRAINED_CONTRACT
                if "canonical_integrator_configuration" in provenance
                else CONTRACT
            )
        else:
            contract = provenance["viscosity_contract"]
        if contract not in (CONTRACT, CONSTRAINED_CONTRACT):
            raise ValueError("unsupported cosine contract")
        constrained = contract == CONSTRAINED_CONTRACT
        if constrained and not generic:
            for key in (
                "constraint_configuration",
                "sampling_convention",
                "stress_convention",
                "work_convention",
                "state_precision",
                "thermostat_heat_convention",
            ):
                if not provenance.get(key):
                    raise ValueError(f"constrained provenance missing: {key}")
            if (
                provenance.get("thermostat_heat_method")
                != CONSTRAINED_HEAT_METHOD
            ):
                raise ValueError(
                    "constrained cosine requires joint tangent OU heat"
                )
        output = file["parameters/sponge/output"]
        if text(output["status"][()]) != "finalized":
            raise ValueError("profile file is not finalized")
        group = file["observables/all"]
        times = np.asarray(group["time"], dtype=float)
        steps = np.asarray(group["step"])
        n = len(times)
        if (
            times.ndim != 1
            or steps.shape != times.shape
            or n < 3
            or not np.issubdtype(steps.dtype, np.integer)
        ):
            raise ValueError("invalid step/time shape or too few frames")
        if text(group["time"].attrs.get("unit", "")) != "ps":
            raise ValueError("time unit must be ps")
        dt, stride = (
            float(provenance["dt_ps"]),
            int(provenance["sample_interval"]),
        )
        if (
            not np.isfinite(dt)
            or dt <= 0
            or stride < 1
            or not np.all(np.isfinite(times))
        ):
            raise ValueError("invalid time metadata")
        if (
            not np.all(np.diff(steps) == stride)
            or not np.allclose(
                np.diff(times), dt * stride, rtol=1e-9, atol=1e-12
            )
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("nonuniform sampling or metadata mismatch")
        counts = np.asarray(output["frame_count"])
        complete_steps = np.asarray(output["last_complete_step"])
        complete_times = np.asarray(output["last_complete_time"])
        if (
            counts.ndim != 1
            or not len(counts)
            or counts.shape != complete_steps.shape
            or counts.shape != complete_times.shape
            or not np.issubdtype(counts.dtype, np.integer)
        ):
            raise ValueError("invalid completion journal shape")
        if not np.issubdtype(complete_steps.dtype, np.integer):
            raise ValueError("integer completion steps required")
        if (
            np.any(counts < 0)
            or np.any(counts > n)
            or np.any(np.diff(counts) <= 0)
            or counts[-1] != n
        ):
            raise ValueError("invalid completion frame counts")
        positive = counts > 0
        if np.any(~positive) and (
            counts[0] != 0 or complete_steps[0] != -1 or complete_times[0] != 0
        ):
            raise ValueError("invalid initial completion marker")
        if not np.array_equal(
            complete_steps[positive], steps[counts[positive] - 1]
        ) or not np.array_equal(
            complete_times[positive], times[counts[positive] - 1]
        ):
            raise ValueError("completion history does not match samples")
        bins = int(provenance["bins"])
        if bins < 4 or bins > 10000:
            raise ValueError("invalid bin count")

        def column(name, unit):
            dataset = group[f"{name}/value"]
            array = np.asarray(dataset, dtype=float)
            if (
                array.shape != times.shape
                or not np.all(np.isfinite(array))
                or text(dataset.attrs.get("unit", "")) != unit
            ):
                raise ValueError(f"invalid shape, values or units: {name}")
            return array

        data = {"time": times, "step": steps}
        for key, unit in {
            "mass": "Da",
            "momentum": "Da nm/ps",
            "count": "1",
            "cosine_mass": "Da",
            "sine_mass": "Da",
        }.items():
            data[f"bin_{key}"] = np.column_stack(
                [column(f"bin_{i}_{key}", unit) for i in range(bins)]
            )
        data.update(
            {
                name: column(name, unit)
                for name, unit in UNITS.items()
                if not (generic and name == "acceleration")
            }
        )
        if generic:
            # Read the declared protocol, never infer acceleration from response.
            commands = dict(
                line.split("=", 1)
                for line in provenance["effective_commands"].splitlines()
            )
            drives = tomllib.loads(
                "external_force=" + commands.get("external_force", "[]")
            )["external_force"]
            candidates = [
                d
                for d in drives
                if d.get("type") == "acceleration"
                and d.get("profile", {}).get("type") == "cosine"
            ]
            if len(drives) != 1 or len(candidates) != 1:
                raise ValueError(
                    "cosine analysis requires one declared cosine acceleration drive"
                )
            drive = candidates[0]
            if (
                drive.get("schedule", {}).get("type", "constant") != "constant"
                or "atoms" in drive
            ):
                raise ValueError(
                    "cosine analysis requires constant all-atom acceleration"
                )
            flow, gradient = (
                int(provenance["flow"]),
                int(provenance["gradient"]),
            )
            if drive["profile"].get("axis", "z") != "xyz"[gradient]:
                raise ValueError("drive and measured profile axes differ")
            vector = drive["acceleration"]
            if any(float(x) != 0 for i, x in enumerate(vector) if i != flow):
                raise ValueError(
                    "transverse acceleration is outside the cosine workflow"
                )
            harmonic = int(drive["profile"].get("mode", 1))
            if not np.allclose(
                data["wave_number"] * data["length"], 2 * np.pi * harmonic
            ):
                raise ValueError("drive and measurement wave numbers differ")
            data["acceleration"] = np.full(times.shape, float(vector[flow]))
        if constrained:
            data.update(
                {
                    name: column(name, unit)
                    for name, unit in CONSTRAINED_UNITS.items()
                }
            )
            for name in (
                "constraint_length_residual",
                "constraint_tangent_residual",
                "kinetic_temperature",
            ):
                if np.any(data[name] < 0):
                    raise ValueError(f"negative constrained diagnostic: {name}")
        # Optional extension: old v1 files have unknown heat history, never zero.
        heat_method = provenance.get("thermostat_heat_method")
        expected_heat = (
            "thermal_substep_v1"
            if generic
            else CONSTRAINED_HEAT_METHOD
            if constrained
            else HEAT_METHOD
        )
        if heat_method == expected_heat:
            if "thermostat_heat" not in group:
                raise ValueError("declared OU heat column is missing")
            data["thermostat_heat"] = column("thermostat_heat", "kcal/mol")
        elif heat_method not in (None, "none") or "thermostat_heat" in group:
            raise ValueError("unsupported or undeclared thermostat heat method")
        count, mass = data["bin_count"], data["bin_mass"]
        if (
            np.any(count < 0)
            or np.any(count != np.floor(count))
            or np.any(mass < 0)
            or np.any((mass == 0) != (count == 0))
        ):
            raise ValueError("invalid bin population or mass")
        if not np.all(count.sum(axis=1) == int(provenance["atom_count"])):
            raise ValueError("profile population does not match atom count")
        for name in ("momentum", "cosine_mass", "sine_mass"):
            if np.any(data[f"bin_{name}"][mass == 0] != 0):
                raise ValueError("nonzero content in an empty bin")
        for name in ("cosine_mass", "sine_mass"):
            if np.any(np.abs(data[f"bin_{name}"]) > mass * (1 + 1e-12)):
                raise ValueError("invalid bin Fourier basis")
        if np.any(
            np.hypot(data["bin_cosine_mass"], data["bin_sine_mass"])
            > mass * (1 + 1e-12)
        ):
            raise ValueError("bin Fourier basis lies outside the unit disk")
        flow, gradient = int(provenance["flow"]), int(provenance["gradient"])
        if (
            flow not in (0, 1, 2)
            or gradient not in (0, 1, 2)
            or flow == gradient
        ):
            raise ValueError("invalid flow axes")
        total_mass = mass.sum(axis=1)
        if not np.allclose(total_mass, total_mass[0], rtol=1e-12, atol=0):
            raise ValueError("total mass changes")
        if not np.allclose(
            data["bin_momentum"].sum(axis=1) / total_mass,
            data[f"com_v{'xyz'[flow]}"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("bin momentum and COM disagree")
        if not np.allclose(
            data["bin_cosine_mass"].sum(axis=1) / total_mass,
            data["mean_cosine"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("bin basis and mean cosine disagree")
        if not np.allclose(
            data["velocity_com_fourier"],
            2 * data["variance_cosine"] * data["velocity_fit"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("inconsistent cosine estimators")
        if not np.allclose(
            data["velocity_fourier"],
            data["velocity_com_fourier"]
            + 2 * data["mean_cosine"] * data[f"com_v{'xyz'[flow]}"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("inconsistent COM correction")
        return data, provenance


def analyze(data, start_ps, block_ps, estimator, bootstrap=2000, seed=20261027):
    if estimator not in ESTIMATORS:
        raise ValueError("choose fourier, com_fourier or fit explicitly")
    times = np.asarray(data["time"], dtype=float)
    if times.ndim != 1 or len(times) < 3 or not np.all(np.isfinite(times)):
        raise ValueError("invalid time axis")
    spacing = times[1] - times[0]
    if spacing <= 0 or not np.allclose(
        np.diff(times), spacing, rtol=1e-9, atol=1e-12
    ):
        raise ValueError("uniform increasing time required")
    if not np.isfinite(start_ps) or not np.isfinite(block_ps) or block_ps <= 0:
        raise ValueError("invalid analysis window")
    for name in UNITS:
        column = np.asarray(data[name])
        if column.shape != times.shape or not np.all(np.isfinite(column)):
            raise ValueError(f"invalid scalar data: {name}")
    if "thermostat_heat" in data:
        heat = np.asarray(data["thermostat_heat"])
        if heat.shape != times.shape or not np.all(np.isfinite(heat)):
            raise ValueError("invalid thermostat heat data")
    for name in ("density", "wave_number", "length", "acceleration"):
        column = np.asarray(data[name])
        if not np.allclose(column, column[0], rtol=1e-12, atol=0):
            raise ValueError("constant density, box and acceleration required")
        if (name == "acceleration" and column[0] == 0) or (
            name != "acceleration" and column[0] <= 0
        ):
            raise ValueError(
                "positive geometry/density and nonzero acceleration required"
            )
    if not np.isclose(
        data["wave_number"][0] * data["length"][0], 2 * np.pi, rtol=1e-12
    ):
        raise ValueError("wavelength does not match periodic box")
    if (
        np.any(data["thermal_temperature"] < 0)
        or np.any(data["variance_cosine"] <= 1e-12)
        or np.any(data["variance_cosine"] > 1 + 1e-12)
    ):
        raise ValueError("invalid thermal temperature or cosine variance")
    first = int(np.searchsorted(times, start_ps))
    samples = int(np.floor(block_ps / spacing + 1e-9))
    blocks = (len(times) - first) // max(samples, 1)
    if (
        samples < 1
        or blocks < 2
        or bootstrap < 100
        or int(bootstrap) != bootstrap
    ):
        raise ValueError(
            "need two complete blocks and at least 100 bootstrap draws"
        )
    end = first + blocks * samples
    window = slice(first, end)
    # Disjoint equally weighted sample blocks: no repeated endpoint samples.
    amplitudes = (
        np.asarray(data["velocity_" + estimator])[window]
        .reshape(blocks, samples)
        .mean(axis=1)
    )
    acceleration = float(data["acceleration"][0])
    sign = np.sign(acceleration)
    signed_mean = float(sign * amplitudes.mean())
    # g/cm^3 * nm/ps^2 / (nm^-2 * nm/ps) is exactly mPa s.
    prefactor = float(
        data["density"][0] * abs(acceleration) / data["wave_number"][0] ** 2
    )
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            sign * rng.choice(amplitudes, size=blocks, replace=True).mean()
            for _ in range(bootstrap)
        ]
    )
    amplitude_interval = np.quantile(draws, [0.025, 0.975])
    resolved = signed_mean > 0 and amplitude_interval[0] > 0
    # Invert the interval for the mean, not an average of instantaneous ratios.
    # A zero crossing has an unbounded reciprocal; never remove such draws.
    eta = prefactor / signed_mean if signed_mean > 0 else None
    interval = (
        (prefactor / amplitude_interval[::-1]).tolist() if resolved else None
    )
    mass = np.asarray(data["bin_mass"])
    if (
        mass.ndim != 2
        or mass.shape[0] != len(times)
        or mass.shape[1] < 4
        or np.any(mass < 0)
        or not np.all(np.isfinite(mass))
    ):
        raise ValueError("invalid profile mass")
    averaged = {}
    for name in ("mass", "momentum", "cosine_mass", "sine_mass"):
        values = np.asarray(data["bin_" + name])
        if values.shape != mass.shape or not np.all(np.isfinite(values)):
            raise ValueError("invalid profile shape or values")
        averaged[name] = values[window].mean(axis=0)
    occupied = averaged["mass"] > 0
    if np.count_nonzero(occupied) < 4:
        raise ValueError(
            "at least four occupied bins required for spatial diagnostics"
        )
    velocity, cosine, sine = [
        averaged[name][occupied] / averaged["mass"][occupied]
        for name in ("momentum", "cosine_mass", "sine_mass")
    ]
    design = np.column_stack((np.ones(len(velocity)), cosine, sine))
    weights = np.sqrt(
        averaged["mass"][occupied] / averaged["mass"][occupied].sum()
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None], velocity * weights, rcond=None
    )
    if rank != 3:
        raise ValueError("spatial cosine/sine design is singular")
    residual = velocity - design @ coefficients
    rms = float(np.sqrt(np.sum(weights**2 * residual**2)))
    scale = float(np.hypot(coefficients[1], coefficients[2]))
    profile = []
    index = 0
    for bin_id in range(mass.shape[1]):
        present = occupied[bin_id]
        profile.append(
            dict(
                bin=bin_id,
                center_nm=float(
                    (bin_id + 0.5) * data["length"][0] / mass.shape[1]
                ),
                mean_mass_da=float(averaged["mass"][bin_id]),
                velocity_nm_per_ps=float(velocity[index]) if present else None,
                mean_cosine=float(cosine[index]) if present else None,
                mean_sine=float(sine[index]) if present else None,
                residual_nm_per_ps=float(residual[index]) if present else None,
            )
        )
        index += int(present)
    records = [
        dict(
            first_time_ps=float(times[first + i * samples]),
            last_time_ps=float(times[first + (i + 1) * samples - 1]),
            amplitude_nm_per_ps=float(amplitudes[i]),
            thermal_temperature_k=float(
                np.mean(
                    data["thermal_temperature"][
                        first + i * samples : first + (i + 1) * samples
                    ]
                )
            ),
            fitted_amplitude_nm_per_ps=float(
                np.mean(
                    data["velocity_fit"][
                        first + i * samples : first + (i + 1) * samples
                    ]
                )
            ),
        )
        for i in range(blocks)
    ]
    warnings = [
        "Diagnostic only: steady state, weak drive, block independence and convergence are not established.",
        "The conditional interval resamples whole amplitude blocks and inverts the interval for their mean; no zero or wrong-sign draws are discarded.",
        "Fourier, COM-subtracted Fourier and fitted amplitude differ at finite particle count. Compare their sensitivity; do not select by agreement with experiment.",
        "Profile fits use actual mass-averaged sine/cosine basis, not bin-center approximations.",
        "Uncertainty excludes force-field, finite-size, wave-number, timestep and thermostat bias.",
    ]
    if blocks < 10:
        warnings.append(
            "Fewer than ten blocks: interval coverage is especially unreliable."
        )
    if not resolved:
        warnings.append(
            "Mean response or its 95% interval reaches zero/wrong sign: viscosity interval is unresolved and unbounded, reported as null."
        )
    report = dict(
        status="diagnostic_only" if resolved else "unresolved_response",
        estimator=estimator,
        viscosity_mpa_s=eta,
        conditional_bootstrap_95_interval_mpa_s=interval,
        raw_mean_amplitude_nm_per_ps=float(amplitudes.mean()),
        signed_mean_amplitude_nm_per_ps=signed_mean,
        signed_amplitude_95_interval_nm_per_ps=amplitude_interval.tolist(),
        nonpositive_bootstrap_draws=int(np.count_nonzero(draws <= 0)),
        block_standard_error_nm_per_ps=float(
            amplitudes.std(ddof=1) / np.sqrt(blocks)
        ),
        blocks=blocks,
        samples_per_block=samples,
        block_records=records,
        first_time_ps=float(times[first]),
        last_time_ps=float(times[end - 1]),
        requested_start_ps=start_ps,
        requested_block_ps=block_ps,
        actual_block_ps=float(samples * spacing),
        sample_ps=float(spacing),
        skipped_initial_frames=first,
        dropped_tail_frames=len(times) - end,
        bootstrap_draws=bootstrap,
        seed=seed,
        mean_estimators_nm_per_ps={
            key: float(np.mean(data["velocity_" + key][window]))
            for key in ESTIMATORS
        },
        spatial_fit=dict(
            offset_nm_per_ps=float(coefficients[0]),
            cosine_nm_per_ps=float(coefficients[1]),
            sine_nm_per_ps=float(coefficients[2]),
            residual_rms_nm_per_ps=rms,
            relative_residual_rms=rms / scale if scale else None,
            phase_rad=float(np.arctan2(coefficients[2], coefficients[1])),
        ),
        mean_thermal_temperature_k=float(
            np.mean(data["thermal_temperature"][window])
        ),
        mean_com_nm_per_ps=[
            float(np.mean(data["com_v" + axis][window])) for axis in "xyz"
        ],
        mean_cosine=float(np.mean(data["mean_cosine"][window])),
        mean_cosine_variance=float(np.mean(data["variance_cosine"][window])),
        work_in_window_kcal_per_mol=float(
            data["coupling_work"][end - 1] - data["coupling_work"][first]
        ),
        warnings=warnings,
    )
    report["mean_response_per_acceleration_ps"] = float(
        amplitudes.mean() / acceleration
    )
    report["cosine_mode_peak_shear_rate_ps_inv"] = float(
        data["wave_number"][0] * abs(np.mean(data["velocity_fit"][window]))
    )
    report["temperature_block_standard_error_k"] = float(
        np.std([record["thermal_temperature_k"] for record in records], ddof=1)
        / np.sqrt(blocks)
    )
    report["sample_weight_duration_ps"] = float((end - first) * spacing)
    report["ledger_endpoint_duration_ps"] = float(times[end - 1] - times[first])
    report["heat_available"] = "thermostat_heat" in data
    report["heat_in_window_kcal_per_mol"] = (
        float(data["thermostat_heat"][end - 1] - data["thermostat_heat"][first])
        if "thermostat_heat" in data
        else None
    )
    report["work_plus_heat_in_window_kcal_per_mol"] = (
        report["work_in_window_kcal_per_mol"]
        + report["heat_in_window_kcal_per_mol"]
        if report["heat_available"]
        else None
    )
    if report["heat_available"]:
        warnings.append(
            "Work plus heat is supplied energy, not an energy-balance residual: compare with same-time internal energy and finite-step discretization error separately."
        )
    if "constraint_drift_energy" in data:
        for name in CONSTRAINED_UNITS:
            values = np.asarray(data[name])
            if values.shape != times.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid constrained diagnostic: {name}")
        report["constrained_diagnostics"] = dict(
            drift_kinetic_change_in_window_kcal_per_mol=float(
                data["constraint_drift_energy"][end - 1]
                - data["constraint_drift_energy"][first]
            ),
            max_relative_length_residual=float(
                np.max(data["constraint_length_residual"][window])
            ),
            max_tangent_residual_nm_per_ps=float(
                np.max(data["constraint_tangent_residual"][window])
            ),
            mean_physical_kinetic_temperature_k=float(
                np.mean(data["kinetic_temperature"][window])
            ),
            mean_pressure_pa={
                name: float(np.mean(data[name][window]))
                for name in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")
            },
        )
        warnings.append(
            "Constrained diagnostics do not establish dynamic ensemble or viscosity convergence; drift kinetic change is not thermostat heat."
        )
    return report, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--start-ps", type=float, required=True)
    parser.add_argument("--block-ps", type=float, required=True)
    parser.add_argument("--estimator", choices=ESTIMATORS, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, metadata = load_profile(args.input)
    report, profile = analyze(
        data,
        args.start_ps,
        args.block_ps,
        args.estimator,
        args.bootstrap,
        args.seed,
    )
    report.update(
        input=str(args.input.resolve()),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        provenance=metadata,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "profile.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(profile[0]))
        writer.writeheader()
        writer.writerows(profile)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "viscosity_mpa_s",
                    "conditional_bootstrap_95_interval_mpa_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
