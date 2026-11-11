#!/usr/bin/env python3
"""Explicit-window RNEMD diagnostics; finite estimates do not prove steady state.

Analyze finalized SPONGE centered-NVE files. Fit both periodic branches separately,
exclude exchange slabs, and resample whole time blocks to preserve covariance.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

CONTRACT = "sponge.rnemd.centered_nve.v1"
DA_KG = 1.66053906660e-27


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
        if (
            generic
            and (
                provenance["observable_contract"]
                != "sponge.observables.physical_state.v1"
                or provenance.get("profile_kind") != "slab"
                or "exchange_interval" not in provenance
            )
        ) or (not generic and provenance["viscosity_contract"] != CONTRACT):
            raise ValueError("unsupported RNEMD contract")
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
        if bins < 4 or bins % 2 or bins > 10000:
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

        data = {"time": times}
        for key, unit in {
            "mass": "Da",
            "momentum": "Da nm/ps",
            "count": "1",
        }.items():
            data[f"bin_{key}"] = np.column_stack(
                [column(f"bin_{i}_{key}", unit) for i in range(bins)]
            )
        units = {
            "momentum": "Da nm/ps",
            "momentum_error": "Da nm/ps",
            "energy_error": "kcal/mol",
            "area": "nm^2",
            "length": "nm",
            "thermal_temperature": "K",
            "kinetic_temperature": "K",
        }
        units.update(
            {
                name: "1"
                for name in ("attempts", "swaps", "empty", "no_candidate")
            }
        )
        data.update({name: column(name, unit) for name, unit in units.items()})
        count, mass, momentum = (
            data[f"bin_{key}"] for key in ("count", "mass", "momentum")
        )
        if (
            np.any(count < 0)
            or np.any(count != np.floor(count))
            or np.any(mass < 0)
            or np.any((mass == 0) != (count == 0))
            or np.any(momentum[mass == 0] != 0)
        ):
            raise ValueError("invalid bin population/mass/momentum")
        if not np.all(count.sum(axis=1) == int(provenance["atom_count"])):
            raise ValueError("profile population does not match atom count")
        for name in ("attempts", "swaps", "empty", "no_candidate"):
            values = data[name]
            if (
                np.any(values < 0)
                or np.any(values >= 2**53)
                or np.any(values != np.floor(values))
                or np.any(np.diff(values) < 0)
            ):
                raise ValueError("invalid cumulative exchange counters")
        if not np.array_equal(
            data["attempts"],
            data["swaps"] + data["empty"] + data["no_candidate"],
        ):
            raise ValueError("inconsistent exchange counters")
        if np.any(data["momentum"] < 0) or np.any(
            np.diff(data["momentum"]) < 0
        ):
            raise ValueError("invalid cumulative momentum")
        if np.any(data["thermal_temperature"] < 0) or np.any(
            data["kinetic_temperature"] < 0
        ):
            raise ValueError("invalid temperature")
        return data, provenance


def analyze(
    data, start_ps, block_ps, exclude_neighbors=0, bootstrap=2000, seed=20261027
):
    times = np.asarray(data["time"], dtype=float)
    mass, momentum = (
        np.asarray(data[f"bin_{key}"], dtype=float)
        for key in ("mass", "momentum")
    )
    if (
        times.ndim != 1
        or len(times) < 3
        or mass.ndim != 2
        or mass.shape[0] != len(times)
        or momentum.shape != mass.shape
    ):
        raise ValueError("incompatible profile shapes")
    for value in (
        times,
        mass,
        momentum,
        data["momentum"],
        data["area"],
        data["length"],
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError("nonfinite profile input")
    spacing = times[1] - times[0]
    if spacing <= 0 or not np.allclose(
        np.diff(times), spacing, rtol=1e-9, atol=1e-12
    ):
        raise ValueError("uniform increasing time required")
    if (
        not np.isfinite(start_ps)
        or not np.isfinite(block_ps)
        or block_ps <= 0
        or exclude_neighbors < 0
        or int(exclude_neighbors) != exclude_neighbors
    ):
        raise ValueError("invalid analysis window or excluded neighbors")
    counters = ("attempts", "swaps", "empty", "no_candidate")
    for key in (
        *counters,
        "momentum_error",
        "energy_error",
        "thermal_temperature",
        "kinetic_temperature",
    ):
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=float)
        if values.shape != times.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid optional column: {key}")
        if key in counters and (
            np.any(values < 0)
            or np.any(values >= 2**53)
            or np.any(values != np.floor(values))
            or np.any(np.diff(values) < 0)
        ):
            raise ValueError("invalid exchange counters")
        if key.endswith("temperature") and np.any(values < 0):
            raise ValueError("invalid temperature")
    if all(key in data for key in counters) and not np.array_equal(
        data["attempts"],
        np.asarray(data["swaps"]) + data["empty"] + data["no_candidate"],
    ):
        raise ValueError("inconsistent exchange counters")
    bins = mass.shape[1]
    if bins < 4 or bins % 2:
        raise ValueError("even bin count >= 4 required")
    for key in ("area", "length"):
        values = np.asarray(data[key])
        if (
            values.shape != times.shape
            or np.any(values <= 0)
            or not np.allclose(values, values[0], rtol=1e-8, atol=0)
        ):
            raise ValueError("fixed positive box geometry required")
    cumulative = np.asarray(data["momentum"])
    if (
        cumulative.shape != times.shape
        or np.any(cumulative < 0)
        or np.any(np.diff(cumulative) < 0)
        or np.any(mass < 0)
    ):
        raise ValueError("invalid cumulative momentum or bin mass")
    half = bins // 2
    branches = [
        np.arange(1 + exclude_neighbors, half - exclude_neighbors),
        np.arange(half + 1 + exclude_neighbors, bins - exclude_neighbors),
    ]
    if min(map(len, branches)) < 2:
        raise ValueError(
            "at least two interior bins per branch required after exclusion"
        )
    intervals = int(np.floor(block_ps / spacing + 1e-9))
    begin = int(np.searchsorted(times, start_ps))
    blocks = (len(times) - 1 - begin) // max(intervals, 1)
    if intervals < 1 or blocks < 2 or bootstrap < 100:
        raise ValueError(
            "need positive block intervals, two complete blocks and 100 bootstrap draws"
        )
    z = (np.arange(bins) + 0.5) * float(data["length"][0]) / bins
    area = float(data["area"][0])
    # Resample extensive block averages jointly, before division or spatial fits.
    # Equal-duration blocks reconstruct the full-window trapezoid exactly.
    used = np.concatenate(branches)
    designs = [np.column_stack((z[idx], np.ones(len(idx)))) for idx in branches]
    inverses = [np.linalg.pinv(design) for design in designs]
    signs = np.array([1.0, -1.0])

    def fit_profile(avg_mass, avg_momentum):
        velocity = np.divide(
            avg_momentum,
            avg_mass,
            out=np.full(bins, np.nan),
            where=avg_mass > 0,
        )
        if np.any(avg_mass[used] <= 0):
            return velocity, None, None
        slopes, details = [], []
        for indices, design, inverse in zip(branches, designs, inverses):
            slope, intercept = inverse @ velocity[indices]
            residual_rms = float(
                np.sqrt(
                    np.mean(
                        (velocity[indices] - design @ [slope, intercept]) ** 2
                    )
                )
            )
            span = abs(slope * (z[indices[-1]] - z[indices[0]]))
            slopes.append(slope)
            details.append(
                dict(
                    slope_per_ps=float(slope),
                    intercept_nm_per_ps=float(intercept),
                    residual_rms_nm_per_ps=residual_rms,
                    relative_residual_rms=residual_rms / span
                    if span > 0
                    else None,
                    bins=indices.tolist(),
                )
            )
        return velocity, np.array(slopes), details

    def branch_ratios(slopes, flux):
        if slopes is None or flux <= 0:
            return [None, None]
        return [
            float(flux / (signed * 1e12)) if signed > 0 else None
            for signed in signs * slopes
        ]

    records, profiles, block_mass, block_momentum, impulses = [], [], [], [], []
    weights = np.ones(intervals + 1)
    weights[[0, -1]] = 0.5
    duration = float(intervals * spacing)
    flux_factor = DA_KG * 1e3 / (2 * area * 1e-18 * duration * 1e-12)
    for block in range(blocks):
        first, last = begin + block * intervals, begin + (block + 1) * intervals
        avg_mass = weights @ mass[first : last + 1] / intervals
        avg_momentum = weights @ momentum[first : last + 1] / intervals
        impulse = float(cumulative[last] - cumulative[first])
        flux = impulse * flux_factor
        velocity, slopes, details = fit_profile(avg_mass, avg_momentum)
        block_mass.append(avg_mass)
        block_momentum.append(avg_momentum)
        impulses.append(impulse)
        records.append(
            dict(
                first_time_ps=float(times[first]),
                last_time_ps=float(times[last]),
                flux_pa=float(flux),
                momentum_da_nm_per_ps=impulse,
                branch_viscosity_pa_s=branch_ratios(slopes, flux),
                fits=details,
            )
        )
        profiles.extend(
            (
                block,
                i,
                float(z[i]),
                float(avg_mass[i]),
                float(velocity[i]) if np.isfinite(velocity[i]) else None,
            )
            for i in range(bins)
        )
    block_mass, block_momentum, impulses = map(
        np.asarray, (block_mass, block_momentum, impulses)
    )
    avg_mass, avg_momentum = (
        block_mass.mean(axis=0),
        block_momentum.mean(axis=0),
    )
    velocity, slopes, details = fit_profile(avg_mass, avg_momentum)
    flux = float(impulses.mean() * flux_factor)
    if slopes is None:
        raise ValueError("unoccupied fitted bin in full analysis window")
    if flux <= 0 or not np.isfinite(flux):
        raise ValueError("no positive momentum flux in full analysis window")
    # Fluidity = signed gradient / flux; fixed equal branch weights.
    # This is not an average of inverse noisy per-block gradients.
    fluidity = signs * slopes * 1e12 / flux
    common = float(fluidity.mean())
    branch_eta = branch_ratios(slopes, flux)
    eta = 1 / common if np.all(fluidity > 0) else None
    rng = np.random.default_rng(seed)
    draws = np.full((bootstrap, 2), np.nan)
    for draw in range(bootstrap):
        indices = rng.integers(blocks, size=blocks)
        _, sampled_slopes, _ = fit_profile(
            block_mass[indices].mean(axis=0),
            block_momentum[indices].mean(axis=0),
        )
        sampled_flux = float(impulses[indices].mean() * flux_factor)
        if sampled_slopes is not None and sampled_flux > 0:
            draws[draw] = signs * sampled_slopes * 1e12 / sampled_flux
    invalid = int(np.count_nonzero(~np.all(np.isfinite(draws), axis=1)))
    # Never drop undefined or wrong-sign bootstrap draws to get a finite interval.
    response_interval = branch_intervals = eta_interval = None
    branch_eta_intervals = [None, None]
    if invalid == 0:
        response_interval = np.quantile(
            draws.mean(axis=1), [0.025, 0.975]
        ).tolist()
        branch_intervals = np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()
        branch_eta_intervals = [
            [1 / hi, 1 / lo] if lo > 0 else None for lo, hi in branch_intervals
        ]
        if response_interval[0] > 0 and all(
            value is not None for value in branch_eta_intervals
        ):
            eta_interval = [1 / response_interval[1], 1 / response_interval[0]]
    resolved = eta is not None and eta_interval is not None
    end = begin + blocks * intervals
    warnings = [
        "Diagnostic only: steady state, weak driving, linear constitutive response and block independence are not established.",
        "Point estimate pools extensive mass/momentum and impulse over the whole window, then inverts the equally weighted signed branch fluidities.",
        "Bootstrap jointly resamples whole blocks of mass, momentum and impulse and refits; negative responses are retained. Undefined draws invalidate the interval.",
        "A finite interval requires positive lower fluidity bounds for BOTH branches and the common response. Crossing zero leaves viscosity unresolved, not clipped.",
        "Adjacent blocks share endpoint samples. Increase block length and use independent trajectories to assess dependence.",
        "Velocity is the ratio of trapezoid-averaged bin momentum and mass. Flux uses matching cumulative endpoints and TWO periodic paths.",
        "Bootstrap interval excludes force-field, finite-size, timestep, fitting-range and finite-driving bias.",
        "Per-block viscosity ratios are diagnostics only and are never averaged into the reported estimate.",
        "Skip counts include signed-candidate absence, zero impulse and exchanges unresolved at engine float precision.",
    ]
    if blocks < 10:
        warnings.append(
            "Fewer than ten blocks: interval coverage is especially unreliable."
        )
    report = dict(
        status="diagnostic_only" if resolved else "unresolved_response",
        estimator="aggregate-profile-equal-branch-fluidity.v2",
        viscosity_pa_s=eta,
        branch_viscosity_pa_s=branch_eta,
        branch_difference_pa_s=branch_eta[1] - branch_eta[0]
        if all(v is not None for v in branch_eta)
        else None,
        fluidity_per_pa_s=common,
        branch_fluidity_per_pa_s=fluidity.tolist(),
        flux_pa=flux,
        conditional_bootstrap_95_interval_pa_s=eta_interval,
        conditional_bootstrap_95_fluidity_interval_per_pa_s=response_interval,
        conditional_bootstrap_95_branch_fluidity_intervals_per_pa_s=branch_intervals,
        conditional_bootstrap_95_branch_viscosity_intervals_pa_s=branch_eta_intervals,
        bootstrap_invalid_draws=invalid,
        bootstrap_nonpositive_branch_draws=np.sum(draws <= 0, axis=0).tolist(),
        aggregate_fits=details,
        aggregate_profile=dict(
            z_nm=z.tolist(),
            mean_mass_da=avg_mass.tolist(),
            mean_momentum_da_nm_per_ps=avg_momentum.tolist(),
            velocity_nm_per_ps=[
                float(v) if np.isfinite(v) else None for v in velocity
            ],
        ),
        blocks=blocks,
        block_records=records,
        first_time_ps=float(times[begin]),
        last_time_ps=float(times[end]),
        requested_start_ps=start_ps,
        requested_block_ps=block_ps,
        actual_block_ps=duration,
        sample_ps=float(spacing),
        skipped_initial_frames=begin,
        dropped_tail_intervals=len(times) - 1 - end,
        exclude_neighbors=exclude_neighbors,
        bootstrap_draws=bootstrap,
        seed=seed,
        warnings=warnings,
    )
    for key in (
        "attempts",
        "swaps",
        "empty",
        "no_candidate",
        "momentum_error",
        "energy_error",
    ):
        if key in data:
            report[f"window_{key}"] = float(data[key][end] - data[key][begin])
    for key in ("thermal_temperature", "kinetic_temperature"):
        if key in data:
            report[f"mean_{key}_k"] = float(np.mean(data[key][begin : end + 1]))
    return report, profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--start-ps", type=float, required=True)
    parser.add_argument("--block-ps", type=float, required=True)
    parser.add_argument("--exclude-neighbors", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, metadata = load_profile(args.input)
    report, profiles = analyze(
        data,
        args.start_ps,
        args.block_ps,
        args.exclude_neighbors,
        args.bootstrap,
        args.seed,
    )
    digest = hashlib.sha256()
    with args.input.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    report.update(
        input=str(args.input.resolve()),
        input_sha256=digest.hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,
        h5py_version=h5py.__version__,
        provenance=metadata,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "profiles.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["block", "bin", "z_nm", "mean_mass_da", "velocity_nm_per_ps"]
        )
        writer.writerows(profiles)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "viscosity_pa_s",
                    "branch_viscosity_pa_s",
                    "blocks",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"RNEMD analysis rejected: {error}") from error
