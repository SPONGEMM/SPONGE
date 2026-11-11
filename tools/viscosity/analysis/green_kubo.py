#!/usr/bin/env python3
"""Explicit-window Green--Kubo diagnostics for finalized SPONGE stress files.

Requires NumPy and h5py. A finite estimate/interval is not a convergence test.
The isotropic equilibrium liquid assumption and block independence are external
requirements. Never concatenate unrelated trajectories as time origins.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

KB = 1.380649e-23  # exact SI J/K
CONTRACT = "sponge.green_kubo.centered_nve.v1"
CONSTRAINED_CONTRACT = "sponge.green_kubo.constrained_nve.v1"
CONSTRAINT_UNITS = {
    "constraint_length_residual": "1",
    "constraint_tangent_residual": "nm/ps",
    "constraint_acceleration_residual": "nm/ps^2",
    **{
        f"constraint_w{x}": "kcal/mol"
        for x in ("xx", "yy", "zz", "xy", "xz", "yz")
    },
}
NAMES = ("pxy", "pxz", "pyz")
COMPONENTS = {
    "shear3": NAMES,
    "isotropic5": NAMES + ("half_pxx_minus_pyy", "half_pyy_minus_pzz"),
}


def component_names(components):
    if components not in COMPONENTS:
        raise ValueError("components must be shear3 or isotropic5")
    return COMPONENTS[components]


def stress_components(tensor):
    """Five equally normalized shear observables for an isotropic liquid.

    tensor has columns xx, yy, zz, xy, xz, yz (Pa). The two normal-stress
    differences are correlated; these five columns are not five replicates.
    Their equal weighting is an isotropic-ensemble estimator, not a
    rotationally invariant contraction for an individual finite trajectory.
    """
    tensor = np.asarray(tensor, dtype=float)
    if (
        tensor.ndim != 2
        or tensor.shape[1] != 6
        or not np.all(np.isfinite(tensor))
    ):
        raise ValueError("expected finite stress tensor with six columns")
    return np.column_stack(
        (
            tensor[:, 3:],
            (tensor[:, 0] - tensor[:, 1]) * 0.5,
            (tensor[:, 1] - tensor[:, 2]) * 0.5,
        )
    )


def text(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar metadata value")
    value = array.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def load_stress(path, components="shear3"):
    """Reject partial/failed streams, unequal lengths and ambiguous units."""
    component_names(components)
    input_names = (
        NAMES if components == "shear3" else ("pxx", "pyy", "pzz") + NAMES
    )
    with h5py.File(path, "r") as file:
        provenance = file["parameters/sponge/provenance"]
        generic = "observable_contract" in provenance
        if generic:
            if (
                text(provenance["observable_contract"][()])
                != "sponge.observables.physical_state.v1"
            ):
                raise ValueError("unsupported observable contract")
            if text(provenance["velocity_time_layer"][()]) != "force_centered":
                raise ValueError("GK workflow requires force-centered velocity")
            commands = dict(
                line.split("=", 1)
                for line in text(
                    provenance["effective_commands"][()]
                ).splitlines()
            )
            if commands.get("mode", "").lower() != "nve":
                raise ValueError(
                    "GK workflow requires an equilibrium NVE segment"
                )
            if any(
                key in commands
                for key in (
                    "external_force",
                    "momentum_exchange_interval",
                    "velocity_bias_mode",
                )
            ):
                raise ValueError("GK workflow rejects driven segments")
            contract = (
                CONSTRAINED_CONTRACT
                if "constraint_observer" in provenance
                else CONTRACT
            )
        else:
            contract = text(provenance["viscosity_contract"][()])
        if contract not in (CONTRACT, CONSTRAINED_CONTRACT):
            raise ValueError("unsupported viscosity contract")
        output = file["parameters/sponge/output"]
        if text(output["status"][()]) != "finalized":
            raise ValueError("stress file is not finalized")
        group = file["observables/all"]
        times = np.asarray(group["time"], dtype=float)
        steps = np.asarray(group["step"])
        n = len(times)
        if times.ndim != 1 or steps.shape != times.shape or n < 4:
            raise ValueError("invalid stress step/time shape or too few frames")
        if text(group["time"].attrs.get("unit", "")) != "ps":
            raise ValueError("time unit must be ps")
        values = []
        for name, unit in [(x, "Pa") for x in input_names] + [
            ("temperature", "K"),
            ("volume", "nm^3"),
        ]:
            dataset = group[f"{name}/value"]
            array = np.asarray(dataset, dtype=float)
            if array.shape != times.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"invalid shape/nonfinite values: {name}")
            if text(dataset.attrs.get("unit", "")) != unit:
                raise ValueError(f"invalid unit: {name}")
            values.append(array)
        counts = np.asarray(output["frame_count"])
        complete_steps = np.asarray(output["last_complete_step"])
        complete_times = np.asarray(output["last_complete_time"])
        if (
            counts.ndim != 1
            or not len(counts)
            or counts.shape != complete_steps.shape
            or counts.shape != complete_times.shape
        ):
            raise ValueError("invalid completion journal")
        if not np.issubdtype(counts.dtype, np.integer) or not np.issubdtype(
            steps.dtype, np.integer
        ):
            raise ValueError("integer step/count datasets required")
        if (
            not np.all(np.isfinite(complete_times))
            or np.any(counts < 0)
            or np.any(counts > n)
            or np.any(np.diff(counts) <= 0)
        ):
            raise ValueError("invalid completion journal counts/times")
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
        if int(counts[-1]) != n:
            raise ValueError("completion frame count does not match datasets")
        if (
            int(complete_steps[-1]) != steps[-1]
            or float(complete_times[-1]) != times[-1]
        ):
            raise ValueError("completion endpoint does not match datasets")
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
            raise ValueError("time must be finite and strictly increasing")
        dt = float(text(provenance["dt_ps"][()]))
        stride = int(text(provenance["sample_interval"][()]))
        if not np.isfinite(dt) or dt <= 0 or stride < 1:
            raise ValueError("positive dt and stride required for correlation")
        if not np.all(np.diff(steps) == stride) or not np.allclose(
            np.diff(times), dt * stride, rtol=1e-9, atol=1e-12
        ):
            raise ValueError("nonuniform sampling or metadata mismatch")
        stress = np.column_stack(values[:-2])
        if components == "isotropic5":
            stress = stress_components(stress)
        temperature, volume = values[-2:]
        if np.any(temperature <= 0) or np.any(volume <= 0):
            raise ValueError("positive temperature and volume required")
        if not np.allclose(volume, volume[0], rtol=1e-8, atol=0):
            raise ValueError("fixed volume required")
        metadata = {name: text(value[()]) for name, value in provenance.items()}
        if contract == CONSTRAINED_CONTRACT:
            required = {
                "constraint_observer",
                "constraint_count",
                "thermal_dof",
                "atom_count",
                "constraint_solver",
                "constraint_length_tolerance",
            }
            if not required.issubset(metadata):
                raise ValueError("missing constrained GK metadata")
            if metadata["constraint_observer"] != "instantaneous_tangent_v1":
                raise ValueError("unsupported constraint observer")
            if metadata["constraint_solver"].split("|", 1)[0] not in (
                "SETTLE",
                "SHAKE",
                "RATTLE",
            ):
                raise ValueError("unsupported constrained GK solver")
            atoms, constraints, dof = [
                int(metadata[key])
                for key in ("atom_count", "constraint_count", "thermal_dof")
            ]
            tolerance = float(metadata["constraint_length_tolerance"])
            if (
                atoms < 2
                or constraints < 1
                or dof <= 0
                or dof != 3 * atoms - constraints - 3
            ):
                raise ValueError("invalid constrained GK thermal DOF")
            parts = metadata["constraint_solver"].split("|")
            try:
                fields = dict(part.split("=", 1) for part in parts[1:])
                if len(fields) != len(parts) - 1:
                    raise ValueError("duplicate solver fields")
                if parts[0] == "RATTLE":
                    if (
                        set(fields) != {"rattle_pairs", "precision", "step"}
                        or int(fields["rattle_pairs"]) != constraints
                        or fields["precision"]
                        != "canonical_double_native_force_v1"
                        or fields["step"] != "BA_full_drift"
                    ):
                        raise ValueError("invalid RATTLE state/step metadata")
                else:
                    counts = [
                        int(fields[key])
                        for key in (
                            "settle_pairs",
                            "settle_triangles",
                            "shake_pairs",
                        )
                    ]
                    if (
                        min(counts) < 0
                        or counts[0] + 3 * counts[1] + counts[2] != constraints
                    ):
                        raise ValueError(
                            "inconsistent solver constraint counts"
                        )
                    keys = {"settle_pairs", "settle_triangles", "shake_pairs"}
                    # Legacy v1 files predate explicit solver precision.
                    if "precision" in fields:
                        if fields["precision"] not in ("single", "double"):
                            raise ValueError("invalid constraint precision")
                        keys.add("precision")
                    if counts[2]:
                        if parts[0] != "SHAKE":
                            raise ValueError("SETTLE cannot leave SHAKE pairs")
                        keys.update(("iterations", "step_length", "tolerance"))
                        iteration = int(fields["iterations"])
                        step, tol = (
                            float(fields["step_length"]),
                            float(fields["tolerance"]),
                        )
                        if (
                            iteration <= 0
                            or not np.isfinite(step)
                            or step <= 0
                            or not 0 < tol < 1
                        ):
                            raise ValueError("invalid SHAKE settings")
                    if set(fields) != keys:
                        raise ValueError("unexpected solver fields")
            except (ValueError, KeyError) as error:
                raise ValueError(
                    "invalid constrained GK solver metadata"
                ) from error
            if tolerance != (2e-12 if parts[0] == "RATTLE" else 1e-4):
                raise ValueError("unsupported constraint length tolerance")
            for name, unit in CONSTRAINT_UNITS.items():
                if f"{name}/value" not in group:
                    raise ValueError(f"missing constraint diagnostic: {name}")
                dataset = group[f"{name}/value"]
                array = np.asarray(dataset, dtype=float)
                if array.shape != times.shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"invalid constraint diagnostic: {name}")
                if text(dataset.attrs.get("unit", "")) != unit:
                    raise ValueError(
                        f"invalid constraint diagnostic unit: {name}"
                    )
                if name.endswith("residual") and np.any(array < 0):
                    raise ValueError(f"negative constraint residual: {name}")
                if name == "constraint_length_residual" and np.any(
                    array > tolerance
                ):
                    raise ValueError(
                        "constraint length residual exceeds tolerance"
                    )
        return times, stress, temperature, volume, metadata


def autocorrelation(samples, max_lag, demean=False):
    """Unbiased multiple-origin estimate: denominator N-lag, each component."""
    samples = np.asarray(samples, dtype=float)
    if (
        samples.ndim != 2
        or not np.all(np.isfinite(samples))
        or not 0 <= max_lag < len(samples)
    ):
        raise ValueError("invalid stress samples or lag")
    data = samples - samples.mean(axis=0) if demean else samples
    n = len(data)
    length = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(data, n=length, axis=0)
    products = np.fft.irfft(spectrum.conj() * spectrum, n=length, axis=0)[
        : max_lag + 1
    ]
    return products / np.arange(n, n - max_lag - 1, -1)[:, None]


def integrate(correlation, sample_ps, volume_nm3, temperature_k):
    factor = volume_nm3 * 1e-27 / (KB * temperature_k) * sample_ps * 1e-12
    result = np.zeros_like(correlation)
    result[1:] = (
        np.cumsum((correlation[1:] + correlation[:-1]) * 0.5, axis=0) * factor
    )
    return result  # Pa s


def analyze(
    times,
    stress,
    temperature,
    volume,
    start_ps,
    window_ps,
    block_ps,
    demean=False,
    bootstrap=2000,
    seed=20261027,
    components="shear3",
):
    names = component_names(components)
    times, stress, temperature, volume = [
        np.asarray(value, dtype=float)
        for value in (times, stress, temperature, volume)
    ]
    if (
        times.ndim != 1
        or len(times) < 2
        or stress.shape != (len(times), len(names))
        or temperature.shape != times.shape
        or volume.shape != times.shape
    ):
        raise ValueError("incompatible time/stress/temperature/volume shapes")
    if not all(
        np.all(np.isfinite(value))
        for value in (times, stress, temperature, volume)
    ):
        raise ValueError("nonfinite analysis inputs")
    differences = np.diff(times)
    if np.any(differences <= 0) or not np.allclose(
        differences, differences[0], rtol=1e-9, atol=1e-12
    ):
        raise ValueError("uniform strictly increasing time required")
    if (
        np.any(temperature <= 0)
        or np.any(volume <= 0)
        or not np.allclose(volume, volume[0], rtol=1e-8, atol=0)
    ):
        raise ValueError(
            "positive temperature and fixed positive volume required"
        )
    if (
        not all(np.isfinite(x) for x in (start_ps, window_ps, block_ps))
        or window_ps <= 0
        or block_ps <= 0
    ):
        raise ValueError(
            "finite start and positive window/block lengths required"
        )
    spacing = times[1] - times[0]
    lag = int(np.floor(window_ps / spacing + 1e-9))
    block_size = int(np.floor(block_ps / spacing + 1e-9))
    if lag < 1 or block_size < 2 * lag + 1:
        raise ValueError(
            "need at least one lag and block samples >= 2*max_lag+1"
        )
    begin = int(np.searchsorted(times, start_ps))
    blocks = (len(times) - begin) // block_size
    if blocks < 2 or bootstrap < 100:
        raise ValueError(
            "need at least two complete blocks and 100 bootstrap draws"
        )
    end = begin + blocks * block_size
    temperatures = temperature[begin:end]
    mean_temperature = float(temperatures.mean())
    mean_volume = float(volume[begin:end].mean())
    data = stress[begin:end].reshape(blocks, block_size, len(names))
    correlations = np.stack(
        [autocorrelation(block, lag, demean) for block in data]
    )
    correlation = correlations.mean(axis=0)
    integrals = np.stack(
        [
            integrate(c, spacing, mean_volume, mean_temperature)
            for c in correlations
        ]
    )
    if not np.all(np.isfinite(integrals)):
        raise ValueError("nonfinite correlation integrals; inspect input scale")
    curves = integrals.mean(axis=0)
    # Average correlated tensor components BEFORE resampling blocks.
    estimates = integrals[:, -1, :].mean(axis=1)
    rng = np.random.default_rng(seed)
    # Draw in a loop to keep memory bounded independently of trajectory length.
    resampled = np.array(
        [
            rng.choice(estimates, size=blocks, replace=True).mean()
            for _ in range(bootstrap)
        ]
    )
    ci = np.quantile(resampled, [0.025, 0.975])
    means = data.mean(axis=(0, 1))
    rms = np.sqrt((data**2).mean(axis=(0, 1)))
    warnings = [
        "No convergence or method-equivalence claim. Check window, block length, sampling interval, timestep and system size independently.",
        "Percentile interval is conditional on approximately independent blocks; it excludes tail truncation, temperature estimation and model error.",
        "All stress components are averaged within each block, not treated as independent replicates.",
    ]
    if components == "isotropic5":
        warnings.append(
            "Five-component equal weighting assumes isotropy; normal-stress differences are correlated and do not provide independent replicates."
        )
    if blocks < 10:
        warnings.append(
            "Fewer than ten blocks: bootstrap coverage is especially unreliable."
        )
    if demean:
        warnings.append(
            "Mean stress removed separately in each block; finite-block demeaning can bias long-time correlations downward."
        )
    else:
        warnings.append(
            "Uses the theoretical zero mean shear stress of an isotropic equilibrium liquid; inspect mean-stress diagnostics."
        )
    if float(np.std(temperatures) / mean_temperature) > 0.1:
        warnings.append(
            "Temperature varies by more than 10% RMS: inspect equilibration, finite-size fluctuations and energy drift."
        )
    report = dict(
        status="diagnostic_only",
        estimator="mean of equal-length block GK integrals; mean of "
        + ",".join(names),
        components=components,
        component_names=list(names),
        viscosity_pa_s=float(estimates.mean()),
        block_standard_error_pa_s=float(
            estimates.std(ddof=1) / np.sqrt(blocks)
        ),
        conditional_bootstrap_95_interval_pa_s=ci.tolist(),
        component_viscosity_pa_s=curves[-1].tolist(),
        block_viscosity_pa_s=estimates.tolist(),
        blocks=blocks,
        samples_per_block=block_size,
        used_frames=end - begin,
        skipped_initial_frames=begin,
        dropped_tail_frames=len(times) - end,
        first_time_ps=float(times[begin]),
        last_time_ps=float(times[end - 1]),
        sample_ps=float(spacing),
        requested_window_ps=window_ps,
        actual_window_ps=float(lag * spacing),
        requested_block_ps=block_ps,
        temperature_k=mean_temperature,
        volume_nm3=mean_volume,
        stress_mean_pa=means.tolist(),
        stress_rms_pa=rms.tolist(),
        demean_per_block=demean,
        bootstrap_draws=bootstrap,
        seed=seed,
        warnings=warnings,
    )
    return report, np.arange(lag + 1) * spacing, correlation, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--start-ps",
        type=float,
        required=True,
        help="explicit equilibration cutoff",
    )
    parser.add_argument("--window-ps", type=float, required=True)
    parser.add_argument("--block-ps", type=float, required=True)
    parser.add_argument("--demean", action="store_true")
    parser.add_argument(
        "--components",
        choices=tuple(COMPONENTS),
        default="shear3",
        help="isotropic5 adds (Pxx-Pyy)/2 and (Pyy-Pzz)/2",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for JSON and CSV",
    )
    args = parser.parse_args()
    times, stress, temperature, volume, metadata = load_stress(
        args.input, args.components
    )
    report, lags, correlation, curves = analyze(
        times,
        stress,
        temperature,
        volume,
        args.start_ps,
        args.window_ps,
        args.block_ps,
        args.demean,
        args.bootstrap,
        args.seed,
        args.components,
    )
    hasher = hashlib.sha256()
    with args.input.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    report.update(
        input=str(args.input.resolve()),
        input_sha256=digest,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,
        h5py_version=h5py.__version__,
        provenance=metadata,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "correlation.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        names = component_names(args.components)
        writer.writerow(
            [
                "lag_ps",
                *(f"c_{name}_pa2" for name in names),
                *(f"eta_{name}_pa_s" for name in names),
                "eta_mean_pa_s",
            ]
        )
        writer.writerows(
            np.column_stack((lags, correlation, curves, curves.mean(axis=1)))
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "viscosity_pa_s",
                    "conditional_bootstrap_95_interval_pa_s",
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
        raise SystemExit(f"Green-Kubo analysis rejected: {error}") from error
