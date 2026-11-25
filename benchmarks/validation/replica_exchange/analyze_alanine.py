"""Validate the temperature distributions and MBAR interpolation on saved frames."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pymbar import MBAR, timeseries
from scipy.optimize import root
from scipy.special import expit, logsumexp
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

KB = 0.00198716
REGIONS = ("alpha_R", "PPII", "beta", "positive_phi", "other")


def load(directory, burn_ps):
    meta = json.loads((directory / "study.json").read_text())
    windows = []
    for i in range(len(meta["temperatures"])):
        rows = np.loadtxt(
            directory / f"window-{i}/samples.csv", delimiter=",", skiprows=1
        )
        if not np.isfinite(rows).all() or not np.array_equal(
            rows[:, 0], np.arange(len(rows))
        ):
            raise ValueError(f"invalid frames in {directory}/window-{i}")
        windows.append(rows[rows[:, 1] > burn_ps, 2:])
    return meta, np.stack(windows)


def features(frames):
    energy, phi, psi = frames.T
    extended_psi = (psi + 120) % 360 - 120
    alpha = (phi >= -120) & (phi < -30) & (psi >= -100) & (psi < 30)
    ppii = (
        (phi >= -110)
        & (phi < -40)
        & (extended_psi >= 90)
        & (extended_psi < 220)
    )
    beta = (phi < -110) & (extended_psi >= 90) & (extended_psi < 220)
    positive = phi >= 0
    other = ~(alpha | ppii | beta | positive)
    return np.column_stack((energy, alpha, ppii, beta, positive, other))


def mbar_input(windows, temperatures):
    counts = np.array([len(window) for window in windows])
    frames = np.concatenate(windows)
    # Centering energy changes per-temperature constants but leaves weights invariant.
    energies = frames[:, 0] - np.mean(frames[:, 0])
    return (
        counts,
        frames,
        energies,
        energies[None, :] / (KB * temperatures[:, None]),
    )


def weights(f, u, counts, energies, target):
    denominator = logsumexp((np.log(counts) + f)[:, None] - u, axis=0)
    log_weights = -energies / (KB * target) - denominator
    return np.exp(log_weights - logsumexp(log_weights))


def bootstrap_fit(u, counts):
    log_counts = np.log(counts)

    def error(relative):
        f = np.r_[0.0, relative]
        denominator = logsumexp((log_counts + f)[:, None] - u, axis=0)
        update = -logsumexp(-u - denominator, axis=1)
        return f[1:] - (update[1:] - update[0])

    result = root(error, np.zeros(len(counts) - 1), tol=1e-9)
    if np.max(np.abs(error(result.x))) > 1e-6:
        raise ValueError("MBAR bootstrap equations did not converge")
    return np.r_[0.0, result.x]


def sample_blocks(groups, size, block_frames, rng):
    # A block is drawn jointly across the windows of a given REMD run.
    selected = []
    for index in rng.integers(0, len(groups), size=len(groups)):
        data = groups[index]
        length = data.shape[1]
        starts = rng.integers(
            length - block_frames + 1,
            size=(length + block_frames - 1) // block_frames,
        )
        indices = (starts[:, None] + np.arange(block_frames)).ravel()[:length]
        selected.append(data[:, indices, :])
    return [
        np.concatenate([group[i] for group in selected]) for i in range(size)
    ]


def audit(directory, size, stride_ps, burn_ps):
    pair_counts = defaultdict(lambda: [0, 0])
    states = list(range(size))
    tails = {}
    visits = defaultdict(list)
    rounds = 0
    for line in (directory / "execution.log").read_text().splitlines():
        row = line.split()
        if row[0:1] == ["pair"]:
            _, step, left, right, score, draw, accepted = row
            left, right, accepted = int(left), int(right), int(accepted)
            if accepted != int(np.log(float(draw)) < float(score)):
                raise ValueError("exchange decision does not match its draw")
            if accepted:
                states[left], states[right] = states[right], states[left]
            if int(step) * stride_ps > burn_ps:
                pair_counts[f"{left}-{right}"][0] += 1
                pair_counts[f"{left}-{right}"][1] += accepted
        elif row[0:1] == ["label"]:
            _, step, walker, window, _ = row
            window, walker = int(window), int(walker)
            tails[window] = walker
            if len(tails) == size:
                if [tails[i] for i in range(size)] != states:
                    raise ValueError(
                        "walker mapping does not match accepted exchanges"
                    )
                if int(step) * stride_ps > burn_ps:
                    rounds += 1
                    for boundary in (0, size - 1):
                        history = visits[states[boundary]]
                        if not history or history[-1] != boundary:
                            history.append(boundary)
                tails.clear()
    if tails or not rounds:
        raise ValueError("incomplete exchange trace")
    return {
        "rounds_in_production": rounds,
        "pair_attempts_and_accepts": dict(pair_counts),
        "completed_round_trips": sum(
            max(0, (len(history) - 1) // 2) for history in visits.values()
        ),
    }


def ensemble_slope(low, high, first, second):
    energy = np.r_[low, high]
    centered = (energy - energy.mean()) / energy.std()
    labels = np.r_[np.zeros(len(low)), np.ones(len(high))]
    design = np.column_stack((np.ones(len(energy)), centered))
    solution = root(
        lambda params: design.T @ (expit(design @ params) - labels), [0, 0]
    )
    if not solution.success and np.max(np.abs(solution.fun)) > 1e-5:
        raise ValueError("neighbor energy distribution fit did not converge")
    return (solution.x[1] / energy.std()) / ((1 / first - 1 / second) / KB)


def distribution_distance(estimate, reference, estimate_weights=None):
    def histogram(frames, weights):
        return np.histogram2d(
            frames[:, 1],
            frames[:, 2],
            bins=30,
            range=[[-180, 180], [-180, 180]],
            weights=weights,
        )[0].ravel()

    return {
        "potential_wasserstein_kcal_mol": wasserstein_distance(
            estimate[:, 0],
            reference[:, 0],
            u_weights=estimate_weights,
        ),
        "ramachandran_js_distance": jensenshannon(
            histogram(estimate, estimate_weights),
            histogram(reference, None),
            base=2,
        ),
    }


def plots(output, rows):
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(11, 3.5 * len(rows)),
        layout="constrained",
        squeeze=False,
    )
    for row, (name, temperature, estimate, reference, estimate_weights) in zip(
        axes, rows
    ):
        edges = np.linspace(
            min(
                np.quantile(estimate[:, 0], 0.001),
                np.quantile(reference[:, 0], 0.001),
            ),
            max(
                np.quantile(estimate[:, 0], 0.999),
                np.quantile(reference[:, 0], 0.999),
            ),
            38,
        )
        row[0].hist(
            estimate[:, 0],
            bins=edges,
            weights=estimate_weights,
            density=True,
            histtype="step",
            label=name,
        )
        row[0].hist(
            reference[:, 0],
            bins=edges,
            density=True,
            histtype="step",
            label="Independent NVT",
        )
        row[0].set(
            xlabel="Potential energy (kcal/mol)",
            ylabel="Probability density",
            title=f"{temperature:g} K",
        )
        row[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
        row[0].legend(fontsize=8)
        for panel, title, frame, w in zip(
            row[1:],
            (name, "Independent NVT"),
            (estimate, reference),
            (estimate_weights, None),
        ):
            density, _, _ = np.histogram2d(
                frame[:, 1],
                frame[:, 2],
                bins=30,
                range=[[-180, 180], [-180, 180]],
                weights=w,
            )
            free_energy = np.full_like(density, np.nan, dtype=float)
            nonzero = density > 0
            free_energy[nonzero] = (
                -KB * temperature * np.log(density[nonzero] / density.max())
            )
            image = panel.imshow(
                free_energy.T,
                extent=(-180, 180, -180, 180),
                origin="lower",
                cmap="viridis_r",
                vmin=0,
                vmax=4,
            )
            panel.set(
                xlabel="φ (degrees)",
                ylabel="ψ (degrees)",
                title=title,
                xticks=[-180, -90, 0, 90, 180],
                yticks=[-180, -90, 0, 90, 180],
            )
        fig.colorbar(
            image, ax=row[1:], label="Free energy (kcal/mol)", shrink=0.75
        )
    fig.savefig(output / "distributions.png", dpi=180)
    plt.close(fig)


def run(args):
    remd = [load(Path(folder), args.burn_ps) for folder in args.remd]
    nvt = [load(Path(folder), args.burn_ps) for folder in args.nvt]
    if len(remd) < 2 or len(nvt) < 2:
        raise ValueError(
            "at least two independent REMD and NVT repetitions are required"
        )
    temperatures = np.asarray(remd[0][0]["temperatures"])
    target = nvt[0][0]["temperatures"][1]
    if (
        target in temperatures
        or not temperatures.min() < target < temperatures.max()
    ):
        raise ValueError(
            "need an unsampled target temperature inside the ladder"
        )
    seeds = [meta.get("seed") for meta, _ in remd + nvt]
    if any(value is not None for value in seeds) and len(set(seeds)) != len(
        seeds
    ):
        raise ValueError("repeated study seed")
    for meta, data in remd + nvt:
        if (
            meta["inputs"] != remd[0][0]["inputs"]
            or meta["engine_sha256"] != remd[0][0]["engine_sha256"]
        ):
            raise ValueError("mismatched system or engine")
        expected = (
            temperatures.tolist()
            if meta["mode"] == "remd"
            else [temperatures[0], target]
        )
        if (
            meta["temperatures"] != expected
            or data.shape[1] < 10
            or meta["dt_ps"] != remd[0][0]["dt_ps"]
            or meta["sample_steps"] != remd[0][0]["sample_steps"]
        ):
            raise ValueError("invalid temperature or sample count")
    groups = [data for _, data in remd]
    controls = [data for _, data in nvt]
    windows = [
        np.concatenate([group[i] for group in groups])
        for i in range(len(temperatures))
    ]
    reference = [
        np.concatenate([group[i] for group in controls]) for i in range(2)
    ]
    counts, frames, energies, u = mbar_input(windows, temperatures)
    mbar = MBAR(u, counts, solver_protocol="robust")
    predicted_weights = weights(mbar.f_k, u, counts, energies, target)
    direct = features(windows[0]).mean(axis=0)
    predicted = predicted_weights @ features(frames)
    reference_means = np.array(
        [features(group).mean(axis=0) for group in reference]
    )
    overlap = mbar.compute_overlap()["matrix"]
    ratios = [
        ensemble_slope(a[:, 0], b[:, 0], t0, t1)
        for a, b, t0, t1 in zip(
            windows, windows[1:], temperatures, temperatures[1:]
        )
    ]
    stride = remd[0][0]["sample_steps"] * remd[0][0]["dt_ps"]
    block = max(1, round(args.block_ps / stride))
    if any(data.shape[1] < 3 * block for data in groups + controls):
        raise ValueError("too few production frames for block bootstrap")
    rng = np.random.default_rng(20261116)
    errors = []
    slope_samples = []
    for _ in range(args.bootstrap):
        replica = sample_blocks(groups, len(temperatures), block, rng)
        control = sample_blocks(controls, 2, block, rng)
        n, x, e, scaled = mbar_input(replica, temperatures)
        w = weights(bootstrap_fit(scaled, n), scaled, n, e, target)
        slope_samples.append(
            [
                ensemble_slope(a[:, 0], b[:, 0], t0, t1)
                for a, b, t0, t1 in zip(
                    replica, replica[1:], temperatures, temperatures[1:]
                )
            ]
        )
        errors.append(
            [
                features(replica[0]).mean(axis=0)
                - features(control[0]).mean(axis=0),
                w @ features(x) - features(control[1]).mean(axis=0),
            ]
        )
    interval = np.quantile(errors, [0.025, 0.975], axis=0).transpose(1, 2, 0)
    labels = ("potential_kcal_mol", *REGIONS)

    def comparison(
        estimate, ref, ci, energy_sd, estimate_count, reference_count
    ):
        output = {}
        for label, obs, md, limits, margin in zip(
            labels,
            estimate,
            ref,
            ci,
            (0.2 * energy_sd, *([0.05] * len(REGIONS))),
        ):
            if label == labels[0]:
                supported = True
            else:
                # Require a minimum number of observations before accounting for time correlation.
                supported = (
                    min(obs, 1 - obs) * estimate_count >= 20
                    and min(md, 1 - md) * reference_count >= 20
                )
            output[label] = {
                "estimate": obs,
                "nvt": md,
                "difference": obs - md,
                "ci95_difference": limits.tolist(),
                "equivalence_margin": margin,
                "has_support": bool(supported),
                "within_margin": bool(
                    supported and limits[0] >= -margin and limits[1] <= margin
                ),
            }
        return output

    def efficiency(runs, gpu_seconds):
        values = [features(data[0])[:, 1] for _, data in runs]
        if any(np.std(series) == 0 for series in values):
            return None
        inefficiency = [
            timeseries.statistical_inefficiency(series) for series in values
        ]
        effective = sum(
            len(sample) / g for sample, g in zip(values, inefficiency)
        )
        return {
            "effective_samples": effective,
            "effective_samples_per_gpu_hour": effective * 3600 / gpu_seconds,
            "inefficiency_by_repeat": inefficiency,
            "gpu_hours": gpu_seconds / 3600,
        }

    weighted_count = float(1 / np.sum(predicted_weights**2))
    result = {
        "temperatures": temperatures.tolist(),
        "target_temperature": target,
        "burn_ps": args.burn_ps,
        "block_ps": args.block_ps,
        "bootstrap_draws": args.bootstrap,
        "frames_remd_each_temperature": counts.tolist(),
        "frames_nvt": list(map(len, reference)),
        "mbar_overlap": overlap.tolist(),
        "neighbor_slope_over_theory": ratios,
        "neighbor_slope_ratio_ci95": np.quantile(
            slope_samples, [0.025, 0.975], axis=0
        ).T.tolist(),
        "target_weight_effective_count_before_time_correlation": weighted_count,
        "direct_vs_nvt": comparison(
            direct,
            reference_means[0],
            interval[0],
            reference[0][:, 0].std(),
            len(windows[0]),
            len(reference[0]),
        ),
        "direct_distribution_distances": distribution_distance(
            windows[0], reference[0]
        ),
        "mbar_distribution_distances": distribution_distance(
            frames, reference[1], predicted_weights
        ),
        "mbar_vs_nvt": comparison(
            predicted,
            reference_means[1],
            interval[1],
            reference[1][:, 0].std(),
            weighted_count,
            len(reference[1]),
        ),
        "remd_300K_per_repeat": [
            dict(zip(labels, features(group[0]).mean(axis=0)))
            for group in groups
        ],
        "remd_300K_halves": [
            [
                dict(zip(labels, features(frame).mean(axis=0)))
                for frame in np.array_split(group[0], 2)
            ]
            for group in groups
        ],
        "nvt_per_repeat": [
            [
                dict(zip(labels, features(window).mean(axis=0)))
                for window in group
            ]
            for group in controls
        ],
        "exchange_audit": [
            audit(Path(folder), len(temperatures), meta["dt_ps"], args.burn_ps)
            for folder, (meta, _) in zip(args.remd, remd)
        ],
    }
    if all("wall_seconds_by_run" in meta for meta, _ in remd + nvt):
        result["alpha_R_300K_efficiency"] = {
            "remd": efficiency(
                remd,
                len(temperatures)
                * sum(meta["wall_seconds_by_run"][0] for meta, _ in remd),
            ),
            "nvt": efficiency(
                nvt, sum(meta["wall_seconds_by_run"][0] for meta, _ in nvt)
            ),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    plots(
        args.output,
        [
            ("Direct T-REMD", temperatures[0], windows[0], reference[0], None),
            (
                "MBAR interpolation",
                target,
                frames,
                reference[1],
                predicted_weights,
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remd", nargs="+", required=True)
    parser.add_argument("--nvt", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--burn-ps", type=float, default=300)
    parser.add_argument("--block-ps", type=float, default=100)
    parser.add_argument("--bootstrap", type=int, default=250)
    run(parser.parse_args())
