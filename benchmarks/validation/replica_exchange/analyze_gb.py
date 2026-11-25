"""Compare GB1 fixed-temperature sampling, replica exchange and held-out MBAR."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymbar import MBAR, timeseries

from analyze_alanine import (
    KB,
    REGIONS,
    audit,
    bootstrap_fit,
    distribution_distance,
    features,
    load,
    mbar_input,
    plots,
    sample_blocks,
    weights,
)


def windows(groups):
    return [
        np.concatenate([group[i] for group in groups])
        for i in range(len(groups[0]))
    ]


def resample(groups, block, rng, coupled):
    size = len(groups[0])
    if coupled:
        return sample_blocks(groups, size, block, rng)
    return [
        sample_blocks([group[i : i + 1] for group in groups], 1, block, rng)[0]
        for i in range(size)
    ]


def fit(samples, temperatures, targets):
    n, frames, e, u = mbar_input(samples, temperatures)
    f = bootstrap_fit(u, n)
    return frames, [weights(f, u, n, e, t) for t in targets]


def core_visits(series, dwell_frames):
    # Ignore other and count an entry only after a contiguous core visit.
    visits = []
    run = 0
    previous = -1
    other = REGIONS.index("other")
    for label in np.argmax(series, axis=1):
        run = run + 1 if label == previous else 1
        previous = label
        if (
            label != other
            and run == dwell_frames
            and (not visits or visits[-1] != label)
        ):
            visits.append(label)
    return visits


def walker_frames(path, frames, metadata, burn_ps):
    size, samples, _ = frames.shape
    steps = np.arange(1, 1 + metadata["steps"] // metadata["sample_steps"])
    steps *= metadata["sample_steps"]
    steps = steps[steps * metadata["dt_ps"] > burn_ps]
    if len(steps) != samples:
        raise ValueError(
            f"cannot align walker trace with saved frames in {path}"
        )
    assignment = np.arange(size)
    assignments = np.empty((samples, size), dtype=int)
    frame_index = 0
    with (path / "execution.log").open() as stream:
        for line in stream:
            if not line.startswith("pair "):
                continue
            _, step, left, right, _, _, accepted = line.split()
            step, left, right = int(step), int(left), int(right)
            # Same-step saved coordinates precede the exchange. The energy
            # audit confirms that rerun frame s gives the pre-exchange state.
            end = np.searchsorted(steps, step, side="right")
            assignments[frame_index:end] = assignment
            frame_index = end
            if int(accepted):
                assignment[left], assignment[right] = (
                    assignment[right],
                    assignment[left],
                )
    assignments[frame_index:] = assignment
    walkers = np.empty_like(frames)
    walkers[assignments, np.arange(samples)[:, None]] = frames.transpose(
        1, 0, 2
    )
    return walkers


def diagnostics(series, dwell_frames):
    visits = core_visits(series, dwell_frames)
    transitions = np.zeros((len(REGIONS), len(REGIONS)), dtype=int)
    if len(visits) > 1:
        np.add.at(transitions, (visits[:-1], visits[1:]), 1)
    round_trips = {}
    for i, left in enumerate(REGIONS[:-1]):
        for j, right in enumerate(REGIONS[i + 1 : -1], start=i + 1):
            alternating = []
            for label in visits:
                if label in (i, j) and (
                    not alternating or alternating[-1] != label
                ):
                    alternating.append(label)
            round_trips[f"{left}/{right}"] = max(0, (len(alternating) - 1) // 2)

    def effective(observable):
        if np.std(observable) == 0:
            return None
        return len(observable) / timeseries.statistical_inefficiency(observable)

    half = len(series) // 2
    return dict(
        populations=series.mean(axis=0).tolist(),
        populations_by_half=[
            series[:half].mean(axis=0).tolist(),
            series[half:].mean(axis=0).tolist(),
        ],
        core_transitions=transitions.tolist(),
        core_round_trips=round_trips,
        core_visits_by_region=np.bincount(
            np.asarray(visits, dtype=int), minlength=len(REGIONS)
        ).tolist(),
        effective_frames=[
            effective(column)
            if i in visits or i == REGIONS.index("other")
            else None
            for i, column in enumerate(series.T)
        ],
    )


def compare(estimate, ref, limits, reference_samples, effective_count):
    result = {}
    for i, key in enumerate(("potential_kcal_mol", *REGIONS)):
        margin = 0.2 * np.std(reference_samples[:, 0]) if i == 0 else 0.05
        lo, hi = limits[:, i]
        enough = (
            i == 0
            or min(
                min(estimate[i], 1 - estimate[i]) * effective_count,
                min(ref[i], 1 - ref[i]) * len(reference_samples),
            )
            >= 20
        )
        result[key] = dict(
            estimate=estimate[i],
            reference=ref[i],
            difference=estimate[i] - ref[i],
            ci95=[lo, hi],
            margin=margin,
            support=bool(enough),
            equivalent=bool(enough and lo >= -margin and hi <= margin),
        )
    return result


def plot_populations(output, result):
    tests = list(result["direct"].items())
    titles = [f"T-REMD vs NVT · {float(t):g} K" for t, _ in tests]
    comparisons = [values for _, values in tests]
    for arm, targets in result["held_out"].items():
        for t in result["held_out_K"]:
            titles.append(f"{arm.upper()} MBAR · {t:g} K")
            comparisons.append(targets[str(t)]["comparison"])
    fig, panels = plt.subplots(
        2, 4, figsize=(13, 6), sharex=True, sharey=True, layout="constrained"
    )
    positions = np.arange(len(REGIONS) - 1)
    for panel, title, comparison in zip(panels.flat, titles, comparisons):
        panel.axvspan(-5, 5, color="#e7f2ec")
        panel.axvline(0, color="#64748b", linewidth=0.8)
        for i, region in enumerate(REGIONS[:-1]):
            data = comparison[region]
            lo, hi = np.array(data["ci95"]) * 100
            color = "#167557" if data["equivalent"] else "#aa482f"
            panel.hlines(i, lo, hi, color=color, linewidth=2)
            panel.plot(
                data["difference"] * 100, i, "o", color=color, markersize=4
            )
        panel.set(
            title=title,
            yticks=positions,
            yticklabels=("αR", "PPII", "β", "Positive φ"),
        )
        panel.grid(axis="x", alpha=0.2)
    panels[0, 0].invert_yaxis()
    fig.supxlabel("Difference from independent NVT (percentage points)")
    fig.savefig(output / "population-differences.png", dpi=180)
    plt.close(fig)


def plot_convergence(output, records, stride_ps):
    colors = ("#117583", "#bf5a25", "#6d54a3")
    for arm, runs in records.items():
        temperatures = runs[0][0]["temperatures"]
        fig, panels = plt.subplots(
            len(temperatures),
            len(REGIONS) - 1,
            figsize=(12, 2.1 * len(temperatures)),
            sharex=True,
            sharey="col",
            layout="constrained",
        )
        initial_states = {}
        for metadata, samples in runs:
            initial = metadata["inputs"]["amber_rst7"]
            color = colors[
                initial_states.setdefault(initial, len(initial_states))
                % len(colors)
            ]
            for row, window in zip(panels, samples):
                indicators = features(window)[:, 1:-1]
                count = np.arange(1, len(window) + 1)
                average = 100 * np.cumsum(indicators, axis=0) / count[:, None]
                for column, y in zip(row, average.T):
                    column.plot(
                        count * stride_ps / 1000,
                        y,
                        color=color,
                        alpha=0.75,
                        linewidth=0.9,
                    )
        for column, title in zip(panels[0], ("αR", "PPII", "β", "Positive φ")):
            column.set(title=title)
        for row, temperature in zip(panels, temperatures):
            row[0].set_ylabel(f"{temperature:g} K (%)")
        fig.supxlabel("Time after discarded segment (ns)")
        fig.savefig(output / f"convergence-{arm}.png", dpi=160)
        plt.close(fig)


def run(args):
    folders = {
        arm: sorted(
            path for path in args.data.glob(f"{arm}-*") if path.is_dir()
        )
        for arm in ("remd", "nvt", "control")
    }
    if any(len(paths) < args.repeats for paths in folders.values()):
        raise ValueError("missing independent repeats")
    records = {
        arm: [load(path, args.burn_ps) for path in paths]
        for arm, paths in folders.items()
    }
    ladder = np.array(records["remd"][0][0]["temperatures"])
    targets = records["control"][0][0]["temperatures"]
    if len(targets) != 2 or any(
        t in ladder or not ladder[0] < t < ladder[-1] for t in targets
    ):
        raise ValueError(
            "two held-out temperatures inside the exchange ladder are required"
        )
    reference = records["remd"][0][0]
    seeds = set()
    for arm, recs in records.items():
        for meta, data in recs:
            if (
                meta["seed"] in seeds
                or meta["temperatures"]
                != (targets if arm == "control" else ladder.tolist())
                or any(
                    meta[key] != reference[key]
                    for key in (
                        "engine_sha256",
                        "dt_ps",
                        "sample_steps",
                        "steps",
                    )
                )
                or any(
                    meta["inputs"][key] != reference["inputs"][key]
                    for key in ("amber_parm7", "gb_in_file")
                )
            ):
                raise ValueError(f"incompatible metadata in {arm}: {meta}")
            seeds.add(meta["seed"])
    raw = {arm: [data for _, data in recs] for arm, recs in records.items()}
    result = {
        "ladder_K": ladder.tolist(),
        "held_out_K": targets,
        "burn_ps": args.burn_ps,
        "block_ps": args.block_ps,
        "bootstrap_draws": args.bootstrap,
        "regions": REGIONS,
    }
    result["exchange"] = [
        audit(path, len(ladder), reference["dt_ps"], args.burn_ps)
        for path in folders["remd"]
    ]
    stride_ps = reference["dt_ps"] * reference["sample_steps"]
    dwell_frames = max(1, round(args.core_dwell_ps / stride_ps))
    result["core_dwell_ps"] = dwell_frames * stride_ps
    result["sampling"] = {
        arm: [
            [
                diagnostics(features(window)[:, 1:], dwell_frames)
                for window in repeat
            ]
            for repeat in groups
        ]
        for arm, groups in raw.items()
    }
    result["remd_walker_sampling"] = [
        [
            diagnostics(features(walker)[:, 1:], dwell_frames)
            for walker in walker_frames(path, frames, meta, args.burn_ps)
        ]
        for path, (meta, frames) in zip(folders["remd"], records["remd"])
    ]
    # Keep all frames for estimates and thin evenly for the block bootstrap.
    raw_block = max(1, round(args.block_ps / stride_ps))
    thin = max(1, round(args.bootstrap_stride_ps / stride_ps))
    block = max(1, round(raw_block / thin))
    if any(
        group.shape[1] // thin < 3 * block
        for data in raw.values()
        for group in data
    ):
        raise ValueError("less than three bootstrap blocks per trajectory")
    sampled = {
        arm: [group[:, ::thin, :] for group in groups]
        for arm, groups in raw.items()
    }
    pooled = {arm: windows(groups) for arm, groups in raw.items()}
    labels = [str(t) for t in ladder]
    bootstrap = {"direct": [], "remd": [], "nvt": []}
    rng = np.random.default_rng(162829)
    for _ in range(args.bootstrap):
        batches = {
            arm: resample(groups, block, rng, coupled=(arm == "remd"))
            for arm, groups in sampled.items()
        }
        means = {
            arm: [features(x).mean(axis=0) for x in group]
            for arm, group in batches.items()
        }
        bootstrap["direct"].append(np.array(means["remd"]) - means["nvt"])
        for arm in ("remd", "nvt"):
            frames, w = fit(batches[arm], ladder, targets)
            estimates = np.asarray(w) @ features(frames)
            bootstrap[arm].append(estimates - means["control"])
    intervals = {
        arm: np.quantile(values, [0.025, 0.975], axis=0)
        for arm, values in bootstrap.items()
    }
    result["direct"] = {
        label: compare(
            features(a).mean(axis=0),
            features(b).mean(axis=0),
            intervals["direct"][:, i],
            b,
            len(a),
        )
        for i, (label, a, b) in enumerate(
            zip(labels, pooled["remd"], pooled["nvt"])
        )
    }
    result["direct_distances"] = {
        key: distribution_distance(a, b)
        for key, a, b in zip(labels, pooled["remd"], pooled["nvt"])
    }
    result["bootstrap_stride_ps"] = stride_ps * thin
    result["held_out"] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "direct").mkdir(exist_ok=True)
    plots(
        args.output / "direct",
        [
            ("Direct T-REMD", t, a, b, None)
            for t, a, b in zip(ladder, pooled["remd"], pooled["nvt"])
        ],
    )
    for arm in ("remd", "nvt"):
        frames, w = fit(pooled[arm], ladder, targets)
        features_of_frames = features(frames)
        result["held_out"][arm] = {}
        for i, (t, b, wt) in enumerate(zip(targets, pooled["control"], w)):
            ess = float(1 / np.sum(wt**2))
            result["held_out"][arm][str(t)] = dict(
                comparison=compare(
                    wt @ features_of_frames,
                    features(b).mean(axis=0),
                    intervals[arm][:, i],
                    b,
                    ess,
                ),
                distances=distribution_distance(frames, b, wt),
                weight_ess_before_time_correlation=ess,
            )
        n, _, energies, u = mbar_input(pooled[arm], ladder)
        u_targets = energies[None, :] / (KB * np.asarray(targets)[:, None])
        independent = MBAR(
            np.vstack([u, u_targets]), np.r_[n, 0, 0], solver_protocol="robust"
        )
        weight_error = np.max(
            np.abs(independent.weights()[:, -2:] - np.asarray(w).T)
        )
        if weight_error > 1e-9:
            raise ValueError(
                f"MBAR weight cross-check failed for {arm}: {weight_error:g}"
            )
        result["held_out"][arm]["max_weight_difference_vs_pymbar"] = float(
            weight_error
        )
        result["held_out"][arm]["overlap"] = (
            MBAR(u, n, solver_protocol="robust")
            .compute_overlap()["matrix"]
            .tolist()
        )
        (args.output / arm).mkdir(exist_ok=True)
        plots(
            args.output / arm,
            [
                (f"{arm.upper()} MBAR", t, frames, b, wt)
                for t, b, wt in zip(targets, pooled["control"], w)
            ],
        )
    plot_populations(args.output, result)
    plot_convergence(args.output, records, stride_ps)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--burn-ps", type=float, default=2000)
    parser.add_argument("--core-dwell-ps", type=float, default=3)
    parser.add_argument("--block-ps", type=float, default=500)
    parser.add_argument("--bootstrap-stride-ps", type=float, default=5)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=6)
    run(parser.parse_args())
