"""Observable-specific correlation, sampling efficiency and distribution checks."""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.performance.amber.tests.utils import (
    read_box_trajectory,
    read_coordinate_trajectory,
)
from benchmarks.utils import Extractor

LABELS = {
    "normal_2fs": "Normal 2 fs",
    "hmr_2fs": "HMR 2 fs",
    "hmr_4fs": "HMR 4 fs",
}
COLORS = {"normal_2fs": "black", "hmr_2fs": "C0", "hmr_4fs": "C1"}
OBSERVABLES = [
    "psi_A",
    "cos_psi",
    "sin_psi",
    "cos_phi",
    "sin_phi",
    "phi_positive",
]


def correlation(x, sample_ps):
    """FFT covariance + Geyer initial-positive/monotone paired sequence.

    g=-1+2*sum(rho[2k]+rho[2k+1]); ESS=N/g. We conservatively cap ESS at N.
    Constant observables have undefined ESS, never N independent samples.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or not np.isfinite(x).all():
        raise ValueError("Need at least 20 finite values")
    y = x - x.mean()
    if np.mean(y * y) < 1e-14:
        return dict(g=None, ess=None, tau_int_ps=None, cutoff_ps=None), None
    n = len(x)
    transform = np.fft.rfft(y, n=1 << (2 * n - 1).bit_length())
    covariance = np.fft.irfft(transform * transform.conj())[:n] / n
    rho = covariance / covariance[0]
    pairs = rho[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    cutoff = int(stop[0]) if len(stop) else len(pairs)
    monotone = np.minimum.accumulate(pairs[:cutoff])
    g = max(1.0, float(-1 + 2 * monotone.sum()))
    return dict(
        g=g,
        ess=n / g,
        tau_int_ps=0.5 * g * sample_ps,
        cutoff_ps=2 * cutoff * sample_ps,
    ), rho


def observables(phi, psi):
    return dict(
        psi_A=(np.cos(psi) > 0).astype(float),
        cos_psi=np.cos(psi),
        sin_psi=np.sin(psi),
        cos_phi=np.cos(phi),
        sin_phi=np.sin(phi),
        phi_positive=(phi > 0).astype(float),
    )


def batch_ess(x, sample_ps, block_ps):
    """Nonoverlapping batch-means cross-check, requiring at least eight batches."""
    x = np.asarray(x, float)
    width = round(block_ps / sample_ps)
    count = len(x) // width
    variance = np.var(x, ddof=1)
    if count < 8 or variance < 1e-14:
        return None
    means = x[: count * width].reshape(count, width).mean(axis=1)
    g = max(1.0, float(width * np.var(means, ddof=1) / variance))
    return len(x) / g


def core_transitions(psi):
    degrees = np.degrees(psi)
    states = np.full(len(psi), -1)
    states[(degrees >= -90) & (degrees <= 45)] = 0
    states[(degrees >= 90) | (degrees <= -150)] = 1
    visited = states[states >= 0]
    return int(np.count_nonzero(np.diff(visited)))


def positive_phi_core_visits(phi):
    degrees = np.degrees(phi)
    states = np.full(len(phi), -1)
    states[(degrees >= -150) & (degrees <= -30)] = 0
    states[(degrees >= 30) & (degrees <= 150)] = 1
    visited = states[states >= 0]
    if not len(visited):
        return 0
    return int(visited[0] == 1) + int(np.count_nonzero(np.diff(visited) == 1))


def mean_interval(values):
    a = np.asarray(values, float)
    return dict(
        mean=float(a.mean()),
        half_width=float(4.3026527299 * a.std(ddof=1) / np.sqrt(3))
        if len(a) == 3
        else None,
        values=a.tolist(),
    )


def split_rhat(series):
    n = min(len(x) // 2 for x in series)
    chains = np.array(
        [
            part[:n]
            for x in series
            for part in (x[: len(x) // 2], x[len(x) // 2 :])
        ]
    )
    within = np.var(chains, axis=1, ddof=1).mean()
    if within <= 1e-14:
        return None
    between = n * np.var(chains.mean(axis=1), ddof=1)
    return float(np.sqrt(((n - 1) * within / n + between / n) / within))


def trajectory_dihedral(coordinates, boxes, atoms):
    """Independent atan2/projection reference, using minimum-image bonds."""
    a, b, c, d = atoms
    b0 = coordinates[:, a] - coordinates[:, b]
    b1 = coordinates[:, c] - coordinates[:, b]
    b2 = coordinates[:, d] - coordinates[:, c]
    for vector in (b0, b1, b2):
        vector -= np.rint(vector / boxes) * boxes
    b1 /= np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.sum(b0 * b1, axis=1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=1)[:, None] * b1
    return np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1)
    )


def analyze_case(root, case):
    status = json.loads((case / "status.json").read_text())
    if not status["completed"]:
        raise ValueError(f"Failed simulation: {case}")
    rows = Extractor.parse_mdout_rows(
        case / "mdout.txt", ["step", "temperature", "potential", "phi", "psi"]
    )
    arrays = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    options = status["specification"]["options"]
    if not np.array_equal(
        arrays["step"],
        np.arange(
            0, options["step_limit"] + 1, options["write_mdout_interval"]
        ),
    ):
        raise ValueError("Incomplete mdout")
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Nonfinite output")
    log = (case / "run.log").read_text()
    if re.search(r"Fatal SPONGE|(?:cuda|hip)Error[A-Z]\w*|SITS mode =", log):
        raise ValueError("Runtime error or unintended bias")
    for required in (
        "Mode set to NVT",
        "START INITIALIZING MIDDLE LANGEVIN DYNAMICS:",
        "target temperature is 300.00 K",
        "friction coefficient is 1.00 ps^-1",
        "constrain pair number is 1830",
        "rigid triangle numbers is 606",
        "rigid pair numbers is 3",
        "remaining simple constrain pair numbers is 9",
        "final freedom 3690",
    ):
        if required not in log:
            raise ValueError(
                f"Missing intended runtime configuration: {required}"
            )
    if status["hmr"]:
        if "12 hydrogens repartitioned, 606 waters excluded" not in log:
            raise ValueError("Incorrect HMR selection")
    elif "hydrogens repartitioned" in log:
        raise ValueError("HMR enabled in ordinary-mass control")
    keep = (arrays["step"] > 0) & (
        arrays["step"] >= 0.2 * options["step_limit"]
    )
    phi, psi = arrays["phi"][keep], arrays["psi"][keep]
    if max(np.max(np.abs(phi)), np.max(np.abs(psi))) > np.pi + 0.001:
        raise ValueError("CV dihedral unit is not radians")
    sample_ps = status["dt_fs"] / 1000 * options["write_mdout_interval"]
    result = dict(
        status,
        name=case.name,
        sample_ps=sample_ps,
        analyzed_ps=float(
            (arrays["step"][keep][-1] - arrays["step"][keep][0])
            * status["dt_fs"]
            / 1000
        ),
        temperature=float(arrays["temperature"][keep].mean()),
        potential=float(arrays["potential"][keep].mean()),
        psi_A_population=float((np.cos(psi) > 0).mean()),
        phi_positive_population=float((phi > 0).mean()),
        core_transitions=core_transitions(psi),
        positive_phi_samples=int(np.count_nonzero(phi > 0)),
        positive_phi_core_visits=positive_phi_core_visits(phi),
        retained_samples=len(phi),
        metrics={},
        psi_A_batch_ess={
            str(block): batch_ess(
                (np.cos(psi) > 0).astype(float), sample_ps, block
            )
            for block in (50, 100, 200)
        },
    )
    saved = dict(
        phi=phi,
        psi=psi,
        elapsed_ps=arrays["step"][keep] * status["dt_fs"] / 1000,
    )
    for key, values in observables(phi, psi).items():
        info, rho = correlation(values, sample_ps)
        info.update(
            mean=float(values.mean()),
            ess_per_ns=info["ess"] / result["analyzed_ps"] * 1000
            if info["ess"] is not None
            else None,
            ess_per_hour=info["ess"] / status["wall_seconds"] * 3600
            if info["ess"] is not None
            else None,
        )
        result["metrics"][key] = info
        if rho is not None:
            saved["acf_" + key] = rho
    # Same original-mass X-H selection, including rigid-water H-H edges.
    fixture = root / "statics/ala2_sits"
    mass = np.loadtxt(fixture / "ALA_mass.txt", skiprows=1)
    raw_bonds = np.loadtxt(fixture / "ALA_bond.txt", skiprows=1)
    bonds = np.array(
        list(
            {
                tuple(sorted(row[:2].astype(int))): row for row in raw_bonds
            }.values()
        )
    )
    a, b = bonds[:, :2].astype(int).T
    mask = (mass[a] < 2) | (mass[b] < 2)
    a, b, target = a[mask], b[mask], bonds[mask, 3]
    trajectory = read_coordinate_trajectory(
        case / "mdcrd.dat", len(mass)
    ).astype(float)
    boxes = read_box_trajectory(case / "mdbox.txt")
    if len(trajectory) != options["step_limit"] // options[
        "write_trajectory_interval"
    ] or len(boxes) != len(trajectory):
        raise ValueError("Incomplete trajectory")
    if (
        not np.isfinite(trajectory).all()
        or not np.isfinite(boxes).all()
        or not (boxes[:, :3] > 0).all()
    ):
        raise ValueError("Invalid trajectory")
    if not np.allclose(boxes[:, 3:], 90):
        raise ValueError("Requires orthorhombic box")
    prepared_box = np.loadtxt(
        root / "runs/equilibrated/restart_coordinate.txt",
        skiprows=len(mass) + 1,
    )
    if not np.allclose(boxes, prepared_box, rtol=0, atol=1e-5):
        raise ValueError("NVT volume changed from common prepared state")
    steps = (
        np.arange(1, len(trajectory) + 1) * options["write_trajectory_interval"]
    )
    indices = np.searchsorted(arrays["step"], steps)
    if not np.array_equal(arrays["step"][indices], steps):
        raise ValueError("Stored coordinates are not aligned with CV records")
    result["cv_reference_errors_rad"] = {}
    for key, atoms in (("phi", [4, 6, 8, 14]), ("psi", [6, 8, 14, 16])):
        reference = trajectory_dihedral(trajectory, boxes[:, :3], atoms)
        error = float(
            np.max(
                np.abs(
                    np.angle(np.exp(1j * (reference - arrays[key][indices])))
                )
            )
        )
        result["cv_reference_errors_rad"][key] = error
        # mdout angles have four decimal places; engine clamps acos near 0/pi.
        if error > 0.002:
            raise ValueError(
                f"CV disagrees with independent coordinate reference: {key}"
            )
    delta = trajectory[:, a] - trajectory[:, b]
    delta -= np.rint(delta / boxes[:, None, :3]) * boxes[:, None, :3]
    result["sampled_constraint_error"] = float(
        np.max(np.abs(np.linalg.norm(delta, axis=-1) / target - 1))
    )
    result["ns_per_day"] = (
        status["duration_ps"] / 1000 / status["wall_seconds"] * 86400
    )
    result["finite_constraint_screen_pass"] = (
        result["sampled_constraint_error"] < 0.001
        and 290 < result["temperature"] < 310
    )
    if not result["finite_constraint_screen_pass"]:
        raise ValueError(f"Temperature/constraint screen failed: {case}")
    result["diagnostic_warnings"] = []
    if result["core_transitions"] < 20:
        result["diagnostic_warnings"].append(
            "fewer than 20 psi core-state alternations"
        )
    if result["positive_phi_samples"] < 20:
        result["diagnostic_warnings"].append(
            "fewer than 20 positive-phi observations"
        )
    if result["positive_phi_core_visits"] < 10:
        result["diagnostic_warnings"].append(
            "fewer than 10 positive-phi core visits"
        )
    for key, info in result["metrics"].items():
        if info["ess"] is None or info["ess"] < 50:
            result["diagnostic_warnings"].append(
                f"{key}: undefined ESS or ESS below 50"
            )
    np.savez_compressed(case / "sampling_series.npz", **saved)
    (case / "sampling_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    return result, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = json.loads((root / "provenance.json").read_text())
    if (
        hashlib.sha256((root / "SPONGE").read_bytes()).hexdigest()
        != provenance["binary_sha256"]
    ):
        raise ValueError("Frozen binary changed")
    for name, digest in provenance["fixture_sha256"].items():
        if (
            hashlib.sha256(
                (root / "statics/ala2_sits" / name).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise ValueError(f"Frozen fixture changed: {name}")
    plan = json.loads((root / "production_plan.json").read_text())
    expected = {
        f"production_{key}_{seed}" for key in LABELS for seed in plan["seeds"]
    }
    found = {p.name for p in (root / "runs").glob("production_*")}
    if found != expected or not all(
        (root / "runs" / name / "status.json").exists() for name in expected
    ):
        raise ValueError(
            "Require all nine planned production runs before final analysis"
        )
    groups = defaultdict(list)
    for case in sorted((root / "runs").glob("production_*")):
        if (case / "status.json").exists():
            for name, digest in provenance["fixture_sha256"].items():
                if (
                    hashlib.sha256((case / name).read_bytes()).hexdigest()
                    != digest
                ):
                    raise ValueError(
                        f"Production fixture changed: {case.name}/{name}"
                    )
            result, data = analyze_case(root, case)
            if result["duration_ps"] != plan["duration_ps"]:
                raise ValueError(
                    "Production duration does not match frozen plan"
                )
            if (
                result["specification"]["binary_sha256"]
                != provenance["binary_sha256"]
            ):
                raise ValueError(
                    "Production binary does not match frozen provenance"
                )
            key = f"{'hmr' if result['hmr'] else 'normal'}_{result['dt_fs']}fs"
            groups[key].append((result, data))
    if not groups:
        raise ValueError("No completed production cases")
    lines = [
        "# HMR timestep and sampling efficiency: alanine dipeptide",
        "",
        "Public repository ACE–ALA–NME + 606 waters, 1840 atoms. Conventional NVT without enhanced-sampling bias potentials, 300 K, Middle Langevin tau 1 ps, identical original constraint selection, default double SETTLE + remaining SHAKE. HMR factors 2.0/2.5, water unchanged. No SITS or other enhanced sampling. This describes the simulation setup, not a claim of absence of finite-timestep bias.",
        "",
        "Normal 2 fs versus HMR 2 fs isolates mass changes; HMR 2 versus 4 fs isolates timestep changes. Three velocity seeds start at the same prepared coordinates. Runs have equal physical durations and 0.1 ps CV output; first 20% excluded. Measured full production wall time includes initialization and discarded burn-in. ESS/ns uses retained physical time. This estimates ESS/wall rates; these were not fixed-wall-duration runs.",
        "",
        "Primary observable: psi_A = I[cos(psi)>0], a periodic psi basin indicator. Secondary observables are sin/cos(phi/psi) and I[phi>0]. ESS is observable-specific, not an independent count of complete molecular configurations. Constant/unvisited indicators have undefined ESS. FFT covariance uses N normalization; Geyer initial positive/monotone paired sums, g>=1, ESS=N/g, tau_int=g*sample_interval/2. This is a per-trajectory estimator, not Stan's multi-chain ESS. Split Rhat is an ordinary, non-rank-normalized diagnostic on six half-chains. Core-state alternations use narrower cores A=[-90,45] and B=[90,180] union [-180,-150] degrees with gap observations removed; these are diagnostic counts, not kinetic rates.",
        "",
        "[ESS method](https://mc-stan.org/docs/reference-manual/analysis.html); [molecular-simulation uncertainty and block-averaging guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC6286151/).",
        "",
        "## Per-run diagnostics",
        "",
        "| condition | seed | duration ns | wall s | ns/day | T K | U kcal/mol | p(psi_A) | core alternations | psi_A tau_int ps | psi_A ESS | ESS/hour | constraint error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    aggregates = {}
    for key in LABELS:
        if key not in groups:
            continue
        group = groups[key]
        aggregates[key] = dict(
            replicas=len(group),
            ns_per_day=mean_interval([r["ns_per_day"] for r, _ in group]),
            psi_A_population=mean_interval(
                [r["psi_A_population"] for r, _ in group]
            ),
            phi_positive_population=mean_interval(
                [r["phi_positive_population"] for r, _ in group]
            ),
            temperature=mean_interval([r["temperature"] for r, _ in group]),
            potential=mean_interval([r["potential"] for r, _ in group]),
            metrics={},
        )
        for observable in OBSERVABLES:
            values = [r["metrics"][observable] for r, _ in group]
            rhat = (
                split_rhat(
                    [
                        observables(d["phi"], d["psi"])[observable]
                        for _, d in group
                    ]
                )
                if len(group) == 3
                else None
            )
            aggregates[key]["metrics"][observable] = dict(split_rhat=rhat)
            for field in ("ess_per_ns", "ess_per_hour", "tau_int_ps"):
                valid = [v[field] for v in values if v[field] is not None]
                aggregates[key]["metrics"][observable][field] = (
                    mean_interval(valid) if valid else None
                )
        for r, _ in group:
            m = r["metrics"]["psi_A"]
            values = [
                LABELS[key],
                r["seed"],
                r["duration_ps"] / 1000,
                r["wall_seconds"],
                r["ns_per_day"],
                r["temperature"],
                r["potential"],
                r["psi_A_population"],
                r["core_transitions"],
                m["tau_int_ps"],
                m["ess"],
                m["ess_per_hour"],
                r["sampled_constraint_error"],
            ]
            lines.append(
                "| "
                + " | ".join(
                    f"{v:.5g}" if isinstance(v, float) else str(v)
                    for v in values
                )
                + " |"
            )
    differences = []
    lines += [
        "",
        "## Matched-seed speed ratios",
        "",
        "Target/reference ratios use geometric means of the three seed ratios; intervals are nominal 95% Student-t repeatability intervals on log ratios (df=2), unadjusted for multiple comparisons. They describe between-seed variation, not a formal confidence interval for the ESS estimator. A wide interval is inconclusive, not evidence of equal efficiency.",
        "",
        "| target / reference | observable | throughput ratio | ESS/ns ratio | ESS/hour ratio [95% interval] |",
        "|---|---|---:|---:|---:|",
    ]
    for target, reference in [
        ("hmr_2fs", "normal_2fs"),
        ("hmr_4fs", "hmr_2fs"),
        ("hmr_4fs", "normal_2fs"),
    ]:
        if target not in groups or reference not in groups:
            continue
        refs = {r["seed"]: r for r, _ in groups[reference]}
        pairs = [
            (r, refs[r["seed"]]) for r, _ in groups[target] if r["seed"] in refs
        ]
        if len(pairs) != 3:
            continue
        for observable in OBSERVABLES:
            item = dict(
                target=target, reference=reference, observable=observable
            )
            for field in ("ns_per_day", "ess_per_ns", "ess_per_hour"):
                ratios = []
                for a, b in pairs:
                    va = (
                        a[field]
                        if field == "ns_per_day"
                        else a["metrics"][observable][field]
                    )
                    vb = (
                        b[field]
                        if field == "ns_per_day"
                        else b["metrics"][observable][field]
                    )
                    if va is not None and vb is not None:
                        ratios.append(va / vb)
                if len(ratios) == 3:
                    stats = mean_interval(np.log(ratios))
                    center = stats["mean"]
                    half = stats["half_width"]
                    item[field] = dict(
                        ratio=float(np.exp(center)),
                        low=float(np.exp(center - half)),
                        high=float(np.exp(center + half)),
                        seed_ratios=ratios,
                    )
                else:
                    item[field] = None
            differences.append(item)
            if all(
                item[k] is not None
                for k in ("ns_per_day", "ess_per_ns", "ess_per_hour")
            ):
                t, p, e = (
                    item[k]
                    for k in ("ns_per_day", "ess_per_ns", "ess_per_hour")
                )
                lines.append(
                    f"| {target}/{reference} | {observable} | {t['ratio']:.3f} | {p['ratio']:.3f} | {e['ratio']:.3f} [{e['low']:.3f}, {e['high']:.3f}] |"
                )
    (root / "sampling_summary.json").write_text(
        json.dumps(aggregates, indent=2, allow_nan=False)
    )
    (root / "sampling_ratios.json").write_text(
        json.dumps(differences, indent=2, allow_nan=False)
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    keys = [k for k in LABELS if k in groups]
    for ax, field, label in [
        (axes[0, 0], "ns_per_day", "Throughput (ns/day)"),
        (axes[0, 1], "ess_per_ns", "psi basin ESS / ns"),
        (axes[1, 0], "ess_per_hour", "psi basin ESS / wall hour"),
    ]:
        for i, key in enumerate(keys):
            values = [
                r[field]
                if field == "ns_per_day"
                else r["metrics"]["psi_A"][field]
                for r, _ in groups[key]
            ]
            values = [v for v in values if v is not None]
            if values:
                stat = mean_interval(np.log(values))
                center = np.exp(stat["mean"])
                ax.plot(i, center, marker="D", color=COLORS[key], markersize=7)
                ax.scatter(
                    i + np.linspace(-0.1, 0.1, len(values)),
                    values,
                    color=COLORS[key],
                    s=20,
                    alpha=0.6,
                )
                if stat["half_width"] is not None:
                    ax.errorbar(
                        i,
                        center,
                        yerr=[
                            [
                                center
                                - np.exp(stat["mean"] - stat["half_width"])
                            ],
                            [
                                np.exp(stat["mean"] + stat["half_width"])
                                - center
                            ],
                        ],
                        color="black",
                        capsize=4,
                    )
        ax.set(
            xticks=range(len(keys)),
            xticklabels=[LABELS[k] for k in keys],
            ylabel=label,
            yscale="log",
        )
        ax.grid(axis="y", alpha=0.25)
    selected = [
        d
        for d in differences
        if d["target"] == "hmr_4fs"
        and d["reference"] == "hmr_2fs"
        and d["ess_per_hour"] is not None
    ]
    for i, d in enumerate(selected):
        v = d["ess_per_hour"]
        axes[1, 1].errorbar(
            v["ratio"],
            i,
            xerr=[[v["ratio"] - v["low"]], [v["high"] - v["ratio"]]],
            fmt="o",
            color="C1",
            capsize=3,
        )
    axes[1, 1].set(
        yticks=range(len(selected)),
        yticklabels=[d["observable"] for d in selected],
        xlabel="HMR 4 fs / HMR 2 fs: ESS per wall time",
        xscale="log",
    )
    axes[1, 1].axvline(1, color="gray")
    axes[1, 1].axvline(2, color="C3", linestyle="--")
    axes[1, 1].set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    axes[1, 1].grid(alpha=0.25)
    fig.savefig(root / "sampling_efficiency.png", dpi=180)
    plt.close(fig)
    lines += [
        "",
        "## Distribution and rare-state diagnostics",
        "",
        "Mean +/- nominal 95% Student-t interval across the three seed means (df=2). Positive-phi observations are correlated, not independent visits. Phi core visits count entries into [30,150] degrees after filtering gaps between that core and [-150,-30], including an initially occupied positive core. Warning thresholds (ESS<50, fewer than 20 psi core alternations, 20 positive-phi observations or 10 positive-phi core visits) flag fragile estimates; passing them does not establish convergence.",
        "",
        "| condition | T K | U kcal/mol | p(psi_A) | p(phi>0) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in keys:
        stats = [
            aggregates[key][field]
            for field in (
                "temperature",
                "potential",
                "psi_A_population",
                "phi_positive_population",
            )
        ]
        lines.append(
            "| "
            + LABELS[key]
            + " | "
            + " | ".join(
                f"{s['mean']:.6g} +/- {s['half_width']:.3g}" for s in stats
            )
            + " |"
        )
    lines += [
        "",
        "Paired target-minus-reference differences (same seed; nominal 95% t repeatability intervals, df=2). These do not define an equivalence margin or prove absence of bias.",
        "",
        "| target / reference | delta T K | delta U kcal/mol | delta p(psi_A) | delta p(phi>0) |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, reference in (
        ("hmr_2fs", "normal_2fs"),
        ("hmr_4fs", "hmr_2fs"),
        ("hmr_4fs", "normal_2fs"),
    ):
        refs = {r["seed"]: r for r, _ in groups[reference]}
        stats = [
            mean_interval(
                [r[field] - refs[r["seed"]][field] for r, _ in groups[target]]
            )
            for field in (
                "temperature",
                "potential",
                "psi_A_population",
                "phi_positive_population",
            )
        ]
        lines.append(
            "| "
            + target
            + "/"
            + reference
            + " | "
            + " | ".join(
                f"{s['mean']:.6g} +/- {s['half_width']:.3g}" for s in stats
            )
            + " |"
        )
    lines += [
        "",
        "| run | positive-phi observations | positive-phi core visits | warnings |",
        "|---|---:|---:|---|",
    ]
    for key in keys:
        for r, _ in groups[key]:
            lines.append(
                f"| {r['name']} | {r['positive_phi_samples']}/{r['retained_samples']} | {r['positive_phi_core_visits']} | {'; '.join(r['diagnostic_warnings']) or 'none'} |"
            )
    lines += [
        "",
        "Small dots are seed values; diamonds are geometric means with nominal 95% three-seed log-scale repeatability intervals. Axes use logarithmic scales. Right-bottom intervals use paired log ratios; dashed line marks 2x.",
        "",
        "![sampling efficiency](sampling_efficiency.png)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    cutoffs = [
        r["metrics"]["psi_A"]["cutoff_ps"]
        for group in groups.values()
        for r, _ in group
        if r["metrics"]["psi_A"]["cutoff_ps"] is not None
    ]
    max_lag = min(
        max([50] + cutoffs),
        0.25
        * min(r["analyzed_ps"] for group in groups.values() for r, _ in group),
    )
    for key in keys:
        group = groups[key]
        acfs = [d["acf_psi_A"] for _, d in group if "acf_psi_A" in d]
        if acfs:
            length = min(
                min(map(len, acfs)),
                round(max_lag / group[0][0]["sample_ps"]) + 1,
            )
            a = np.array([x[:length] for x in acfs])
            time = np.arange(length) * group[0][0]["sample_ps"]
            axes[0].plot(
                time, a.mean(axis=0), label=LABELS[key], color=COLORS[key]
            )
            axes[0].fill_between(
                time,
                a.mean(axis=0) - a.std(axis=0),
                a.mean(axis=0) + a.std(axis=0),
                alpha=0.1,
                color=COLORS[key],
            )
        for r, d in group:
            indicator = (np.cos(d["psi"]) > 0).astype(float)
            axes[1].plot(
                d["elapsed_ps"] / 1000,
                np.cumsum(indicator) / np.arange(1, len(indicator) + 1),
                color=COLORS[key],
                alpha=0.6,
                linewidth=0.8,
            )
    axes[0].set(xlabel="Lag (ps)", ylabel="psi basin autocorrelation")
    axes[0].legend()
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[1].set(
        xlabel="Elapsed production time (ns)",
        ylabel="Running p(psi basin A) after burn-in",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(root / "sampling_correlation.png", dpi=180)
    plt.close(fig)
    lines += [
        "",
        "ACF shading is one between-seed SD. Running populations show each seed separately after the discarded first 20%.",
        "",
        "![correlation and population](sampling_correlation.png)",
    ]
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(4 * len(keys), 3.5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, key in zip(axes.flat, keys):
        phi = np.concatenate([d["phi"] for _, d in groups[key]])
        psi = np.concatenate([d["psi"] for _, d in groups[key]])
        hist, x, y = np.histogram2d(
            phi, psi, bins=36, range=[[-np.pi, np.pi], [-np.pi, np.pi]]
        )
        free = np.full_like(hist, np.nan)
        mask = hist >= 5
        free[mask] = -0.0019872041 * 300 * np.log(hist[mask] / hist.max())
        mesh = ax.pcolormesh(
            np.degrees(x),
            np.degrees(y),
            free.T,
            vmin=0,
            vmax=4,
            cmap="viridis",
            rasterized=True,
        )
        ax.set(
            title=LABELS[key],
            xlabel="phi (degrees)",
            ylabel="psi (degrees)",
            xticks=[-180, 0, 180],
            yticks=[-180, 0, 180],
        )
    fig.colorbar(mesh, ax=list(axes.flat), label="-kBT ln(P/Pmax), kcal/mol")
    fig.savefig(root / "sampling_free_energy.png", dpi=180)
    plt.close(fig)
    lines += [
        "",
        "![finite-sample free energy](sampling_free_energy.png)",
        "",
        "Bins with fewer than five combined observations are masked. These are finite-sample occupancy maps, not converged free-energy surfaces.",
        "",
        "## Stationarity and limits",
        "",
        "| condition | observable | split Rhat |",
        "|---|---|---:|",
    ]
    for key in keys:
        for obs in OBSERVABLES:
            lines.append(
                f"| {key} | {obs} | {aggregates[key]['metrics'][obs]['split_rhat']} |"
            )
    lines += [
        "",
        "### Primary ESS estimator sensitivity",
        "",
        "Nonoverlapping batch means provide an independent variance-estimator cross-check. Each block size needs at least eight batches; long blocks still have large uncertainty and short blocks may miss slow correlation. Geyer truncation is a practical single-series estimate for these MD observables, not a convergence theorem for inertial Langevin trajectories. Agreement between estimators does not prove convergence.",
        "",
        "| run | Geyer ESS | 50 ps batch ESS | 100 ps batch ESS | 200 ps batch ESS |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in keys:
        for r, _ in groups[key]:
            values = [r["metrics"]["psi_A"]["ess"]] + list(
                r["psi_A_batch_ess"].values()
            )
            lines.append(
                "| "
                + r["name"]
                + " | "
                + " | ".join(
                    f"{v:.5g}" if v is not None else "undefined" for v in values
                )
                + " |"
            )
    lines += [
        "",
        "Ratios are interpretable only for observables with adequate transitions and stable populations. ESS for a rapidly fluctuating observable does not establish sampling of rare phi-positive states or other slow modes. A small split Rhat is not a proof of convergence. Equal populations within uncertainty are not an equivalence test; timestep bias must be assessed separately. Constraints are checked only in stored float32 frames at 10 ps intervals. No general FEP, protein, kinetic-rate, or universal timestep conclusion follows from this small system.",
        "",
    ]
    (root / "sampling_report.md").write_text("\n".join(lines))
    (root / "sampling_analysis_provenance.json").write_text(
        json.dumps(
            dict(
                script_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (
                        Path(__file__),
                        Path(__file__).with_name("run_sampling.py"),
                    )
                },
                planned_runs=len(expected),
                completed_runs=sum(map(len, groups.values())),
                binary_sha256=provenance["binary_sha256"],
                numpy_version=np.__version__,
                matplotlib_version=matplotlib.__version__,
                burn_fraction=0.2,
                primary_observable="psi_A",
                observables=OBSERVABLES,
            ),
            indent=2,
        )
    )
    print(root / "sampling_report.md")


if __name__ == "__main__":
    main()
