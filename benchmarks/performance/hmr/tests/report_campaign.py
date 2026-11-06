"""Render inspectable tables and scientific figures from completed analyses."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "temperature": "T (K)",
    "potential": "U (kcal/mol)",
    "density": "density (g/mL)",
    "ca_rmsd": "CA RMSD (A)",
    "ca_rg": "CA Rg (A)",
    "ligand_rmsd": "ligand RMSD (A)",
    "native_contacts": "native contact fraction",
    "pressure": "pressure (bar)",
}


def mean_of(record, key):
    value = record.get(key)
    return value.get("mean") if isinstance(value, dict) else value


def number(value):
    return f"{value:.5g}" if value is not None else "unavailable"


def finite_stats(values):
    valid = np.array([v for v in values if v is not None and np.isfinite(v)])
    mean = float(np.mean(valid)) if len(valid) else None
    half = (
        float(4.3026527299 * np.std(valid, ddof=1) / np.sqrt(3))
        if len(valid) == 3
        else None
    )
    return mean, half, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    records = [
        json.loads(p.read_text())
        for p in sorted((root / "runs").glob("*/analysis.json"))
    ]
    screen = [r for r in records if r["name"].startswith("screen_")]
    production = [r for r in records if r["name"].startswith("production_")]
    provenance = json.loads((root / "provenance.json").read_text())
    lines = [
        "# Local GPU HMR campaign",
        "",
        f"Frozen engine: `{provenance['source_commit']}`; binary SHA-256 `{provenance['binary_sha256']}`.",
        "",
        "TYK2 / jmc_28 physical FEP endpoint, 31,247 atoms, 288 protein residues, 8,829 waters. "
        "RTX 4090, driver 595.71.05. HMR ratios 2.0 for XH1 and 2.5 for other XH groups; water masses unchanged.",
        "",
        "The default double-precision GPU SETTLE diagnostic failed at ordinary masses and 1 fs, including from SHAKE-projected starting coordinates: water geometry errors and elevated kinetic temperature. "
        "The main comparisons below use double SHAKE for the full, unchanged constraint set (`settle_disable=true`). "
        "They do not validate default SETTLE. Their original engine binary remained frozen throughout; a separate corrected build is evaluated below.",
        "",
        "All branches start at the same ordinary-mass coordinates after 200 ps of all-SHAKE pre-equilibration at 1 fs, 300 K and 1 bar, "
        "with newly generated constrained Maxwell velocities. Statistics omit the first 20% and step zero. "
        "RMSD is measured against these starting coordinates; ligand RMSD uses a protein CA fit. "
        "Pressure is unavailable for MC NPT because this revision skips its evaluation.",
        "The earlier SETTLE preparation diagnostics (including elevated temperatures) are retained but excluded from comparisons. "
        "After SHAKE preparation, the protein CA RMSD against the original complex is 1.036 A, "
        "protein-fitted ligand heavy RMSD is 0.990 A, and 75.5% of the original 147 near contacts remain. "
        "This is a local short-time comparison, not a claim of global protein equilibration.",
        "",
        "## 50 ps screen",
        "",
        "PASS below only excludes predefined catastrophic failures (temperature 100–600 K, sampled relative constraint error <1%, "
        "CA/ligand RMSD <10 A, finite and complete output). It is not an accuracy or ensemble equivalence criterion.",
        "",
        "| masses | dt fs | mode | process | screen | T K | CA RMSD A | ligand RMSD A | max constraint error | approximate NVE drift kBT/DOF/ns |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(screen, key=lambda x: (x["hmr"], x["dt_fs"], x["mode"])):
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        "HMR" if r["hmr"] else "normal",
                        r["dt_fs"],
                        r["mode"],
                        r["completed"],
                        r["catastrophic_screen_pass"],
                        number(mean_of(r, "temperature")),
                        number(mean_of(r, "ca_rmsd")),
                        number(mean_of(r, "ligand_rmsd")),
                        number(r.get("xh_max_relative_error_sampled")),
                        number(r.get("energy_drift_kbt_per_dof_ns")),
                    ],
                )
            )
            + " |"
        )
    if screen:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for hmr, marker in [(False, "o"), (True, "s")]:
            for mode, style in [("nve", "-"), ("nvt", "--")]:
                group = sorted(
                    [
                        r
                        for r in screen
                        if r["hmr"] == hmr and r["mode"] == mode
                    ],
                    key=lambda r: r["dt_fs"],
                )
                label = f"{'HMR' if hmr else 'normal'} {mode.upper()}"
                x = [r["dt_fs"] for r in group]
                for ax, key in zip(
                    axes.flat,
                    [
                        "temperature",
                        "energy_drift_kbt_per_dof_ns",
                        "ca_rmsd",
                        "xh_max_relative_error_sampled",
                    ],
                ):
                    y = [
                        mean_of(r, key)
                        if mean_of(r, key) is not None
                        else np.nan
                        for r in group
                    ]
                    ax.plot(x, y, style, marker=marker, label=label)
        for ax, ylabel in zip(
            axes.flat,
            [
                "Temperature (K)",
                "Approx. NVE drift (kBT/DOF/ns)",
                "Protein CA RMSD (A)",
                "Max sampled constraint relative error",
            ],
        ):
            ax.set(xlabel="Timestep (fs)", ylabel=ylabel)
            ax.grid(alpha=0.25)
        failed_steps = sorted(
            {
                r["dt_fs"]
                for r in screen
                if not r.get("finite_coordinates", False)
            }
        )
        if failed_steps:
            for ax in axes.flat:
                ax.axvspan(
                    min(failed_steps) - 0.25,
                    max(failed_steps) + 0.25,
                    color="red",
                    alpha=0.07,
                )
                ax.scatter(
                    failed_steps,
                    [0.96] * len(failed_steps),
                    marker="x",
                    color="black",
                    transform=ax.get_xaxis_transform(),
                )
                ax.set_xlim(0.5, max(failed_steps) + 0.5)
            axes[0, 0].text(
                0.97,
                0.88,
                "6-12 fs: nonfinite output",
                ha="right",
                transform=axes[0, 0].transAxes,
                fontsize=8,
            )
        axes[0, 0].axhline(300, color="gray", linewidth=0.8)
        axes[1, 1].set_yscale("log")
        axes[0, 0].legend(fontsize=8)
        fig.savefig(root / "timestep_screen.png", dpi=180)
        plt.close(fig)
        lines += ["", "![timestep screen](timestep_screen.png)"]
    groups = defaultdict(list)
    for r in production:
        groups[(r["mode"], r["hmr"], r["dt_fs"])].append(r)
    lines += [
        "",
        "## Replicated finite-duration comparisons",
        "",
        "Each cell gives the mean across seed means. Nominal 95% intervals use Student t "
        "(df=2 for three seeds); they describe finite-run replicate variability, not converged equilibrium uncertainty. "
        "A failure is retained in the table and must be investigated.",
        "",
        "| mode | masses | dt fs | seeds | ps/seed | all screen checks | T K | U kcal/mol | density g/mL | pressure bar | CA RMSD A | ligand RMSD A | native contacts |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    aggregates = []
    for (mode, hmr, dt), group in sorted(groups.items()):
        agg = dict(mode=mode, hmr=hmr, dt_fs=dt, seeds=len(group), metrics={})
        cells = []
        for metric in [
            "temperature",
            "potential",
            "density",
            "pressure",
            "ca_rmsd",
            "ligand_rmsd",
            "native_contacts",
        ]:
            mean, half, values = finite_stats(
                [mean_of(r, metric) for r in group]
            )
            agg["metrics"][metric] = dict(
                mean=mean,
                nominal_95_half_width=half,
                seed_means=values.tolist(),
                valid_seed_count=len(values),
            )
            cells.append(
                number(mean)
                + (f" ± {number(half)}" if half is not None else "")
            )
        aggregates.append(agg)
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        mode,
                        "HMR" if hmr else "normal",
                        dt,
                        len(group),
                        group[0]["duration_ps"],
                        all(r["catastrophic_screen_pass"] for r in group),
                        *cells,
                    ],
                )
            )
            + " |"
        )
    (root / "replicate_summary.json").write_text(
        json.dumps(aggregates, indent=2)
    )
    lines += [
        "",
        "## Differences across matched seeds",
        "",
        "Differences are target minus reference seed means; nominal 95% Student-t intervals use three paired seeds. "
        "These exploratory intervals are not corrected for multiple comparisons. HMR 1 fs versus normal 1 fs isolates mass changes; "
        "HMR 4/5 fs versus HMR 1 fs isolates timestep changes.",
        "",
        "| mode | target | reference | seeds | delta T K | delta U kcal/mol | delta density g/mL | delta pressure bar | delta CA RMSD A | delta ligand RMSD A | delta contact fraction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    differences = []
    for (mode, hmr, dt), group in sorted(groups.items()):
        if not hmr:
            continue
        reference_key = (mode, False, 1.0) if dt == 1 else (mode, True, 1.0)
        reference = {r["seed"]: r for r in groups.get(reference_key, [])}
        paired = [
            (r, reference[r["seed"]])
            for r in group
            if r["seed"] in reference
            and r["catastrophic_screen_pass"]
            and reference[r["seed"]]["catastrophic_screen_pass"]
        ]
        if len(paired) != 3:
            continue
        item = dict(
            mode=mode,
            target_dt_fs=dt,
            reference_hmr=reference_key[1],
            metrics={},
        )
        cells = []
        for metric in [
            "temperature",
            "potential",
            "density",
            "pressure",
            "ca_rmsd",
            "ligand_rmsd",
            "native_contacts",
        ]:
            mean, half, delta = finite_stats(
                [
                    mean_of(a, metric) - mean_of(b, metric)
                    if mean_of(a, metric) is not None
                    and mean_of(b, metric) is not None
                    else None
                    for a, b in paired
                ]
            )
            item["metrics"][metric] = dict(
                mean=mean,
                nominal_95_half_width=half,
                seed_differences=delta.tolist(),
            )
            cells.append(
                f"{number(mean)} ± {number(half)}"
                if half is not None
                else number(mean)
            )
        differences.append(item)
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        mode,
                        f"HMR {dt:g} fs",
                        "normal 1 fs" if dt == 1 else "HMR 1 fs",
                        3,
                        *cells,
                    ],
                )
            )
            + " |"
        )
    (root / "replicate_differences.json").write_text(
        json.dumps(differences, indent=2)
    )
    if groups:
        fig, axes = plt.subplots(
            3, 2, figsize=(11, 10), constrained_layout=True
        )
        grid = np.arange(1, 201)
        plot_metrics = [
            "temperature",
            "potential",
            "density",
            "ca_rmsd",
            "ligand_rmsd",
            "native_contacts",
        ]
        for (mode, hmr, dt), group in sorted(groups.items()):
            series = [
                dict(np.load(root / "runs" / r["name"] / "analysis_series.npz"))
                for r in group
                if r["catastrophic_screen_pass"]
            ]
            color = {
                False: "black",
                True: {1: "C0", 4: "C1", 5: "C3"}.get(dt, "C2"),
            }[hmr]
            label = f"{mode.upper()} {'HMR' if hmr else 'normal'} {dt:g} fs"
            for ax, metric in zip(axes.flat, plot_metrics):
                values = [
                    np.interp(
                        grid,
                        data[
                            "step"
                            if metric in ("temperature", "potential")
                            else "frame_step"
                        ]
                        * dt
                        / 1000,
                        data[metric],
                    )
                    for data in series
                ]
                if not values:
                    continue
                mean = np.mean(values, axis=0)
                std = np.std(values, axis=0)
                ax.plot(
                    grid,
                    mean,
                    "-" if mode == "nvt" else "--",
                    color=color,
                    label=label,
                    linewidth=1,
                )
                ax.fill_between(
                    grid, mean - std, mean + std, color=color, alpha=0.06
                )
        for ax, metric in zip(axes.flat, plot_metrics):
            ax.set(xlabel="Elapsed time (ps)", ylabel=METRICS[metric])
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            fontsize=8,
        )
        fig.savefig(
            root / "sampling_timeseries.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)
        lines += [
            "",
            "Time traces show seed means with one between-seed standard deviation, sampled on a 1 ps grid. "
            "These traces include the initial transient; table statistics discard the first 20%.",
            "",
            "![sampling time traces](sampling_timeseries.png)",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for (mode, hmr, dt), group in sorted(groups.items()):
            arrays = []
            for record in group:
                p = root / "runs" / record["name"] / "rdf.npz"
                if p.exists():
                    arrays.append(dict(np.load(p)))
            if arrays:
                axes[0].plot(
                    arrays[0]["radius"],
                    np.mean([a["rdf"] for a in arrays], axis=0),
                    label=f"{mode} {'HMR' if hmr else 'normal'} {dt:g} fs",
                )
            histograms = [
                dict(np.load(root / "runs" / r["name"] / "backbone.npz"))
                for r in group
                if (root / "runs" / r["name"] / "backbone.npz").exists()
            ]
            if histograms:
                edges = histograms[0]["edges"]
                axes[1].plot(
                    np.degrees((edges[:-1] + edges[1:]) / 2),
                    np.mean([h["histogram"] for h in histograms], axis=0),
                    label=f"{mode} {'HMR' if hmr else 'normal'} {dt:g} fs",
                )
        axes[0].set(xlabel="O-O distance (A)", ylabel="g(r)")
        axes[1].set(
            xlabel="Pooled protein backbone dihedral (degrees)",
            ylabel="Density (per radian)",
        )
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
        fig.savefig(root / "structure_distributions.png", dpi=180)
        plt.close(fig)
        lines += ["", "![structure distributions](structure_distributions.png)"]
    corrected = [r for r in records if r["name"].startswith("fixed_settle_")]
    if corrected:
        lines += [
            "",
            "## Independent double-precision GPU SETTLE correction",
            "",
            "CUDA Driver API queries of the frozen binary's embedded kernels found 124 registers/thread and a 512-thread maximum for double `settle_triangle`; "
            "double velocity projection used 72 registers/thread and supported at most 896 threads. Both were launched with 1024 threads on this GPU, exceeding the per-block resource budget. "
            "Single precision passed a separate 20 ps diagnostic. This failure is independent of HMR.",
            "The correction caps double triangle blocks at 256 threads, including initial velocity projection. It changes launch configuration only. "
            "The existing independent-reference constraint regression failed before the fix and passed afterward; corrected CPU and CUDA builds also passed the HMR unit/runtime regressions. "
            "The following separate runs enable default double SETTLE from the same prepared coordinates, with newly generated velocities. They are excluded from the 24-run production aggregates.",
            "",
            "| masses | dt fs | duration ps | screen | T K | max sampled constraint error |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for r in sorted(corrected, key=lambda r: (r["hmr"], r["dt_fs"])):
            lines.append(
                f"| {'HMR' if r['hmr'] else 'normal'} | {r['dt_fs']} | {r['duration_ps']} | "
                f"{r['catastrophic_screen_pass']} | {number(mean_of(r, 'temperature'))} | "
                f"{number(r.get('xh_max_relative_error_sampled'))} |"
            )
        lines += [
            "",
            "Corrected binary SHA-256: "
            + ", ".join(
                f"`{h}`"
                for h in sorted({r["binary_sha256"] for r in corrected})
            )
            + ". These short checks establish the local launch correction, not long-time sampling equivalence.",
        ]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "Constraint errors use float32 stored coordinates at approximately 1 ps spacing, not every integration step. "
        "NVE energy drift is reconstructed from printed leapfrog temperature; it is an approximate diagnostic. "
        "Water RDF uses a fixed 512-oxygen subset. Pooled backbone distributions can hide residue-specific changes. "
        "This duration cannot establish converged protein ensembles, binding free energies, physical kinetic rates, "
        "or a universal maximum timestep. HMR changes dynamics; apparent thermodynamic agreement is not a proof of identical sampling.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    print(root / "report.md")


if __name__ == "__main__":
    main()
