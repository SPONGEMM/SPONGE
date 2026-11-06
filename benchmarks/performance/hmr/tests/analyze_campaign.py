"""Analyze a bounded HMR campaign; process completion is not scientific acceptance."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.performance.amber.tests.utils import (
    compute_oo_rdf,
    read_box_trajectory,
    read_coordinate_trajectory,
)
from benchmarks.utils import Extractor

KB = 0.0019872041  # kcal mol^-1 K^-1


def minimum_image(delta, box):
    return delta - np.rint(delta / box) * box


def summarize(values):
    values = np.asarray(values, dtype=float)
    blocks = [
        np.mean(b)
        for b in np.array_split(values, min(5, len(values)))
        if len(b)
    ]
    return dict(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        block_se=float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))
        if len(blocks) > 1
        else None,
        blocks=len(blocks),
    )


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def molecular_coordinates(q, box, selection):
    ca = selection["protein_ca"].astype(int)
    p = q[ca].copy()
    p[1:] = p[0] + np.cumsum(minimum_image(np.diff(p, axis=0), box), axis=0)
    ligand = selection["ligand"].astype(int)
    local = {int(a): i for i, a in enumerate(ligand)}
    neighbors = {a: [] for a in local}
    for a, b in selection["bonds"][:, :2].astype(int):
        if a in local and b in local:
            neighbors[a].append(b)
            neighbors[b].append(a)
    whole = np.empty((len(ligand), 3))
    anchor = int(ligand[0])
    nearest = int(
        np.argmin(np.linalg.norm(minimum_image(q[ca] - q[anchor], box), axis=1))
    )
    whole[0] = p[nearest] + minimum_image(q[anchor] - q[ca[nearest]], box)
    seen, stack = {anchor}, [anchor]
    while stack:
        a = stack.pop()
        for b in neighbors[a]:
            if b not in seen:
                whole[local[b]] = whole[local[a]] + minimum_image(
                    q[b] - q[a], box
                )
                seen.add(b)
                stack.append(b)
    if len(seen) != len(ligand):
        raise ValueError("Ligand bond graph is disconnected")
    heavy = selection["protein_heavy"].astype(int)
    ca_res = {int(selection["residue"][a]): i for i, a in enumerate(ca)}
    anchors = np.array([ca_res[int(selection["residue"][a])] for a in heavy])
    protein = p[anchors] + minimum_image(q[heavy] - q[ca[anchors]], box)
    return p, whole, protein


def analyze(root, case, selection, reference, reference_box, rdf=False):
    status = json.loads((case / "status.json").read_text())
    result = dict(status)
    columns = ["step", "time", "temperature", "potential", "pressure"]
    try:
        rows = Extractor.parse_mdout_rows(case / "mdout.txt", columns)
        thermo = {k: np.array([r[k] for r in rows]) for k in columns}
        # The common parser can skip malformed rows: detect missing records explicitly.
        interval = max(1, round(0.1 / (status["dt_fs"] / 1000)))
        expected = np.arange(0, status["steps"] + 1, interval)
        result["mdout_complete"] = bool(
            np.array_equal(thermo["step"], expected)
        )
        result["finite_thermo"] = bool(
            all(np.isfinite(a).all() for a in thermo.values())
        )
        finite_rows = np.logical_and.reduce(
            [np.isfinite(a) for a in thermo.values()]
        )
        if not finite_rows.all():
            bad = int(np.flatnonzero(~finite_rows)[0])
            result["first_nonfinite_step"] = int(thermo["step"][bad])
            result["first_nonfinite_elapsed_ps"] = float(
                thermo["step"][bad] * status["dt_fs"] / 1000
            )
        mask = (thermo["step"] > 0) & (thermo["step"] >= status["steps"] * 0.2)
        for k in ("temperature", "potential", "pressure"):
            result[k] = summarize(thermo[k][mask])
        # MC pressure is not computed in this engine; do not report its initial zero as pressure.
        if status["mode"] == "npt":
            result["pressure"] = None
        log = (case / "run.log").read_text(errors="replace")
        matches = re.findall(r"freedom (\d+)", log)
        if matches and status["mode"] == "nve":
            dof = int(matches[-1])
            energy = thermo["potential"] + dof * KB * thermo["temperature"] / 2
            t = thermo["time"][mask]
            slope = np.polyfit(t - t[0], energy[mask], 1)[0]
            result["energy_drift_kcal_mol_ns"] = float(slope * 1000)
            result["energy_drift_kbt_per_dof_ns"] = float(
                slope * 1000 / (dof * KB * 300)
            )
            # Leapfrog kinetic estimator and printed precision limit this diagnostic.
            result["energy_estimator"] = (
                "U + DOF*kB*T/2 from printed leapfrog kinetic temperature"
            )
        n = len(selection["mass"])
        trajectory = read_coordinate_trajectory(case / "mdcrd.dat", n)
        boxes = read_box_trajectory(case / "mdbox.txt")
        stride = max(1, round(1 / (status["dt_fs"] / 1000)))
        # print_zeroth_frame controls mdout; native coordinate/box streams start at stride.
        result["trajectory_complete"] = bool(
            len(trajectory) == len(boxes) == status["steps"] // stride
        )
        if len(trajectory) != len(boxes):
            raise ValueError("Coordinate and box frame counts disagree")
        if not np.allclose(boxes[:, 3:], 90):
            raise ValueError("This suite currently requires orthorhombic boxes")
        result["finite_coordinates"] = bool(np.isfinite(trajectory).all())
        result["finite_boxes"] = bool(np.isfinite(boxes).all())
        if not result["finite_coordinates"]:
            bad_frame = int(
                np.flatnonzero(~np.isfinite(trajectory).all(axis=(1, 2)))[0]
            )
            result["first_nonfinite_coordinate_elapsed_ps"] = float(
                (bad_frame + 1) * stride * status["dt_fs"] / 1000
            )
        if (
            not result["finite_coordinates"]
            or not result["finite_boxes"]
            or not np.all(boxes[:, :3] > 0)
        ):
            raise ValueError("Nonfinite coordinates or invalid box")
        frame_steps = (np.arange(len(trajectory)) + 1) * stride
        keep = frame_steps >= max(1, status["steps"] * 0.2)
        volume = np.prod(boxes[:, :3], axis=1)
        result["density"] = summarize(
            (selection["mass"].sum() * 1.6605390666 / volume)[keep]
        )
        # Native bonded connectivity uses the final record per atom pair. This
        # endpoint contains zero-force ligand placeholders before active bonds.
        unique = {
            tuple(sorted(row[:2].astype(int))): row
            for row in selection["bonds"]
        }
        bonds = np.array(list(unique.values()))
        a, b = bonds[:, 0].astype(int), bonds[:, 1].astype(int)
        xh = (selection["mass"][a] < 2) | (selection["mass"][b] < 2)
        a, b, length = a[xh], b[xh], bonds[xh, 3]
        residual = []
        ca_ref, lig_ref, prot_ref = molecular_coordinates(
            reference, reference_box, selection
        )
        ref_center = ca_ref.mean(axis=0)
        ca_ref -= ref_center
        lh = np.flatnonzero(
            np.isin(selection["ligand"], selection["ligand_heavy"])
        )
        contacts = np.argwhere(
            np.linalg.norm(prot_ref[:, None] - lig_ref[None, lh], axis=-1) < 4.5
        )
        metrics = {
            k: []
            for k in ("ca_rmsd", "ca_rg", "ligand_rmsd", "native_contacts")
        }
        for q, box in zip(trajectory, boxes[:, :3]):
            d = np.linalg.norm(minimum_image(q[a] - q[b], box), axis=1)
            residual.append(np.max(np.abs(d / length - 1)))
            ca, ligand, protein = molecular_coordinates(q, box, selection)
            center = ca.mean(axis=0)
            ca -= center
            u, _, vt = np.linalg.svd(ca.T @ ca_ref)
            correction = np.diag([1, 1, np.linalg.det(u @ vt)])
            rotation = u @ correction @ vt
            metrics["ca_rmsd"].append(
                np.sqrt(np.mean(np.sum((ca @ rotation - ca_ref) ** 2, axis=1)))
            )
            metrics["ca_rg"].append(np.sqrt(np.mean(np.sum(ca**2, axis=1))))
            metrics["ligand_rmsd"].append(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                (ligand[lh] - center) @ rotation
                                - (lig_ref[lh] - ref_center)
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            )
            distances = np.linalg.norm(
                protein[contacts[:, 0]] - ligand[lh[contacts[:, 1]]], axis=1
            )
            metrics["native_contacts"].append(
                np.mean(distances < 4.5) if len(contacts) else np.nan
            )
        result["xh_max_relative_error_sampled"] = float(np.max(residual))
        # Store backbone angle distributions only for real, bonded protein paths.
        quads = selection["backbone_quads"].astype(int)
        edges = set(unique)
        valid = [
            q
            for q in quads
            if np.all(selection["residue"][q] < len(selection["protein_ca"]))
            and all(
                tuple(sorted((int(a), int(b)))) in edges
                for a, b in zip(q[:-1], q[1:])
            )
        ]
        quads = np.asarray(valid, dtype=int)
        vectors = minimum_image(
            trajectory[keep, :, :][:, quads[:, 1:], :]
            - trajectory[keep, :, :][:, quads[:, :-1], :],
            boxes[keep, None, None, :3],
        )
        b0, b1, b2 = -vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
        v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
        angles = np.arctan2(
            np.sum(np.cross(b1, v) * w, axis=-1), np.sum(v * w, axis=-1)
        )
        hist, edges_angle = np.histogram(
            angles, bins=72, range=(-np.pi, np.pi), density=True
        )
        np.savez_compressed(
            case / "backbone.npz",
            histogram=hist,
            edges=edges_angle,
            angles=angles,
            quads=quads,
        )
        for k, v in metrics.items():
            result[k] = summarize(np.asarray(v)[keep])
        np.savez_compressed(
            case / "analysis_series.npz",
            **thermo,
            frame_step=frame_steps,
            **{k: np.asarray(v) for k, v in metrics.items()},
            xh_relative_error=residual,
            density=selection["mass"].sum() * 1.6605390666 / volume,
        )
        if rdf:
            radii, values, oxygen = compute_oo_rdf(
                trajectory[keep],
                boxes[keep],
                selection["water_oxygen"],
                r_max=8,
                bin_width=0.1,
                max_oxygen_samples=512,
            )
            np.savez_compressed(
                case / "rdf.npz", radius=radii, rdf=values, oxygen=oxygen
            )
        result["catastrophic_screen_pass"] = bool(
            status["completed"]
            and result["mdout_complete"]
            and result["trajectory_complete"]
            and result["finite_thermo"]
            and result["finite_coordinates"]
            and result["temperature"]["min"] >= 100
            and result["temperature"]["max"] <= 600
            and result["xh_max_relative_error_sampled"] < 0.01
            and result["ca_rmsd"]["max"] < 10
            and result["ligand_rmsd"]["max"] < 10
        )
    except Exception as error:
        result.update(
            analysis_error=repr(error), catastrophic_screen_pass=False
        )
    result = json_finite(result)
    (case / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", default="equilibrated")
    parser.add_argument("--pattern", default="screen_*")
    parser.add_argument("--rdf", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    selection = dict(np.load(root / "selection.npz"))
    n = len(selection["mass"])
    original_mass = np.loadtxt(
        root / "statics/tyk2_jmc28/TMP_mass.txt", skiprows=1
    )
    if not np.array_equal(selection["mass"], original_mass):
        raise ValueError(
            "Selections must retain original topology masses, including for HMR runs"
        )
    ca_residues = selection["residue"][selection["protein_ca"].astype(int)]
    if not np.array_equal(ca_residues, np.arange(len(ca_residues))):
        raise ValueError(
            "This fixed fixture requires ordered contiguous protein residue indices"
        )
    ref_path = root / "runs" / args.reference / "restart_coordinate.txt"
    reference = np.loadtxt(ref_path, skiprows=1, max_rows=n)
    reference_box = np.loadtxt(ref_path, skiprows=n + 1)[:3]
    results = [
        analyze(root, p, selection, reference, reference_box, args.rdf)
        for p in sorted((root / "runs").glob(args.pattern))
        if (p / "status.json").exists()
    ]
    (root / "analysis_summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False)
    )
    fields = [
        "name",
        "completed",
        "catastrophic_screen_pass",
        "temperature",
        "potential",
        "pressure",
        "density",
        "ca_rmsd",
        "ca_rg",
        "ligand_rmsd",
        "native_contacts",
        "xh_max_relative_error_sampled",
        "energy_drift_kbt_per_dof_ns",
        "analysis_error",
    ]
    with (root / "analysis_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    k: r.get(k, {}).get("mean")
                    if isinstance(r.get(k), dict)
                    else r.get(k)
                    for k in fields
                }
            )
    for r in results:
        print(
            r["name"],
            "PASS" if r["catastrophic_screen_pass"] else "FAIL",
            "T",
            r.get("temperature", {}).get("mean"),
            "CA",
            r.get("ca_rmsd", {}).get("mean"),
            "XH",
            r.get("xh_max_relative_error_sampled"),
            r.get("analysis_error", ""),
        )


if __name__ == "__main__":
    main()
