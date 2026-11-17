"""Compare SPONGE core-pair restoration with its native ordinary and 1–4 paths."""
from array import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

engine = Path(sys.argv[1]).resolve()
temporary = tempfile.TemporaryDirectory(prefix="ordinary-restoration-") if len(sys.argv) == 2 else None
output = (Path(temporary.name) / "cases") if temporary else Path(sys.argv[2]).resolve()
output.mkdir(exist_ok=False)
lj_a, lj_b, lj_14, kee_14 = 59500., 120., .5, .8333333
case_weights = {
    "ordinary": (False, 0., 0.),
    "excluded": (True, 0., 0.),
    "native14": (True, 0., 1.),
    "restored": (True, 1., 0.),
    "partial_excluded": (True, .4, 0.),
    "partial_native14": (True, .4, .6),
}
geometries = {"near": (1., 5.), "beyond_cutoff": (1., 11.), "periodic_image": (1., 21.)}
matches = {}
for geometry, xs in geometries.items():
    results = {}
    for name, (exclude, ordinary_fraction, exception_fraction) in case_weights.items():
        work = output / geometry / name
        work.mkdir(parents=True)
        coords = [(x, 2., 3.) for x in xs]
        box = "24 24 24 90 90 90"
        raw = {
            "coordinate": "2\n" + "".join(f"{x} {y} {z}\n" for x, y, z in coords) + box + "\n",
            "mass": "2\n12.011\n12.011\n",
            "charge": "2\n4.2\n-4.2\n",
            "residue": "2 2\n1\n1\n",
            "exclude": "2 1\n1 1\n0\n" if exclude else "2 0\n0\n0\n",
            "LJ_soft_core": f"2 1 1\n{lj_a}\n{lj_b}\n{lj_a}\n{lj_b}\n0 0\n0 0\n",
            "subsys_division": "2\n0\n0\n",
        }
        if exception_fraction:
            factor = exception_fraction * lj_14
            raw["nb14_extra"] = f"1\n0 1 {factor * lj_a} {factor * lj_b} {exception_fraction * kee_14}\n"
        for suffix, value in raw.items():
            (work / f"TMP_{suffix}.txt").write_text(value)
        with (work / "one.dat").open("wb") as handle:
            array("f", (v for row in coords for v in row)).tofile(handle)
        (work / "one.box").write_text(box + "\n")
        args = [str(engine), "-mode", "rerun", "-default_in_file_prefix", "TMP",
                "-crd", "one.dat", "-box", "one.box", "-cutoff", "8",
                "-lambda_lj", ".5", "-mdout", "one.mdout", "-mdinfo", "one.mdinfo",
                "-rerun_output_crd", "out.dat", "-rerun_output_box", "out.box", "-frc", "one.frc", "-print_pressure", "true", "-write_trajectory_interval", "1"]
        if ordinary_fraction:
            (work / "correction.txt").write_text(
                f"1\n0 1 {ordinary_fraction * lj_a} {ordinary_fraction * lj_b} {ordinary_fraction}\n"
            )
            args += ["-fep_core_ordinary_extra_in_file", "correction.txt"]
        with (work / "engine.log").open("w") as handle:
            subprocess.run(args, cwd=work, env={**os.environ, "OMP_NUM_THREADS": "1"},
                           stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=30)
        lines = (work / "one.mdout").read_text().splitlines()
        d = {k: float(v) for k, v in zip(lines[0].split(), lines[-1].split(), strict=True)}
        forces = array("f")
        with (work / "one.frc").open("rb") as handle:
            forces.fromfile(handle, 6)
        results[name] = {"energy": d["eff_pot"], "fx0": forces[0], "fx1": forces[3],
                         **{key: d[key] for key in ("Pxx", "Pyy", "Pzz") if key in d}}
    comparisons = {}
    for case, target, w in (("restored", "ordinary", 1.), ("partial_excluded", "excluded", .4),
                            ("partial_native14", "native14", .4)):
        errors = {key: results[case][key] - (w * val + (1-w)*results[target][key])
                  for key, val in results["ordinary"].items()}
        for key, error in errors.items():
            tolerance = .03 if key.startswith("P") else 2e-5
            assert math.isfinite(error) and abs(error) < tolerance, (geometry, case, key, error, results)
        comparisons[case] = errors
    matches[geometry] = {"states": results, "errors": comparisons}
(output / "restoration-report.json").write_text(json.dumps(matches, indent=2) + "\n")
print({key: data["errors"] for key, data in matches.items()})

if temporary is not None:
    temporary.cleanup()
