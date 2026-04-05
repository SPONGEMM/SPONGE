"""Gradient speed benchmark: measures SPONGE gradient computation time.

Usage:
    pixi run -e dev-cpu python benchmarks/bench_grad_speed.py
    pixi run -e dev-cuda12 python benchmarks/bench_grad_speed.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STATICS = Path(__file__).resolve().parent / "comparison/tests/pyscf/statics"
SPONGE_BIN = os.environ.get("SPONGE_BIN", "SPONGE")

# (case_name, model_chemistry, n_steps, n_repeats, density_fit)
BENCH_CASES = [
    ("h2o_sto3g", "HF/sto-3g", 20, 3, False),
    ("h2o_631g", "HF/6-31g", 20, 3, False),
    ("ch4_sto3g", "HF/sto-3g", 20, 3, False),
    ("ch4_631g", "HF/6-31g", 10, 3, False),
    ("ch4_def2svp", "HF/def2-svp", 10, 3, False),
    ("benzene_sto3g", "HF/sto-3g", 5, 3, False),
    ("benzene_631g", "HF/6-31g", 3, 3, False),
    ("benzene_def2svp", "HF/def2-svp", 2, 2, False),
    ("benzene_def2tzvp", "HF/def2-tzvp", 1, 2, False),
    # RI (density fitting) cases - exercise BLAS and 3c cache optimizations
    ("h2o_def2svp", "HF/def2-svp", 10, 3, True),
    ("ch4_def2svp", "HF/def2-svp", 10, 3, True),
    ("benzene_sto3g", "HF/sto-3g", 5, 3, True),
    ("benzene_631g", "HF/6-31g", 3, 3, True),
]

WALL_TIME_RE = re.compile(r"Core Run Wall Time:\s+([\d.]+)\s+seconds")
FORCE_TIME_RE = re.compile(r"Calculate_Force\s*\|\s*([\d.]+)\s+seconds")


def setup_case(case_name, model_chem, n_steps, density_fit=False):
    """Create temp dir with gradient-enabled input."""
    src = STATICS / case_name / "sponge"
    if not src.exists():
        return None
    tmpdir = tempfile.mkdtemp(prefix=f"bench_{case_name}_")
    for f in src.iterdir():
        if f.name == "mdin.txt" or f.is_dir():
            continue
        shutil.copy(f, tmpdir)

    # Rewrite coordinate.txt to single-line box format (header = natom only)
    coord_lines = (
        open(os.path.join(tmpdir, "coordinate.txt")).read().strip().splitlines()
    )
    natom = int(coord_lines[0].split()[0])
    atoms = coord_lines[1 : 1 + natom]
    # Join remaining box lines into one
    box_parts = []
    for line in coord_lines[1 + natom :]:
        box_parts.extend(line.split())
    with open(os.path.join(tmpdir, "coordinate.txt"), "w") as f:
        f.write(f"{natom}\n")
        for a in atoms:
            f.write(a + "\n")
        f.write(" ".join(box_parts) + "\n")

    # Write mdin with gradient (minimization mode)
    with open(os.path.join(tmpdir, "mdin.txt"), "w") as f:
        f.write(f"{case_name} gradient bench\n")
        f.write("mode = minimization\n")
        f.write("minimization_dynamic_dt = 1\n")
        f.write(f"step_limit = {n_steps}\n")
        f.write("coordinate_in_file = coordinate.txt\n")
        f.write("qc_type_in_file = qc_type.txt\n")
        f.write("mass_in_file = mass.txt\n")
        f.write("charge_in_file = charge.txt\n")
        f.write(f"qc_model_chemistry = {model_chem}\n")
        if density_fit:
            f.write("qc_density_fit = 1\n")
        f.write(f"write_mdout_interval = {n_steps}\n")
        f.write("write_trajectory_interval = 0\n")
    return tmpdir


def run_bench(tmpdir):
    """Run SPONGE and return (wall_time, force_time)."""
    result = subprocess.run(
        [SPONGE_BIN, "-mdin", "mdin.txt"],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + "\n" + result.stderr

    # Extract timing even if process crashed during cleanup (e.g. RI double-free)
    wall_match = WALL_TIME_RE.search(output)
    force_match = FORCE_TIME_RE.search(output)
    if result.returncode != 0 and not wall_match:
        print(f"  FAILED (rc={result.returncode})")
        print(output[-500:])
        return None, None
    wall = float(wall_match.group(1)) if wall_match else None
    force = float(force_match.group(1)) if force_match else None
    return wall, force


def main():
    print(f"SPONGE binary: {shutil.which(SPONGE_BIN) or SPONGE_BIN}")
    print(
        f"{'Case':<25} {'Steps':>5} {'Wall(ms)':>10} {'Force(ms)':>10} "
        f"{'Per-step(ms)':>12}"
    )
    print("-" * 70)

    for case_name, model_chem, n_steps, n_repeats, density_fit in BENCH_CASES:
        label = f"{case_name}{'(RI)' if density_fit else ''}"
        tmpdir = setup_case(case_name, model_chem, n_steps, density_fit)
        if tmpdir is None:
            print(f"{label:<25} SKIP (not found)")
            continue

        best_wall = float("inf")
        best_force = float("inf")
        for _ in range(n_repeats):
            wall, force = run_bench(tmpdir)
            if wall is not None:
                best_wall = min(best_wall, wall)
            if force is not None:
                best_force = min(best_force, force)

        shutil.rmtree(tmpdir, ignore_errors=True)

        if best_wall == float("inf"):
            print(f"{label:<25} FAILED")
            continue

        wall_ms = best_wall * 1000
        force_ms = best_force * 1000 if best_force != float("inf") else 0
        per_step = force_ms / n_steps if force_ms > 0 else wall_ms / n_steps
        print(
            f"{label:<25} {n_steps:>5} {wall_ms:>10.1f} {force_ms:>10.1f} "
            f"{per_step:>12.2f}"
        )


if __name__ == "__main__":
    main()
