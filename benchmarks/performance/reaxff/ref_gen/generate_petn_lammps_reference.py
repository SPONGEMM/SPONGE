#!/usr/bin/env python3

import shutil
import subprocess
import tempfile
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_lammps(case_dir: Path):
    result = subprocess.run(
        ["lmp", "-in", "in.lammps", "-log", "log.lammps"],
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            "LAMMPS failed while generating PETN reference.\n"
            f"cwd={case_dir}\n"
            f"returncode={result.returncode}\n"
            f"output tail:\n{output[-4000:]}"
        )


def main():
    repo_root = get_repo_root()
    case_root = (
        repo_root
        / "benchmarks"
        / "performance"
        / "reaxff"
        / "statics"
        / "petn_16240"
    )
    reference_root = case_root / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "petn_16240"
        shutil.copytree(case_root, run_dir)
        if (run_dir / "reference").exists():
            shutil.rmtree(run_dir / "reference")
        run_lammps(run_dir)

        for name in ("in.lammps", "log.lammps", "forces.dump", "charges.dump"):
            src = run_dir / name
            if not src.exists():
                raise FileNotFoundError(
                    f"Missing generated reference file: {src}"
                )
            shutil.copy2(src, reference_root / name)

    print(reference_root)


if __name__ == "__main__":
    main()
