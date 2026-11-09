"""Native snapshots must preserve host-only subhill deposits between steps."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np


def check(binary, output, vds):
    output.mkdir(parents=True, exist_ok=True)
    (output / "mass.txt").write_text("4\n12\n12\n12\n12\n")
    (output / "charge.txt").write_text("4\n0\n0\n0\n0\n")
    (output / "coordinate.txt").write_text(
        "4\n0.5 0 0\n4 1 0\n8 0 1\n12 1 1\n999 999 999 90 90 90\n"
    )
    (output / "cv.toml").write_text(
        '[x]\nCV_type = "position_x"\natom = 0\n'
        '[meta]\nCV = ["x"]\nsubhill = true\nkde = 0\n'
        "CV_period = [0]\nCV_sigma = [0.2]\n"
        "CV_minimal = [0]\nCV_maximum = [4]\nCV_grid = [4]\n"
        "height = 1.0\nwelltemp_factor = 1000000000.0\n"
        "potential_update_interval = 1\n"
    )
    coordinates = np.array(
        [[0.5, 0, 0], [4, 1, 0], [8, 0, 1], [12, 1, 1]], dtype=np.float32
    )
    np.stack([coordinates] * 4).tofile(output / "frames.dat")
    np.savetxt(output / "frames.box", [[999, 999, 999, 90, 90, 90]] * 4)
    settings = {
        "mode": "rerun",
        "step_limit": 5,
        "dt": 0.00001,
        "rerun_frame_limit": 4,
        "rerun_start": 0,
        "rerun_strip": 0,
        "crd": "frames.dat",
        "box": "frames.box",
        "rerun_output_crd": "output.dat",
        "rerun_output_box": "output.box",
        "pbc": False,
        "cutoff": 999.0,
        "skin": 2.0,
        "mass_in_file": "mass.txt",
        "charge_in_file": "charge.txt",
        "coordinate_in_file": "coordinate.txt",
        "cv_in_file": "cv.toml",
        "write_information_interval": 1,
        "write_trajectory_interval": 1,
        "print_zeroth_frame": False,
    }
    (output / "mdin.spg.toml").write_text(
        "\n".join(
            f"{key} = {json.dumps(value)}" for key, value in settings.items()
        )
        + '\n[output.h5.trajectory]\npath = "run.spg.h5md"\n'
        + f"vds = {str(vds).lower()}\nchunk_size = 2\n"
        + '[output.h5.observable]\npath = "run.obs.spg.h5md"\n'
    )
    result = subprocess.run(
        [str(binary), "-mdin", "mdin.spg.toml"],
        cwd=output,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    (output / "execution.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    group = "/parameters/sponge/metadynamics/meta/potential_snapshots"
    for filename in ["run.spg.h5md", "run.obs.spg.h5md"]:
        with h5py.File(output / filename) as handle:
            snapshots = handle[group]
            labels = [
                v.decode() if isinstance(v, bytes) else v
                for v in snapshots["value_labels"][:]
            ]
            values = snapshots["values"][:]
            raw = values[..., labels.index("potential_raw")]
            # Each step deposits one unit at x=0.5. Export must neither erase
            # that deposit nor prevent accumulation on the next step.
            assert len(raw) >= 3, raw.shape
            np.testing.assert_allclose(
                raw.sum(axis=1),
                np.arange(1, len(raw) + 1),
                rtol=1e-5,
                atol=1e-5,
            )
            assert np.isfinite(values).all()
            backup = values[..., labels.index("potential_backup")]
            assert np.all(backup[-1] >= 0)
            assert backup[-1, 0] > backup[0, 0]
    print(f"PASS: subhill snapshots ({'vds' if vds else 'single'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sponge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for vds in [False, True]:
        check(
            args.sponge.resolve(),
            args.output.resolve() / ("vds" if vds else "single"),
            vds,
        )
