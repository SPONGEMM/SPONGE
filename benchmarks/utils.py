import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np


class Outputer:
    @staticmethod
    def prepare_output_case(
        statics_path,
        outputs_path,
        case_name,
        run_tag=None,
        *,
        mpi_np=None,
        run_name=None,
    ):
        static_case = Path(statics_path) / case_name
        if not static_case.exists():
            raise FileNotFoundError(f"Static case not found: {static_case}")

        case_dir = Path(outputs_path) / (run_tag or case_name)
        if run_name is not None:
            case_dir /= str(run_name)
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(static_case, case_dir)
        return case_dir

    @staticmethod
    def print_table(headers, rows, title=None):
        if not headers:
            return

        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))

        col_widths = [w + 2 for w in col_widths]
        row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
        divider = "-" * (sum(col_widths) + 3 * (len(headers) - 1))

        if title:
            print(f"\n{title}")
        else:
            print()
        print(divider)
        print(row_fmt.format(*headers))
        print(divider)
        for row in rows:
            print(row_fmt.format(*[str(v) for v in row]))
        print(divider)

    @staticmethod
    def summarize_series(series, burn_in=0):
        if burn_in >= len(series) - 1:
            raise ValueError(
                f"burn_in={burn_in} is too large for sample count {len(series)}"
            )
        sample = series[burn_in:]
        std = statistics.stdev(sample) if len(sample) > 1 else 0.0
        return {
            "sample_count": len(sample),
            "mean": statistics.fmean(sample),
            "std": std,
            "min": min(sample),
            "max": max(sample),
        }


class Extractor:
    @staticmethod
    def _read_mdout(case_dir, mdout_name):
        from Xponge.analysis import MdoutReader

        mdout_path = Path(case_dir) / mdout_name
        if not mdout_path.exists():
            raise FileNotFoundError(
                f"SPONGE mdout file not found: {mdout_path}"
            )
        return MdoutReader(str(mdout_path))

    @staticmethod
    def _require_scalar(values, field, mdout_path):
        if not hasattr(values, field):
            raise ValueError(
                f"SPONGE mdout does not contain '{field}' in {mdout_path}"
            )
        return float(getattr(values, field)[-1])

    @staticmethod
    def extract_sponge_potential(case_dir, mdout_name="mdout.txt"):
        mdout = Extractor._read_mdout(case_dir, mdout_name)
        return Extractor._require_scalar(mdout, "potential", mdout_name)

    @staticmethod
    def extract_sponge_pressure(case_dir, mdout_name="mdout.txt"):
        mdout = Extractor._read_mdout(case_dir, mdout_name)
        pressure_attr = "pressure"
        if hasattr(mdout, pressure_attr):
            return float(getattr(mdout, pressure_attr)[-1])

        values = []
        for key in ("Pxx", "Pyy", "Pzz"):
            if hasattr(mdout, key):
                values.append(float(getattr(mdout, key)[-1]))
        if values:
            return sum(values) / float(len(values))

        raise ValueError(
            f"SPONGE mdout does not contain pressure info in {mdout_name}"
        )

    @staticmethod
    def extract_sponge_stress(case_dir, mdout_name="mdout.txt"):
        mdout = Extractor._read_mdout(case_dir, mdout_name)
        return {
            key: (
                float(getattr(mdout, key)[-1])
                if hasattr(mdout, key)
                else float("nan")
            )
            for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
        }

    @staticmethod
    def extract_sponge_forces(case_dir, atom_numbers, frc_name="frc.dat"):
        frc_path = Path(case_dir) / frc_name
        if not frc_path.exists():
            raise FileNotFoundError(f"SPONGE force file not found: {frc_path}")

        raw = np.fromfile(frc_path, dtype=np.float32)
        frame_width = int(atom_numbers) * 3
        if frame_width == 0 or raw.size % frame_width != 0:
            raise ValueError(
                f"Invalid SPONGE force file shape: size={raw.size}, atom_numbers={atom_numbers}, "
                f"path={frc_path}"
            )
        return (
            raw[-frame_width:].reshape(int(atom_numbers), 3).astype(np.float64)
        )

    @staticmethod
    def parse_mdout_rows(mdout_path, columns, *, int_columns=("step",)):
        mdout = Extractor._read_mdout(
            Path(mdout_path).parent,
            Path(mdout_path).name,
        )
        missing = [column for column in columns if not hasattr(mdout, column)]
        if missing:
            raise ValueError(f"Missing columns {missing} in mdout {mdout_path}")

        selected = {column: getattr(mdout, column) for column in columns}
        lengths = [len(values) for values in selected.values()]
        if not lengths:
            raise ValueError(
                f"No columns selected for parsing mdout {mdout_path}"
            )
        if any(length <= 0 for length in lengths):
            raise ValueError(f"Empty mdout column found in {mdout_path}")
        max_rows = min(lengths)

        int_columns = set(int_columns)
        rows = []
        for idx in range(max_rows):
            row = {}
            try:
                for column, values in selected.items():
                    value = values[idx]
                    row[column] = (
                        int(value) if column in int_columns else float(value)
                    )
            except (ValueError, TypeError):
                continue
            rows.append(row)

        if not rows:
            raise ValueError(f"No frame data parsed from mdout {mdout_path}")
        return rows

    @staticmethod
    def read_first_field_int(path):
        lines = Path(path).read_text().splitlines()
        if not lines:
            raise ValueError(f"Empty file: {path}")
        fields = lines[0].split()
        if not fields:
            raise ValueError(f"Invalid file header: {path}")
        return int(fields[0])


class Runner:
    @staticmethod
    def run_command(
        cmd,
        *,
        cwd,
        timeout=None,
        env=None,
        input_text=None,
    ):
        cmd = [str(part) for part in cmd]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
            input=input_text,
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            print("\n[stdout]\n")
            print(result.stdout)
            print("\n[stderr]\n")
            print(result.stderr)
            raise RuntimeError(
                f"Command failed in {cwd} with code {result.returncode}\n"
                f"Command: {' '.join(cmd)}"
            )
        return output

    @staticmethod
    def run_sponge(
        case_dir,
        *,
        mpi_np=None,
        mdin_name="mdin.spg.toml",
        timeout=None,
        sponge_cmd=None,
        extra_args=(),
        env=None,
        input_text=None,
    ):
        cmd = [str(sponge_cmd or "SPONGE")]
        if mdin_name is not None:
            cmd.extend(["-mdin", mdin_name])
        cmd.extend(str(arg) for arg in extra_args)
        if mpi_np is not None:
            cmd = ["mpirun", "--oversubscribe", "-np", str(mpi_np)] + cmd
        return Runner.run_command(
            cmd,
            cwd=case_dir,
            timeout=timeout,
            env=env,
            input_text=input_text,
        )
