"""Run isolated baseline, synchronized control and T-REMD on alanine dipeptide."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def run(
    engine,
    manager,
    source,
    output,
    steps,
    interval,
    devices,
    temperatures,
    seed,
    pressure=None,
):
    result = {}
    for mode in ("independent", "sync", "exchange"):
        root = output / mode
        root.mkdir(parents=True)
        placements = []
        for index, (device, temperature) in enumerate(
            zip(devices, temperatures)
        ):
            directory = root / f"replica-{index}"
            directory.mkdir()
            settings = f'''md_name = "Alanine T-REMD {index}"
mode = "{"npt" if pressure is not None else "nvt"}"
step_limit = {steps}
dt = 0.001
cutoff = 12.0
default_in_file_prefix = {json.dumps(str(source / "sys_flexible"))}
thermostat = "middle_langevin"
thermostat_tau = 0.1
thermostat_seed = {2026 + index}
target_temperature = {temperature}
write_mdout_interval = 1000
write_information_interval = {steps}
'''
            if pressure is not None:
                settings += (
                    f'target_pressure = {pressure}\nbarostat = "bussi_barostat"\n'
                    f"barostat_update_interval = 10\nbarostat_seed = {3026 + index}\n"
                )
            (directory / "mdin.spg.toml").write_text(settings, encoding="utf-8")
            placements.append((device, directory))
        started = time.monotonic()
        if mode == "independent":
            with (root / "run.log").open("w") as log:
                workers = []
                try:
                    for device, directory in placements:
                        env = os.environ.copy()
                        if device != "-":
                            env["CUDA_VISIBLE_DEVICES"] = device
                        workers.append(
                            subprocess.Popen(
                                [engine, "-mdin", "mdin.spg.toml"],
                                cwd=directory,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        )
                    if any(worker.wait() != 0 for worker in workers):
                        raise RuntimeError(
                            f"independent run failed, see {log.name}"
                        )
                finally:
                    for worker in workers:
                        if worker.poll() is None:
                            worker.kill()
                        worker.wait()
        else:
            arguments = [item for pair in placements for item in map(str, pair)]
            with (root / "run.log").open("w") as log:
                subprocess.run(
                    [
                        manager,
                        engine,
                        str(interval),
                        "180000",
                        str(seed),
                        mode,
                        *arguments,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        elapsed = time.monotonic() - started
        result[mode] = {
            "wall_seconds": elapsed,
            "aggregate_steps_per_second": len(placements) * steps / elapsed,
        }
        if mode != "independent":
            attempts = []
            arrival_spans = []
            service_times = []
            for line in (root / "run.log").read_text().splitlines():
                if line.startswith("pair "):
                    attempts.append(int(line.split()[-1]))
                elif line.startswith("timing "):
                    _, _, arrival, service = line.split()
                    arrival_spans.append(float(arrival))
                    service_times.append(float(service))
            if not attempts or (mode == "sync" and any(attempts)):
                raise RuntimeError(f"unexpected exchange log in {root}")
            result[mode].update(attempts=len(attempts), accepted=sum(attempts))
            if arrival_spans:
                result[mode].update(
                    rounds=len(arrival_spans),
                    arrival_span_p50_ms=statistics.median(arrival_spans),
                    manager_service_p50_ms=statistics.median(service_times),
                )
    baseline = result["independent"]["aggregate_steps_per_second"]
    for mode in ("sync", "exchange"):
        result[mode]["fraction_of_independent_throughput"] = (
            result[mode]["aggregate_steps_per_second"] / baseline
        )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("manager", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--devices", nargs="+", default=["0", "1"])
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[300, 320]
    )
    parser.add_argument("--steps", type=int, default=10001)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pressure",
        type=float,
        help="use isotropic Bussi NPT at this pressure in bar",
    )
    args = parser.parse_args()
    if len(args.devices) != len(args.temperatures) or len(args.devices) < 2:
        parser.error(
            "provide matching device and temperature lists with at least two entries"
        )
    if not 0 < args.interval < args.steps:
        parser.error("require 0 < interval < steps")
    root = Path(__file__).resolve().parents[3]
    case = root / (
        "benchmarks/comparison/reference/gromacs/sponge_inputs/"
        "alanine_dipeptide_charmm_tip3p/iter0"
    )
    run(
        args.engine.resolve(),
        args.manager.resolve(),
        case,
        args.output.resolve(),
        args.steps,
        args.interval,
        args.devices,
        args.temperatures,
        args.seed,
        args.pressure,
    )
