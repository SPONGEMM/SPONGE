"""Sample a CPU harmonic dimer and compare label-resolved energy to quadrature."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
import subprocess

TEMPERATURES = (300, 600)
STEPS = 2_000_001
INTERVAL = 100
OUTPUT_INTERVAL = 53
BURN_IN = 100_000
BLOCK_SIZE = 100_000
SPRING = 50
R0 = 1.5
KB = 0.00198716


def reference(temperature):
    numerator = denominator = 0.0
    # A radial 3D bond has the configurational measure r^2 dr.
    for i in range(40000):
        radius = (i + 0.5) * 0.0001
        energy = SPRING * (radius - R0) ** 2
        weight = radius**2 * math.exp(-energy / (KB * temperature))
        numerator += weight * energy
        denominator += weight
    return numerator / denominator


def reference_quantiles(temperature):
    samples = []
    for i in range(40000):
        radius = (i + 0.5) * 0.0001
        energy = SPRING * (radius - R0) ** 2
        samples.append(
            (energy, radius**2 * math.exp(-energy / (KB * temperature)))
        )
    total = sum(weight for _, weight in samples)
    limits = iter((0.25, 0.5, 0.75, 0.9))
    next_limit = next(limits)
    cumulative = 0.0
    for energy, weight in sorted(samples):
        cumulative += weight / total
        if cumulative >= next_limit:
            yield energy, next_limit
            next_limit = next(limits, None)
            if next_limit is None:
                return


def prepare(replica, temperature):
    replica.mkdir(parents=True)
    (replica / "toy_coordinate.txt").write_text(
        "2\n0 0 0\n1.5 0 0\n1000 1000 1000 90 90 90\n"
    )
    (replica / "toy_mass.txt").write_text("2\n12\n12\n")
    (replica / "toy_bond.txt").write_text(f"1\n0 1 {SPRING} {R0}\n")
    (replica / "mdin.spg.toml").write_text(f"""md_name = "harmonic dimer"
mode = "nvt"
pbc = 0
step_limit = {STEPS}
dt = 0.001
default_in_file_prefix = "toy"
thermostat = "middle_langevin"
thermostat_tau = 0.05
thermostat_seed = {temperature}
target_temperature = {temperature}
write_mdout_interval = {OUTPUT_INTERVAL}
write_information_interval = {STEPS}
""")


def verify(output, replicas):
    attempts = []
    for line in (output / "manager.log").read_text().splitlines():
        if line.startswith("pair "):
            attempts.append(int(line.split()[-1]))
    if len(attempts) != (STEPS - 1) // INTERVAL:
        raise RuntimeError("incomplete exchange log")
    if not 0 < sum(attempts) < len(attempts):
        raise RuntimeError("expected both accepted and rejected exchanges")
    print(f"Accepted {sum(attempts)} of {len(attempts)} attempts")

    blocks = defaultdict(lambda: defaultdict(list))
    for temperature, replica in zip(TEMPERATURES, replicas):
        with (replica / "mdout.txt").open() as stream:
            header = next(stream).split()
            energy_index = header.index("eff_pot")
            for line in stream:
                fields = line.split()
                if not fields:
                    continue
                step = int(fields[0])
                if step <= BURN_IN or step % INTERVAL == 0:
                    continue
                blocks[temperature][(step - 1) // BLOCK_SIZE].append(
                    float(fields[energy_index])
                )

    for temperature in TEMPERATURES:
        means = [
            statistics.mean(samples) for samples in blocks[temperature].values()
        ]
        observed = statistics.mean(means)
        stderr = statistics.stdev(means) / math.sqrt(len(means))
        expected = reference(temperature)
        print(
            f"{temperature} K: mean {observed:.5f}, block SE {stderr:.5f}, "
            f"reference {expected:.5f} kcal/mol"
        )
        if abs(observed - expected) > max(4 * stderr, 0.03 * expected):
            raise RuntimeError(
                f"bond energy distribution check failed at {temperature} K"
            )
        for cutoff, expected_cdf in reference_quantiles(temperature):
            probabilities = [
                sum(value <= cutoff for value in samples) / len(samples)
                for samples in blocks[temperature].values()
            ]
            observed_cdf = statistics.mean(probabilities)
            cdf_se = statistics.stdev(probabilities) / math.sqrt(
                len(probabilities)
            )
            print(
                f"  P(U <= {cutoff:.5f}) = {observed_cdf:.4f} "
                f"(reference {expected_cdf:.2f}, block SE {cdf_se:.4f})"
            )
            if abs(observed_cdf - expected_cdf) > max(4 * cdf_se, 0.03):
                raise RuntimeError(
                    f"bond energy probability check failed at {temperature} K"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("manager", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    replicas = [args.output / str(temp) for temp in TEMPERATURES]
    for replica, temperature in zip(replicas, TEMPERATURES):
        prepare(replica, temperature)
    with (args.output / "manager.log").open("w") as log:
        subprocess.run(
            [
                args.manager.resolve(),
                args.engine.resolve(),
                str(INTERVAL),
                "30000",
                "42",
                "exchange",
                "-",
                replicas[0].resolve(),
                "-",
                replicas[1].resolve(),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    verify(args.output, replicas)
