"""End-to-end CPU reference cases for replica exchange and restart."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ENGINE = os.environ.get("SPONGE_ENGINE")
MANAGER = os.environ.get("SPONGE_MANAGER")
ROOT = Path(__file__).resolve().parents[2]
H5_CASE = (
    ROOT
    / "tests/h5_bundle/fixtures/input_matrix/core_structural/bundled_input/bundle"
)


@unittest.skipUnless(ENGINE and MANAGER, "set SPONGE_ENGINE and SPONGE_MANAGER")
class EngineTest(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory(
            prefix="sponge-exchange-engine-"
        )
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.engine = Path(ENGINE).resolve()
        self.manager = Path(MANAGER).resolve()

    def managed(self, name, directories, flags=(), *, interval=100, code=0):
        arguments = [
            value for directory in directories for value in ("-", directory)
        ]
        result = subprocess.run(
            [
                self.manager,
                *flags,
                self.engine,
                str(interval),
                "30000",
                "42",
                "exchange",
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=40,
        )
        (self.root / f"{name}.log").write_text(result.stdout + result.stderr)
        logs = [
            (directory / "sponge.log").read_text()[-2500:]
            for directory in directories
            if (directory / "sponge.log").exists()
        ]
        self.assertEqual(
            result.returncode, code, result.stderr + "\n".join(logs)
        )
        return result

    def bundle(self, name, steps, *, barostat=None):
        directories = []
        for temperature in (300, 320):
            directory = self.root / name / str(temperature)
            directory.mkdir(parents=True)
            (
                directory / "mdin.spg.toml"
            ).write_text(f'''md_name = "Managed H5 continuation"
mode = "nvt"
dt = 0.0001
skin = 0.5
cutoff = 2.0
step_limit = {steps}
write_mdout_interval = 53
thermostat = "middle_langevin"
thermostat_seed = {temperature}
target_temperature = {temperature}
input_h5_topology_path = "{H5_CASE}/topology.spgt.h5"
input_h5_protocol_path = "{H5_CASE}/protocol.spgp.h5"
input_h5_restart_path = "{H5_CASE}/restart.spgr.h5"
input_h5_restart_load = "structural"
output_h5_restart_path = "regular.spgr.h5"
''')
            if barostat:
                config = directory / "mdin.spg.toml"
                prefix = (
                    "monte_carlo_barostat"
                    if barostat == "monte_carlo_barostat"
                    else "barostat"
                )
                config.write_text(
                    config.read_text().replace('mode = "nvt"', 'mode = "npt"')
                    + f'target_pressure = 1\nbarostat = "{barostat}"\n'
                    + f"{prefix}_update_interval = 10\n{prefix}_seed = 70\n"
                )
            directories.append(directory)
        return directories

    def dimer(
        self, name, state, distance, *, steps=12, mass=12, temperature=300
    ):
        directory = self.root / name
        directory.mkdir()
        charge, spring, lam = state
        length = 40 if lam is not None else 1000
        (directory / "toy_coordinate.txt").write_text(
            f"2\n10 10 10\n{10 + distance} 10 10\n{length} {length} {length} 90 90 90\n"
        )
        (directory / "toy_mass.txt").write_text(f"2\n12\n{mass}\n")
        (directory / "toy_charge.txt").write_text(f"2\n{charge}\n{-charge}\n")
        (directory / "toy_bond.txt").write_text(f"1\n0 1 {spring} 1.5\n")
        if lam is not None:
            (directory / "toy_LJ_soft_core.txt").write_text(
                "2 1 1\n500\n20\n2500\n40\n0 0\n0 0\n"
            )
        text = f"""md_name = "Exchange dimer"
mode = "nvt"
pbc = {int(lam is not None)}
step_limit = {steps}
dt = 0.000000001
cutoff = 8.0
default_in_file_prefix = "toy"
thermostat = "middle_langevin"
thermostat_tau = 0.05
thermostat_seed = 300
target_temperature = {temperature}
write_mdout_interval = 1
"""
        if lam is not None:
            text += f"lambda_lj = {lam}\n"
        (directory / "mdin.spg.toml").write_text(text)
        return directory

    @staticmethod
    def attempts(result):
        return [
            line.split()
            for line in result.stdout.splitlines()
            if line.startswith("pair ")
        ]

    def single_energy(self, directory, observable="eff_pot"):
        run = subprocess.run(
            [self.engine, "-mdin", "mdin.spg.toml"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr)
        lines = (directory / "mdout.txt").read_text().splitlines()
        column = lines[0].split().index(observable)
        return float(lines[1].split()[column])

    def compare_cross_terms(self, states, positions, temperatures=(300, 300)):
        energies = [
            [
                self.single_energy(
                    self.dimer(
                        f"single-{i}-{j}",
                        state,
                        distance,
                        temperature=temperatures[i],
                    )
                )
                for j, distance in enumerate(positions)
            ]
            for i, state in enumerate(states)
        ]
        expected = (energies[0][0] - energies[0][1]) / (
            0.00198716 * temperatures[0]
        ) + (energies[1][1] - energies[1][0]) / (0.00198716 * temperatures[1])
        directories = [
            self.dimer(
                f"replica-{i}", state, distance, temperature=temperatures[i]
            )
            for i, (state, distance) in enumerate(zip(states, positions))
        ]
        run = self.managed("exchange", directories, interval=1)
        attempt = self.attempts(run)[0]
        self.assertGreater(abs(expected), 0.01)
        self.assertAlmostEqual(float(attempt[4]), expected, delta=0.001)

    def test_fixed_parameters_include_charge_and_bond_terms(self):
        self.compare_cross_terms([(3, 50, None), (5, 75, None)], [1.3, 1.8])

    def test_fixed_periodic_windows_include_pme_charge_and_bond_terms(self):
        self.compare_cross_terms([(3, 50, 0.1), (5, 75, 0.8)], [1.6, 2.0])

    def test_fixed_swap_changes_configurations_but_keeps_window_parameters(
        self,
    ):
        directories = [
            self.dimer("first", (3, 50, None), 1.3, steps=2),
            self.dimer("second", (5, 75, None), 1.8, steps=2),
        ]
        results = self.managed("physical-swap", directories, interval=1)
        self.assertEqual([line[-1] for line in self.attempts(results)], ["1"])
        for directory, expected, charge in zip(directories, (1.8, 1.3), (3, 5)):
            positions = (
                (directory / "restart_coordinate.txt").read_text().splitlines()
            )
            distance = float(positions[2].split()[0]) - float(
                positions[1].split()[0]
            )
            self.assertAlmostEqual(distance, expected, delta=0.0001)
            self.assertEqual(
                (directory / "toy_charge.txt").read_text().split()[1],
                str(charge),
            )

    def test_temperature_exchange_rescales_incoming_velocities(self):
        directories = []
        for index, (temp, speed) in enumerate(((300, 0.3), (600, 0.8))):
            directory = self.dimer(
                f"temperature-{index}",
                (0, 0, None),
                1.5,
                steps=2,
                temperature=temp,
            )
            (directory / "toy_velocity.txt").write_text(
                f"2\n{speed} 0 0\n{-speed} 0 0\n"
            )
            config = directory / "mdin.spg.toml"
            config.write_text(
                config.read_text().replace(
                    "thermostat_tau = 0.05", "thermostat_tau = 1000000"
                )
            )
            directories.append(directory)
        run = self.managed("velocities", directories, interval=1)
        self.assertEqual([line[-1] for line in self.attempts(run)], ["1"])
        for directory, expected in zip(
            directories, (0.8 / 2**0.5, 0.3 * 2**0.5)
        ):
            rows = (directory / "restart_velocity.txt").read_text().splitlines()
            self.assertAlmostEqual(
                float(rows[1].split()[0]), expected, delta=0.0001
            )
            self.assertAlmostEqual(
                float(rows[2].split()[0]), -expected, delta=0.0001
            )

    def test_fixed_windows_reject_different_masses(self):
        directories = [
            self.dimer("first", (3, 50, None), 1.3),
            self.dimer("second", (5, 75, None), 1.8, mass=13),
        ]
        run = self.managed("bad-mass", directories, interval=1, code=1)
        self.assertIn("incompatible replica handshake", run.stderr)

    def test_lambda_cross_terms(self):
        self.compare_cross_terms([(0, 0, 0.1), (0, 0, 0.8)], [1.6, 2.0])

    def test_combined_temperature_lambda_cross_terms(self):
        self.compare_cross_terms(
            [(0, 0, 0.1), (0, 0, 0.8)], [1.6, 2.0], (300, 400)
        )

    def test_fixed_sits_strength_cross_terms(self):
        def sits_dimer(name, distance, strength=1):
            directory = self.dimer(name, (0, 50, None), distance)
            (directory / "weights.txt").write_text("1 2\n")
            with (directory / "mdin.spg.toml").open("a") as stream:
                stream.write(
                    'SITS_mode = "production"\nSITS_k_numbers = 2\n'
                    f'SITS_atom_numbers = "ITS"\nSITS_bias_strength = {strength}\n'
                    'SITS_T_low = 280\nSITS_T_high = 420\nSITS_nk_in_file = "weights.txt"\n'
                )
            return directory

        positions = (1.3, 1.8)
        biases = [
            self.single_energy(
                sits_dimer(f"sits-reference-{i}", position), "SITS_bias"
            )
            for i, position in enumerate(positions)
        ]
        expected = (biases[1] - biases[0]) / (0.00198716 * 300)
        directories = [
            sits_dimer(f"sits-{i}", position, strength=i)
            for i, position in enumerate(positions)
        ]
        run = self.managed("sits", directories, interval=1)
        self.assertGreater(abs(expected), 0.01)
        self.assertAlmostEqual(
            float(self.attempts(run)[0][4]), expected, delta=0.02
        )

    def npt_case(
        self, name, box, pressure, barostat, steps, interval, *, moving=False
    ):
        directory = self.dimer(name, (0, 50, 0.5), 1.5, steps=steps)
        coordinate = directory / "toy_coordinate.txt"
        lines = coordinate.read_text().splitlines()
        lines[-1] = f"{box} {box} {box} 90 90 90"
        coordinate.write_text("\n".join(lines) + "\n")
        (directory / "toy_LJ_soft_core.txt").write_text(
            "2 1 1\n0\n0\n0\n0\n0 0\n0 0\n"
        )
        config = directory / "mdin.spg.toml"
        text = config.read_text().replace('mode = "nvt"', 'mode = "npt"')
        text += f'target_pressure = {pressure}\nbarostat = "{barostat}"\n'
        if moving:
            text = text.replace("dt = 0.000000001", "dt = 0.0005")
            text += "write_trajectory_interval = 1\n"
        if barostat == "monte_carlo_barostat":
            text += (
                f"monte_carlo_barostat_update_interval = {interval}\n"
                "monte_carlo_barostat_seed = 902\n"
            )
            if not moving:
                text += "monte_carlo_barostat_initial_ratio = 0\n"
        else:
            text += (
                f"barostat_update_interval = {interval}\nbarostat_seed = 902\n"
            )
        config.write_text(text)
        return directory

    def test_npt_exchange_includes_pressure_and_moves_box(self):
        for barostat in ("monte_carlo_barostat", "bussi_barostat"):
            with self.subTest(barostat=barostat):
                boxes = (46, 40)
                pressures = (500, 1)
                directories = [
                    self.npt_case(
                        f"{barostat}-{i}", box, pressure, barostat, 2, 100
                    )
                    for i, (box, pressure) in enumerate(zip(boxes, pressures))
                ]
                run = self.managed(barostat, directories, interval=1)
                [attempt] = self.attempts(run)
                expected = (
                    (pressures[0] - pressures[1])
                    * (boxes[0] ** 3 - boxes[1] ** 3)
                    * 0.00001439506089041446
                    / (0.00198716 * 300)
                )
                self.assertAlmostEqual(float(attempt[4]), expected, delta=0.001)
                self.assertEqual(attempt[-1], "1")
                for directory, expected_box in zip(
                    directories, reversed(boxes)
                ):
                    box = (
                        (directory / "restart_coordinate.txt")
                        .read_text()
                        .splitlines()[-1]
                    )
                    self.assertAlmostEqual(
                        float(box.split()[0]), expected_box, delta=0.0001
                    )

    def test_npt_rejection_restores_original_box(self):
        boxes = (46, 40)
        directories = [
            self.npt_case(
                f"reject-{i}", box, pressure, "bussi_barostat", 2, 100
            )
            for i, (box, pressure) in enumerate(zip(boxes, (1, 500)))
        ]
        run = self.managed("npt-reject", directories, interval=1)
        [attempt] = self.attempts(run)
        self.assertEqual(attempt[-1], "0")
        for directory, expected_box in zip(directories, boxes):
            box = (
                (directory / "restart_coordinate.txt")
                .read_text()
                .splitlines()[-1]
            )
            self.assertAlmostEqual(
                float(box.split()[0]), expected_box, delta=0.0001
            )

    def test_npt_barostat_keeps_changing_volume_after_exchange(self):
        for barostat in ("monte_carlo_barostat", "bussi_barostat"):
            with self.subTest(barostat=barostat):
                directories = [
                    self.npt_case(
                        f"moving-{barostat}-{i}",
                        box,
                        1,
                        barostat,
                        11,
                        2,
                        moving=True,
                    )
                    for i, box in enumerate((42, 40))
                ]
                run = self.managed(
                    f"moving-{barostat}", directories, interval=6
                )
                [attempt] = self.attempts(run)
                self.assertEqual(attempt[-1], "1")
                for directory in directories:
                    rows = (directory / "mdbox.txt").read_text().splitlines()
                    lengths = [
                        float(row.split()[0]) for row in rows if row.strip()
                    ]
                    self.assertGreater(len(lengths), 8, rows)
                    self.assertNotEqual(lengths[6], lengths[-1], rows)

    def test_h5_group_resume_matches_uninterrupted_exchange(self):
        self.compare_group_resume(barostat=None)

    def test_npt_bussi_h5_group_resume_matches_uninterrupted_exchange(self):
        self.compare_group_resume(barostat="bussi_barostat")

    def test_npt_monte_carlo_h5_group_resume_matches_uninterrupted_exchange(
        self,
    ):
        self.compare_group_resume(barostat="monte_carlo_barostat")

    def compare_group_resume(self, barostat):
        manifest = self.root / "group.state"
        continuous = self.managed(
            "continuous", self.bundle("continuous", 401, barostat=barostat)
        )
        split = self.managed(
            "split",
            self.bundle("split", 201, barostat=barostat),
            ["--checkpoint", manifest, "2"],
        )
        self.assertIn("checkpoint 200", split.stdout)
        self.assertTrue(manifest.is_file())
        continuation = self.managed(
            "resume",
            self.bundle("resume", 200, barostat=barostat),
            ["--resume", manifest, "2"],
        )
        reference = self.attempts(continuous)
        resumed = self.attempts(split) + self.attempts(continuation)
        self.assertEqual(len(reference), 4)
        self.assertEqual(len(resumed), len(reference))
        for left, right in zip(reference, resumed):
            self.assertEqual(left[:4] + left[5:], right[:4] + right[5:])
            self.assertAlmostEqual(
                float(left[4]), float(right[4]), delta=0.0001
            )


if __name__ == "__main__":
    unittest.main()
