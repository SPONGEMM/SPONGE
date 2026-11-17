"""Exercise the standalone manager with deterministic Linux replica processes."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

MANAGER = os.environ.get("SPONGE_MANAGER")
FAKE = """#!/usr/bin/env python3
import os
from pathlib import Path
import socket
import struct
import time

rank = int(Path.cwd().name)
connection = socket.socket(fileno=int(os.environ["SPONGE_EXCHANGE_FD"]))
interval = int(os.environ["SPONGE_EXCHANGE_INTERVAL"])
temperature = float(300 + 20 * rank)
behavior = Path("mdin.spg.toml").read_text()
if behavior == "fail" and rank == 1:
    raise SystemExit(3)
if behavior == "delay" and rank == 1:
    time.sleep(5)
connection.send(struct.pack("@IiiiidddddQ", 5, 0, 0, 31, 2, 0.001,
                            temperature, 0.0, 1.0, 1.0, 1))
assignment = struct.unpack("@iid", connection.recv(16))
for round_index in range(1, 4):
    step, peer, peer_temperature = assignment
    assert step == round_index * interval
    connection.send(struct.pack("@i", step))
    assert struct.unpack("@i", connection.recv(4))[0] == step
    cross = float(10000 * (round_index > 1)) if peer >= 0 else 0.0
    connection.send(struct.pack("@i4xdd", step, 0.0, cross))
    decision_step, checkpoint, transfer, *assignment = struct.unpack(
        "@iii4xiid", connection.recv(32))
    assert step == decision_step
"""


@unittest.skipUnless(MANAGER, "set SPONGE_MANAGER to the built executable")
class ManagerTest(unittest.TestCase):
    def run_replicas(self, count=2, mode="exchange", behavior="", timeout=5000):
        with tempfile.TemporaryDirectory(prefix="sponge-remd-test-") as path:
            root = Path(path)
            fake = root / "fake"
            fake.write_text(FAKE, encoding="utf-8")
            fake.chmod(0o700)
            placements = []
            for rank in range(count):
                replica = root / str(rank)
                replica.mkdir()
                (replica / "mdin.spg.toml").write_text(
                    behavior, encoding="utf-8"
                )
                placements.extend(["-", str(replica)])
            return subprocess.run(
                [
                    Path(MANAGER).resolve(),
                    fake,
                    "10",
                    str(timeout),
                    "42",
                    mode,
                    *placements,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

    def test_two_replicas_exchange_labels(self):
        run = self.run_replicas()
        self.assertEqual(run.returncode, 0, run.stderr)
        lines = run.stdout.splitlines()
        self.assertIn("label 10 1 0 300", lines)
        self.assertIn("label 10 0 1 320", lines)
        self.assertIn("label 20 1 0 300", lines)
        self.assertIn("label 20 0 1 320", lines)
        self.assertIn("label 30 1 0 300", lines)
        attempts = [line.split() for line in lines if line.startswith("pair ")]
        self.assertEqual([line[-1] for line in attempts], ["1", "0", "0"])

    def test_sync_preserves_labels_and_pairs_by_round(self):
        run = self.run_replicas(count=4, mode="sync")
        self.assertEqual(run.returncode, 0, run.stderr)
        lines = [line.split() for line in run.stdout.splitlines()]
        pairs = [
            tuple(map(int, line[1:4])) for line in lines if line[0] == "pair"
        ]
        self.assertEqual(
            pairs, [(10, 0, 1), (10, 2, 3), (20, 1, 2), (30, 0, 1), (30, 2, 3)]
        )
        self.assertTrue(
            all(line[2] == line[3] for line in lines if line[0] == "label")
        )

    def test_failed_worker_aborts_group(self):
        run = self.run_replicas(behavior="fail")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("replica 1 failed", run.stderr)

    def test_stalled_worker_times_out(self):
        run = self.run_replicas(behavior="delay", timeout=250)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("timed out", run.stderr)


if __name__ == "__main__":
    unittest.main()
