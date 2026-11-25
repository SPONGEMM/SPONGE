"""Real native-service parity with an exhaustive analytic periodic oracle."""

import argparse
from pathlib import Path

import torch
from neighbor_reference import enumerate_periodic_edges
from test_local_mp_mpi import _close, _read_force, _run

CELL = ((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0))
POSITIONS = (
    (0.4, 1.0, 1.0),
    (5.8, 1.0, 1.0),
    (6.2, 1.0, 1.0),
    (11.6, 1.0, 1.0),
)
LENGTH_SCALE = 1.7
ENERGY_SCALE = 2.3


def exhaustive_reference(positions_angstrom, dtype):
    """Enumerate full physical graph; accumulate endpoints one edge at a time."""
    native_positions = torch.tensor(positions_angstrom, dtype=torch.float32)
    native_cell = torch.tensor(CELL, dtype=torch.float32)
    positions = native_positions.to(dtype) / LENGTH_SCALE
    cell = native_cell.to(dtype) / LENGTH_SCALE
    forces = torch.zeros_like(positions)
    energy = torch.zeros((), dtype=dtype)
    virial = torch.zeros((3, 3), dtype=dtype)
    edges = enumerate_periodic_edges(
        native_positions.tolist(),
        range(len(positions)),
        len(positions),
        CELL,
        (True, True, True),
        1.2 * LENGTH_SCALE,
    )
    for target, source, *shift in edges:
        vector = (
            positions[target]
            - positions[source]
            - torch.tensor(shift, dtype=dtype) @ cell
        )
        if float(vector.square().sum()) >= 1.2**2:
            continue
        gradient = 0.5 * vector
        energy += 0.25 * vector.square().sum()
        forces[source] += gradient
        forces[target] -= gradient
        moment = torch.outer(vector, gradient)
        virial -= 0.5 * (moment + moment.T)
    return energy, forces, virial


def run_case(args):
    args.py = Path(__file__).with_name("fixture_edge_output.py").resolve()
    ranks = 2
    positions = POSITIONS
    if args.case == "empty_rank":
        positions = ((1.0, 1.0, 1.0), (1.7, 1.0, 1.0))
    types = (6,) * len(positions)
    directory, events = _run(
        args,
        "analytic",
        ranks=ranks,
        positions=positions,
        cell=CELL,
        atom_types=types,
        steps=1,
        dt=0.0,
    )
    evaluations = [event for event in events if event["event"] == "evaluate"]
    assert evaluations, f"missing native fixture outputs: {directory}"
    for count in sorted({event["count"] for event in evaluations}):
        records = [event for event in evaluations if event["count"] == count]
        assert len(records) == ranks
        actual_positions, forces = (
            [None] * len(positions),
            [None] * len(positions),
        )
        energy, virial = 0.0, torch.zeros((3, 3), dtype=torch.float64)
        for record in records:
            assert record["native_edge_assembly"] == 1
            assert record["torch_cuda_local_virial_parity"] is True
            assert record["producer_stream"] != record["default_stream"]
            energy += record["energy"]
            virial += torch.tensor(record["virial"], dtype=torch.float64)
            for atom, force, point, pre_force in zip(
                record["owned"],
                record["forces"],
                record["positions_angstrom"],
                record["pre_model_forces_native"],
            ):
                assert forces[atom] is None, "duplicate owner output"
                forces[atom], actual_positions[atom] = force, point
                _close(pre_force, [0.125, 0.0, 0.0])
        assert all(point is not None for point in actual_positions)
        for dtype in (torch.float32, torch.float64):
            expected_energy, expected_forces, expected_virial = (
                exhaustive_reference(actual_positions, dtype)
            )
            _close(energy, expected_energy)
            _close(forces, expected_forces)
            _close(virial, expected_virial)
        if args.case == "empty_rank":
            assert any(not record["owned"] for record in records)
    last = max(event["count"] for event in evaluations)
    final = [event for event in evaluations if event["count"] == last]
    native_forces = [None] * len(positions)
    for record in final:
        for atom, force in zip(record["owned"], record["forces"]):
            native_forces[atom] = [
                value * ENERGY_SCALE / LENGTH_SCALE for value in force
            ]
            native_forces[atom][0] += 0.125
    baseline, _ = _run(
        args,
        "baseline",
        ranks=ranks,
        plugin=False,
        positions=positions,
        cell=CELL,
        atom_types=types,
        steps=1,
        dt=0.0,
    )
    observed, control = (
        _read_force(directory, len(positions)),
        _read_force(baseline, len(positions)),
    )
    difference = [
        [a - b for a, b in zip(row, other)]
        for row, other in zip(observed, control)
    ]
    _close(difference, native_forces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "two_pp",
            "empty_rank",
        ),
        required=True,
    )
    parser.add_argument("--sponge", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    run_case(parser.parse_args())


if __name__ == "__main__":
    main()
