"""Deterministic two-layer finite-range model for SPONGE PRIPS parity tests."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from prips.local_mp import (
    CSRNeighborView,
    LocalMPCapabilities,
    LocalMPModelAdapter,
    OwnedPrediction,
)
from prips.local_mp.exchange import LayerwiseGhostExchange

POSITIONS = ((0.4, 1.0, 1.0), (4.8, 1.0, 1.0), (5.2, 1.0, 1.0), (9.6, 1.0, 1.0))
CELL = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
DEFAULT_ATOM_TYPES = (6, 8, 6, 8)
ATOM_TYPES = tuple(
    json.loads(
        os.environ.get(
            "SPONGE_LOCAL_MP_ATOM_TYPES", json.dumps(DEFAULT_ATOM_TYPES)
        )
    )
)
SEED_FORCE = 0.125  # Added by the Python hook before the model contribution.
CUTOFF = 1.2
SKIN = 0.4


class ToyAdapter(LocalMPModelAdapter):
    def __init__(self, rank=None, log_dir=None):
        self._capabilities = LocalMPCapabilities(
            cutoff=CUTOFF,
            skin=SKIN,
            interaction_depth=2,
            exchange_schedule=((0, 1), (1, 1)),
            length_to_angstrom=1.0,
            energy_to_kcal_mol=1.0,
            feature_dtype=torch.float32,
        )
        self.rank = rank
        self.log_dir = Path(log_dir) if log_dir else None
        self.build_count = 0
        self.eval_count = 0

    @property
    def capabilities(self):
        return self._capabilities

    def _record(self, event, **data):
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / f"rank{self.rank}.jsonl").open(
            "a", encoding="utf-8"
        ) as out:
            out.write(
                json.dumps({"event": event, "rank": self.rank, **data}) + "\n"
            )

    def prepare_model(self, device):
        # No weights: the arithmetic below is deliberately fixed and FP32.
        self._record("prepare", device=str(device))

    def validate_atom_types(self, global_atom_types):
        # The analytic formula accepts every integer type label.
        pass

    def build_resident_inputs(self, domain, atom_types, neighbor_view, device):
        self.build_count += 1
        positions = domain.positions.detach().clone()
        n_owned = domain.num_owned
        ids = domain.atom_local.detach().to("cpu").tolist()
        source = neighbor_view.source_local_index.long()
        target = torch.repeat_interleave(
            torch.arange(n_owned, device=device),
            neighbor_view.row_ptr[1:] - neighbor_view.row_ptr[:-1],
            output_size=source.numel(),
        )
        candidates = torch.stack((target, source), dim=1).to("cpu").tolist()
        self._record(
            "build",
            epoch=domain.domain_epoch,
            neighbor_epoch=neighbor_view.neighbor_epoch,
            count=self.build_count,
            owned=ids[:n_owned],
            ghost=ids[n_owned:],
            edges=candidates,
        )
        return dict(
            positions=positions,
            cell=domain.cell,
            pbc=domain.pbc,
            atom_types=atom_types,
            atom_local=domain.atom_local,
            num_owned=n_owned,
            target=target,
            source=source,
            image_shift=neighbor_view.image_shift,
            epoch=domain.domain_epoch,
            neighbor_epoch=neighbor_view.neighbor_epoch,
        )

    def update_positions(self, resident_inputs, positions):
        resident_inputs["positions"] = positions.detach().clone()

    def evaluate_owned(
        self, resident_inputs, exchange, need_energy, need_virial
    ):
        positions = (
            resident_inputs["positions"].detach().clone().requires_grad_(True)
        )
        target, source = resident_inputs["target"], resident_inputs["source"]
        n_owned = resident_inputs["num_owned"]
        cell = resident_inputs["cell"]
        strain = torch.zeros_like(cell, requires_grad=True)
        strained_positions = positions + positions @ strain
        strained_cell = cell + cell @ strain
        if source.numel():
            delta = (
                strained_positions[source]
                - strained_positions[target]
                + resident_inputs["image_shift"].to(positions.dtype)
                @ strained_cell
            )
            distance2 = (delta * delta).sum(dim=1)
            weight = torch.exp(-distance2) * (distance2 < CUTOFF**2)
        else:
            weight = positions.new_zeros((0,))

        base = (
            resident_inputs["atom_types"][:n_owned]
            .to(torch.float32)
            .unsqueeze(1)
            / 10.0
        )
        radial = torch.zeros_like(base).index_add(
            0, target, weight.unsqueeze(1)
        )
        h0 = base + 0.05 * radial + 0.0 * positions[:n_owned, 0:1]
        h0_all = exchange.forward_exchange(h0.contiguous(), layer_id=0)
        msg1 = weight.unsqueeze(1) * h0_all[source]
        h1 = h0 + torch.zeros_like(h0).index_add(0, target, msg1)
        h1_all = exchange.forward_exchange(h1.contiguous(), layer_id=1)
        msg2 = weight.unsqueeze(1) * h1_all[source]
        h2 = h1 + torch.zeros_like(h1).index_add(0, target, msg2)
        energy = 0.02 * h2.square().sum()
        # Every PP rank must traverse both adjoints even with no local edges.
        energy = energy + 0.0 * (
            h0_all.sum() + h1_all.sum() + positions.sum() + strain.sum()
        )
        position_gradient, strain_gradient = torch.autograd.grad(
            energy, (positions, strain)
        )
        forces = exchange.reverse_owned_forces(
            (-position_gradient).contiguous()
        )
        virial = -0.5 * (strain_gradient + strain_gradient.T)
        self.eval_count += 1
        self._record(
            "evaluate",
            epoch=resident_inputs["epoch"],
            neighbor_epoch=resident_inputs["neighbor_epoch"],
            count=self.eval_count,
            owned=resident_inputs["atom_local"][:n_owned].to("cpu").tolist(),
            ghost=resident_inputs["atom_local"][n_owned:].to("cpu").tolist(),
            positions=positions[:n_owned].detach().to("cpu").tolist(),
            energy=float(energy.detach().item()),
            forces=forces.detach().to("cpu").tolist(),
            virial=virial.detach().to("cpu").tolist(),
        )
        return OwnedPrediction(
            energy.detach() if need_energy else None,
            forces.detach(),
            virial.detach() if need_virial else None,
        )


def full_system_reference(
    positions=POSITIONS, cell=CELL, atom_types=ATOM_TYPES
):
    positions = torch.tensor(positions, dtype=torch.float32)
    cell = torch.tensor(cell, dtype=torch.float32)
    atom_types = tuple(atom_types)
    domain = SimpleNamespace(
        num_owned=len(atom_types),
        num_ghost=0,
        atom_local=torch.arange(len(atom_types), dtype=torch.int32),
        positions=positions,
        cell=cell,
        pbc=(True, True, True),
        domain_epoch=1,
    )
    model = ToyAdapter()
    # This standalone test oracle enumerates geometry independently of SPONGE.
    # Runtime ToyAdapter.build_resident_inputs never calls it.
    from neighbor_reference import enumerate_periodic_edges

    edges = enumerate_periodic_edges(
        positions.tolist(),
        list(range(len(atom_types))),
        len(atom_types),
        cell.tolist(),
        domain.pbc,
        CUTOFF + SKIN,
    )
    counts = [0] * len(atom_types)
    for target, *_ in edges:
        counts[target] += 1
    rows = [0]
    for count in counts:
        rows.append(rows[-1] + count)
    neighbors = CSRNeighborView(
        1,
        torch.tensor(rows, dtype=torch.int32),
        torch.tensor([edge[1] for edge in edges], dtype=torch.int32),
        torch.tensor([edge[2:] for edge in edges], dtype=torch.int32).reshape(
            -1, 3
        ),
    )
    resident = model.build_resident_inputs(
        domain,
        torch.tensor(atom_types, dtype=torch.int32),
        neighbors,
        torch.device("cpu"),
    )
    exchange = LayerwiseGhostExchange(
        len(atom_types),
        0,
        torch.device("cpu"),
        lambda *args: 0,
        lambda *args: 0,
        model.capabilities.exchange_schedule,
    )
    return model.evaluate_owned(resident, exchange, True, True)


try:
    from prips import Sponge
except ImportError:
    Sponge = None

if Sponge is not None:
    rank = Sponge.controller.MPI_rank
    local_mp_adapter = ToyAdapter(
        rank, os.environ.get("SPONGE_LOCAL_MP_LOG_DIR")
    )
    atom_types = ATOM_TYPES

    def Calculate_Force():
        Sponge.dd.frc[: Sponge.dd.atom_numbers, 0].add_(SEED_FORCE)
