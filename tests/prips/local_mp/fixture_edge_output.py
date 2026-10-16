"""Model-independent analytic fixture for the real native edge-output service."""

import json
import os
from pathlib import Path

import torch
from prips import Sponge
from prips.local_mp.contracts import (
    LocalMPCapabilities,
    LocalMPModelAdapter,
    OwnedPrediction,
)


class AnalyticEdgeAdapter(LocalMPModelAdapter):
    capabilities = LocalMPCapabilities(
        cutoff=1.2,
        skin=0.4,
        interaction_depth=1,
        exchange_schedule=(),
        length_to_angstrom=1.7,
        energy_to_kcal_mol=2.3,
    )

    def __init__(self):
        self.count = 0
        self.builds = 0
        self.log_dir = Path(os.environ["SPONGE_LOCAL_MP_LOG_DIR"])

    def prepare_model(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def validate_atom_types(self, global_atom_types):
        # The analytic formula accepts every integer type label.
        pass

    def build_resident_inputs(self, domain, atom_types, neighbor_view, device):
        self.builds += 1
        target = torch.repeat_interleave(
            torch.arange(domain.num_owned, device=device),
            neighbor_view.row_ptr[1:] - neighbor_view.row_ptr[:-1],
            output_size=neighbor_view.source_local_index.numel(),
        )
        return dict(
            positions=domain.positions.detach().clone(),
            cell=domain.cell,
            source=neighbor_view.source_local_index.long(),
            target=target,
            image_shift=neighbor_view.image_shift,
            num_owned=domain.num_owned,
            ids=domain.atom_local,
            domain_epoch=domain.domain_epoch,
            neighbor_epoch=neighbor_view.neighbor_epoch,
        )

    def update_positions(self, resident_inputs, positions):
        resident_inputs["positions"].copy_(positions.detach())

    def evaluate_owned(self, resident, exchange, need_energy, need_virial):
        self.count += 1
        pre_force = (
            Sponge.dd.frc[: resident["num_owned"]].detach().cpu().tolist()
        )
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            vectors = (
                resident["positions"][resident["target"]]
                - resident["positions"][resident["source"]]
                - resident["image_shift"].to(torch.float32) @ resident["cell"]
            )
            selected = torch.nonzero(
                vectors.square().sum(dim=1) < self.capabilities.cutoff**2,
                as_tuple=True,
            )[0].contiguous()
            active_vectors = torch.index_select(vectors, 0, selected)
            gradients = (0.5 * active_vectors).contiguous()
            energy = 0.25 * active_vectors.square().sum()
            forces, virial = exchange.assemble_edge_gradients(
                selected,
                gradients,
                resident["neighbor_epoch"],
                self.capabilities.length_to_angstrom,
                need_virial,
            )
            # Independent test reference uses the actual Torch CUDA vectors.
            # It runs only after native completion and never assembles forces.
            if virial is not None:
                cuda_moment = torch.zeros(
                    (3, 3), dtype=torch.float32, device=self.device
                )
                for edge in range(active_vectors.shape[0]):
                    cuda_moment += torch.outer(
                        active_vectors[edge], gradients[edge]
                    )
                cuda_virial = -0.5 * (cuda_moment + cuda_moment.T)
                torch.testing.assert_close(
                    virial, cuda_virial, rtol=1e-5, atol=1e-5
                )
            # Read immediately after the native callback returns. The callback's
            # host-completion contract covers every producer and output write.
            record = dict(
                event="evaluate",
                rank=Sponge.controller.MPI_rank,
                count=self.count,
                owned=resident["ids"][: resident["num_owned"]].cpu().tolist(),
                ghost=resident["ids"][resident["num_owned"] :].cpu().tolist(),
                positions_angstrom=Sponge.dd.crd[: resident["num_owned"]]
                .cpu()
                .tolist(),
                energy=float(energy.cpu()),
                forces=forces.cpu().tolist(),
                virial=None if virial is None else virial.cpu().tolist(),
                pre_model_forces_native=pre_force,
                selected=selected.cpu().tolist(),
                candidate_count=int(vectors.shape[0]),
                native_edge_assembly=1,
                torch_cuda_local_virial_parity=virial is not None,
                producer_stream=int(self.stream.cuda_stream),
                default_stream=int(
                    torch.cuda.default_stream(self.device).cuda_stream
                ),
                domain_epoch=resident["domain_epoch"],
                neighbor_epoch=resident["neighbor_epoch"],
                builds=self.builds,
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / f"rank{record['rank']}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return OwnedPrediction(
            energy.detach() if need_energy else None,
            forces.detach(),
            None if virial is None else virial.detach(),
        )


atom_types = json.loads(os.environ["SPONGE_LOCAL_MP_ATOM_TYPES"])
local_mp_adapter = AnalyticEdgeAdapter()
uses_classic_neighbor_list = False


def Calculate_Force():
    Sponge.dd.frc[: Sponge.dd.atom_numbers, 0].add_(0.125)
