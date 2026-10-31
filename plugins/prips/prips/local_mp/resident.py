"""Resident model lifecycle for the current SPONGE PP domain."""

from dataclasses import dataclass

import torch

from .contracts import LocalMPModelAdapter, OwnedPrediction, map_atom_types
from .exchange import LayerwiseGhostExchange, _CompiledExchangeLayout
from .neighbors import RuntimeNeighborCache


@dataclass
class DomainView:
    num_owned: int
    num_ghost: int
    atom_local: torch.Tensor
    positions: torch.Tensor
    cell: torch.Tensor
    pbc: tuple[bool, bool, bool]
    domain_epoch: int


class ResidentLocalMPDriver:
    """Binds one model adapter to one SPONGE PP rank and its current domain."""

    def __init__(self, sponge, adapter: LocalMPModelAdapter, atom_types):
        if not isinstance(adapter, LocalMPModelAdapter):
            raise TypeError(
                "local_mp_adapter must implement LocalMPModelAdapter"
            )
        self.sponge = sponge
        self.adapter = adapter
        declared_types = torch.as_tensor(atom_types)
        if declared_types.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("atom_types must contain integer entries")
        self.global_types = declared_types.to(
            device="cpu", dtype=torch.int64
        ).clone()
        if (
            self.global_types.ndim != 1
            or self.global_types.numel() != sponge.md_info.atom_numbers
        ):
            raise ValueError(
                "atom_types must have one integer entry per global atom"
            )
        device_index = sponge._device_index()
        self.device = (
            torch.device("cuda", device_index)
            if sponge._backend == 2 and device_index >= 0
            else None
        )
        self._prepared = False
        self._resident = None
        self._neighbors = RuntimeNeighborCache(sponge)
        self._epoch = None
        cap = adapter.capabilities
        self._registered_capabilities = cap
        self._exchange_layout = _CompiledExchangeLayout.from_capabilities(cap)
        status = sponge._register_mlip_capabilities(
            cap.cutoff_angstrom,
            cap.skin_angstrom,
            cap.interaction_depth,
            cap.capability_hash,
        )
        if status != 0:
            raise RuntimeError("SPONGE rejected local-MP capabilities")

    def prepare_setup(self) -> None:
        """Prepare locally before native PP setup-success consensus.

        No graph or communication is created here. Exceptions leave force
        admission closed on all PP ranks.
        """
        if self._prepared:
            return
        if self.device is None:
            raise RuntimeError(
                "local-MP setup requires a selected CUDA+MPI PP device"
            )
        self.adapter.prepare_model(self.device)
        self.adapter.validate_atom_types(self.global_types)
        if self.adapter.capabilities != self._registered_capabilities:
            raise ValueError(
                "model setup changed registered communication capabilities"
            )
        self._prepared = True

    def set_domain_information(self, domain) -> None:
        # Setup prepares weights once; domain changes invalidate only graph state.
        if self._epoch != domain.domain_epoch:
            self._resident = None

    def _discard_resident(self) -> None:
        # Release adapter-held Python borrows before acquiring replacement CSR.
        # Runtime device fencing remains responsible for in-flight GPU work.
        self._resident = None

    def _domain_view(self, domain) -> DomainView:
        cap = self.adapter.capabilities
        return DomainView(
            domain.atom_numbers,
            domain.ghost_numbers,
            domain.atom_local,
            # Out-of-place division already creates independent storage, even
            # at unit scale; retain it to isolate native buffers from autograd.
            cap.to_model_lengths(domain.crd.detach()),
            cap.to_model_lengths(
                torch.as_tensor(
                    domain.cell, dtype=torch.float32, device=self.device
                )
            ),
            domain.pbc,
            domain.domain_epoch,
        )

    def calculate_force(self) -> None:
        sponge = self.sponge
        if sponge.dd is None or sponge.dd.pp_rank is None:
            return
        if self.device is None:
            raise RuntimeError(
                "local-MP inference requires a selected CUDA+MPI PP device"
            )
        if not self._prepared:
            raise RuntimeError(
                "local-MP force requires native prepare_setup consensus"
            )
        runtime_domain = sponge.dd
        num_owned = runtime_domain.atom_numbers
        num_ghost = runtime_domain.ghost_numbers
        domain_epoch = runtime_domain.domain_epoch
        cap = self.adapter.capabilities
        try:
            neighbors = self._neighbors.get(
                num_owned,
                num_ghost,
                self.device,
                before_acquire=self._discard_resident,
            )
        except Exception:
            self._discard_resident()
            self._neighbors.invalidate()
            raise
        rebuild = self._resident is None or self._epoch != domain_epoch
        if rebuild:
            domain = self._domain_view(runtime_domain)
            local_types = map_atom_types(
                self.global_types,
                domain.atom_local,
                sponge.md_info.atom_numbers,
            )
            self._resident = self.adapter.build_resident_inputs(
                domain, local_types, neighbors, self.device
            )
            self._epoch = domain_epoch
        else:
            self.adapter.update_positions_from_sponge(
                self._resident, runtime_domain.crd
            )

        exchange = LayerwiseGhostExchange(
            num_owned,
            num_ghost,
            self.device,
            sponge._update_ghost_tensor,
            sponge._accumulate_ghost_tensor,
            cap.exchange_schedule,
            _layout=self._exchange_layout,
            edge_output_callback=getattr(
                sponge, "_assemble_edge_gradients", None
            ),
        )
        need_energy = bool(sponge._need_potential())
        need_virial = bool(sponge._need_pressure())
        prediction = self.adapter.evaluate_owned(
            self._resident, exchange, need_energy, need_virial
        )
        exchange.validate_complete()
        if not isinstance(prediction, OwnedPrediction):
            raise TypeError("adapter must return OwnedPrediction")
        prediction.validate(num_owned, need_energy, need_virial, self.device)
        converted = prediction.to_sponge_units(cap)
        submit = getattr(sponge, "_submit_owned_outputs", None)
        if submit is None:
            raise RuntimeError("native owned output submission is unavailable")
        # The scalar's one-element view lets the bridge borrow it using the
        # same rank-checked DLPack path as other tensors, without copying.
        energy = (
            None
            if converted.energy is None
            else converted.energy.detach().reshape(1)
        )
        status = submit(
            converted.forces.detach().contiguous(),
            energy,
            None
            if converted.virial is None
            else converted.virial.detach().contiguous(),
        )
        if status != 0:
            raise RuntimeError("SPONGE rejected owned output submission")
