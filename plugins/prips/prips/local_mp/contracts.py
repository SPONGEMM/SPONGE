"""Explicit model and output contracts for the PRIPS local-MP boundary."""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FieldSpec:
    """Semantic columns; static content still follows the current local layout."""

    name: str
    width: int
    scope: str
    lifetime: str
    differentiable: bool

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("field name must be a nonempty string")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("field width must be a positive integer")
        if self.scope not in ("owned", "local", "global"):
            raise ValueError("field scope must be owned, local or global")
        if self.lifetime not in ("static", "domain", "step", "iteration"):
            raise ValueError(
                "field lifetime must be static, domain, step or iteration"
            )
        if type(self.differentiable) is not bool:
            raise TypeError("field differentiable must be a boolean")


@dataclass(frozen=True)
class ExchangeStage:
    stage_id: int
    field_names: tuple[str, ...]

    def __post_init__(self):
        if type(self.stage_id) is not int or self.stage_id < 0:
            raise ValueError("stage ID must be a nonnegative integer")
        names = tuple(self.field_names)
        if not names or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("stage field names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("stage field names must be unique")
        object.__setattr__(self, "field_names", names)


@dataclass(frozen=True)
class LocalMPCapabilities:
    cutoff: float
    skin: float
    interaction_depth: int
    # Explicitly list only layers that exchange hidden features. Many local-MP
    # models seed the first layer from local atom attributes and exchange only
    # between later interaction layers (depth L therefore has L-1 points).
    exchange_schedule: tuple[tuple[int, int], ...]
    length_to_angstrom: float
    energy_to_kcal_mol: float
    feature_dtype: torch.dtype = torch.float32
    fields: tuple[FieldSpec, ...] = ()
    stages: tuple[ExchangeStage, ...] = ()
    requires_long_range: bool = False
    requires_iteration: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "exchange_schedule",
            tuple(tuple(point) for point in self.exchange_schedule),
        )
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "stages", tuple(self.stages))
        if (
            type(self.requires_long_range) is not bool
            or type(self.requires_iteration) is not bool
        ):
            raise TypeError("service requests must be booleans")
        if self.requires_long_range:
            raise ValueError("long-range operators are unavailable in local-MP")
        if self.requires_iteration:
            raise ValueError("iteration execution is unavailable in local-MP")
        if any(not isinstance(field, FieldSpec) for field in self.fields):
            raise TypeError("fields must contain FieldSpec declarations")
        if any(not isinstance(stage, ExchangeStage) for stage in self.stages):
            raise TypeError("stages must contain ExchangeStage declarations")
        declarations = {field.name: field for field in self.fields}
        if len(declarations) != len(self.fields):
            raise ValueError("field names must be unique")
        if any(field.lifetime == "iteration" for field in self.fields):
            raise ValueError(
                "iteration field lifetime requires unavailable iteration execution"
            )
        if self.stages:
            if tuple(stage.stage_id for stage in self.stages) != tuple(
                point[0] for point in self.exchange_schedule
            ):
                raise ValueError(
                    "stage IDs must match the ordered exchange schedule"
                )
            for stage, (_, width) in zip(self.stages, self.exchange_schedule):
                if any(name not in declarations for name in stage.field_names):
                    raise ValueError("stage references an undeclared field")
                stage_fields = [
                    declarations[name] for name in stage.field_names
                ]
                if any(field.scope != "owned" for field in stage_fields):
                    raise ValueError(
                        "communicated stage fields must have owned scope"
                    )
                if sum(field.width for field in stage_fields) != width:
                    raise ValueError(
                        "stage field widths differ from scheduled width"
                    )
        elif self.fields and self.exchange_schedule:
            raise ValueError(
                "explicit fields require stages for the exchange schedule"
            )
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("model cutoff must be finite and positive")
        if not math.isfinite(self.skin) or self.skin < 0:
            raise ValueError("model skin must be finite and nonnegative")
        if (
            not isinstance(self.interaction_depth, int)
            or self.interaction_depth <= 0
        ):
            raise ValueError("interaction depth must be positive")
        if len(self.exchange_schedule) > self.interaction_depth:
            raise ValueError("exchange points cannot exceed interaction depth")
        ids = []
        for point in self.exchange_schedule:
            if len(point) != 2 or not all(isinstance(v, int) for v in point):
                raise ValueError(
                    "exchange points must be integer (layer_id, width) pairs"
                )
            layer_id, width = point
            if layer_id < 0 or layer_id >= self.interaction_depth or width <= 0:
                raise ValueError(
                    "exchange layer IDs must be within interaction depth and "
                    "feature widths must be positive"
                )
            ids.append(layer_id)
        if ids != sorted(set(ids)):
            raise ValueError("exchange layer IDs must be unique and ordered")
        for name in ("length_to_angstrom", "energy_to_kcal_mol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.feature_dtype != torch.float32:
            raise ValueError("PRIPS local-MP v1 supports FP32 model features")

    @property
    def cutoff_angstrom(self) -> float:
        return self.cutoff * self.length_to_angstrom

    @property
    def skin_angstrom(self) -> float:
        return self.skin * self.length_to_angstrom

    @property
    def semantic_metadata(self):
        geometry = (
            self.cutoff,
            self.skin,
            self.interaction_depth,
            self.exchange_schedule,
            self.length_to_angstrom,
            self.energy_to_kcal_mol,
            str(self.feature_dtype),
        )
        return (
            geometry,
            tuple(
                (
                    field.name,
                    field.width,
                    field.scope,
                    field.lifetime,
                    field.differentiable,
                )
                for field in sorted(self.fields, key=lambda field: field.name)
            ),
            tuple((stage.stage_id, stage.field_names) for stage in self.stages),
        )

    @property
    def capability_hash(self) -> int:
        payload = json.dumps(
            self.semantic_metadata,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def to_model_lengths(self, lengths_angstrom: torch.Tensor) -> torch.Tensor:
        return lengths_angstrom / self.length_to_angstrom


def map_atom_types(
    global_atom_types: torch.Tensor,
    atom_local: torch.Tensor,
    global_atom_count: int,
) -> torch.Tensor:
    """Map one declared type per global atom through SPONGE's local IDs."""
    if (
        global_atom_types.ndim != 1
        or global_atom_types.numel() != global_atom_count
    ):
        raise ValueError("atom type map must have one entry per global atom")
    if global_atom_types.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("global atom types must be integers")
    if atom_local.ndim != 1 or atom_local.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("atom_local must be a one-dimensional integer tensor")
    if bool(torch.any(atom_local < 0)) or bool(
        torch.any(atom_local >= global_atom_count)
    ):
        raise ValueError("atom_local contains an invalid global atom ID")
    return global_atom_types.to(atom_local.device)[atom_local.long()]


@dataclass(frozen=True)
class CSRNeighborView:
    """Borrowed runtime CSR, valid only until its neighbor epoch changes.

    Rows are owned receivers; source indices address owned-plus-ghost atoms.
    A shift q translates the source by q @ cell, so receiver-minus-sender
    vectors are positions[target] - positions[source] - q @ cell.
    Validation reads scalar predicates only when a new view is acquired;
    the arrays themselves remain on their runtime device.
    """

    neighbor_epoch: int
    row_ptr: torch.Tensor
    source_local_index: torch.Tensor
    image_shift: torch.Tensor

    def validate(
        self, num_owned: int, num_ghost: int, device: torch.device
    ) -> None:
        if self.neighbor_epoch <= 0 or num_owned < 0 or num_ghost < 0:
            raise ValueError(
                "CSR requires a positive epoch and nonnegative counts"
            )
        tensors = (self.row_ptr, self.source_local_index, self.image_shift)
        for tensor in tensors:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.int32
            ):
                raise TypeError("CSR arrays must be int32 tensors")
            if tensor.device != device or not tensor.is_contiguous():
                raise ValueError(
                    "CSR arrays must be contiguous on the runtime device"
                )
        edges = self.source_local_index.numel()
        if (
            self.row_ptr.shape != (num_owned + 1,)
            or self.source_local_index.shape != (edges,)
            or self.image_shift.shape != (edges, 3)
        ):
            raise ValueError(
                "CSR row/source/shift shapes disagree with domain counts"
            )
        invalid = (
            (self.row_ptr[0] != 0)
            | (self.row_ptr[-1] != edges)
            | torch.any(self.row_ptr[1:] < self.row_ptr[:-1])
            | torch.any(self.source_local_index < 0)
            | torch.any(self.source_local_index >= num_owned + num_ghost)
        )
        if bool(invalid):
            raise ValueError(
                "CSR row pointers or owned-plus-ghost source indices are invalid"
            )


@dataclass
class OwnedPrediction:
    energy: torch.Tensor | None
    forces: torch.Tensor
    virial: torch.Tensor | None

    def validate(
        self,
        num_owned: int,
        need_energy: bool,
        need_virial: bool,
        device: torch.device | None = None,
    ) -> None:
        if not isinstance(self.forces, torch.Tensor) or self.forces.shape != (
            num_owned,
            3,
        ):
            raise ValueError("owned forces must have shape (num_owned, 3)")
        if self.forces.dtype != torch.float32:
            raise ValueError("owned forces must be FP32")
        if device is not None and self.forces.device != device:
            raise ValueError("owned forces are on the wrong device")
        if need_energy != (self.energy is not None):
            raise ValueError("energy presence must match need_energy")
        if self.energy is not None and (
            self.energy.shape != () or self.energy.dtype != torch.float32
        ):
            raise ValueError("owned energy must be an FP32 scalar")
        if (
            device is not None
            and self.energy is not None
            and self.energy.device != device
        ):
            raise ValueError("owned energy is on the wrong device")
        if need_virial != (self.virial is not None):
            raise ValueError("virial presence must match need_virial")
        if self.virial is not None and (
            self.virial.shape != (3, 3) or self.virial.dtype != torch.float32
        ):
            raise ValueError("owned virial must be an FP32 3x3 tensor")
        if (
            device is not None
            and self.virial is not None
            and self.virial.device != device
        ):
            raise ValueError("owned virial is on the wrong device")

    def to_sponge_units(
        self, capabilities: LocalMPCapabilities
    ) -> "OwnedPrediction":
        energy_scale = capabilities.energy_to_kcal_mol
        return OwnedPrediction(
            None if self.energy is None else self.energy * energy_scale,
            self.forces * (energy_scale / capabilities.length_to_angstrom),
            None if self.virial is None else self.virial * energy_scale,
        )


class LocalMPModelAdapter(ABC):
    """Model implementation with an explicit owner readout and reverse force path.

    ``build_resident_inputs`` builds graph-dependent state from the current
    domain and the supplied runtime-owned CSR neighbor view. Adapters must not
    reconstruct adjacency geometrically. Positions and cell use model units.
    ``update_positions`` refreshes coordinates without changing graph edges.
    ``evaluate_owned`` must invoke each declared exchange point in order and
    return owner-unique additive outputs. The native graph and borrowed domain
    coordinates are one fixed geometry for a force call: adapter callbacks and
    other force plugins must not change owned or ghost coordinates after graph
    publication. Coordinate updates belong to SPONGE stepping before the next
    graph publication.
    """

    @property
    @abstractmethod
    def capabilities(self) -> LocalMPCapabilities: ...

    @abstractmethod
    def prepare_model(self, device: torch.device) -> None: ...

    @abstractmethod
    def validate_atom_types(self, global_atom_types: torch.Tensor) -> None:
        """Reject types unsupported by the prepared model.

        Called once after preparation with all global types, including on
        empty PP ranks. Type-independent models may implement a no-op.
        """
        ...

    @abstractmethod
    def build_resident_inputs(
        self,
        domain,
        atom_types: torch.Tensor,
        neighbor_view: CSRNeighborView,
        device: torch.device,
    ): ...

    @abstractmethod
    def update_positions(
        self, resident_inputs, positions: torch.Tensor
    ) -> None: ...

    def update_positions_from_sponge(
        self, resident_inputs, positions_angstrom: torch.Tensor
    ) -> None:
        """Detach and convert borrowed SPONGE Å coordinates to model units."""
        self.update_positions(
            resident_inputs,
            self.capabilities.to_model_lengths(positions_angstrom.detach()),
        )

    @abstractmethod
    def evaluate_owned(
        self, resident_inputs, exchange, need_energy: bool, need_virial: bool
    ) -> OwnedPrediction: ...
