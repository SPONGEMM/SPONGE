"""Autograd-aware owner/ghost transport using SPONGE's PP D_MPI callbacks."""

import math
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .contracts import LocalMPCapabilities


def _call_transport(callback, rows: torch.Tensor) -> None:
    status = callback(rows)
    if status not in (None, 0):
        raise RuntimeError(
            f"SPONGE tensor exchange failed with status {status}"
        )


class _ForwardExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, exchange):
        ctx.exchange = exchange
        rows = torch.empty(
            (exchange.num_owned + exchange.num_ghost, owned.shape[1]),
            dtype=torch.float32,
            device=exchange.device,
        )
        rows[: exchange.num_owned].copy_(owned)
        _call_transport(exchange.forward_callback, rows)
        return rows

    @staticmethod
    def backward(ctx, gradient):
        exchange = ctx.exchange
        # Autograd may supply expanded cotangents (for example from sum).
        rows = exchange.reverse_local_gradients(gradient.contiguous())
        return rows[: exchange.num_owned].clone(), None


@dataclass(frozen=True)
class _CompiledExchangeLayout:
    exchange_schedule: tuple
    fields: MappingProxyType
    stages: MappingProxyType

    @classmethod
    def from_capabilities(cls, capabilities):
        """Compile already admitted metadata once; never holds step tensors."""
        return cls(
            capabilities.exchange_schedule,
            MappingProxyType(
                {field.name: field for field in capabilities.fields}
            ),
            MappingProxyType(
                {stage.stage_id: stage for stage in capabilities.stages}
            ),
        )


class LayerwiseGhostExchange:
    def __init__(
        self,
        num_owned: int,
        num_ghost: int,
        device: torch.device,
        forward_callback,
        reverse_callback,
        exchange_schedule,
        *,
        edge_output_callback=None,
        fields=(),
        stages=(),
        _layout=None,
    ):
        if num_owned < 0 or num_ghost < 0:
            raise ValueError("owned and ghost row counts must be nonnegative")
        self.num_owned = num_owned
        self.num_ghost = num_ghost
        self.device = torch.device(device)
        self.forward_callback = forward_callback
        self.reverse_callback = reverse_callback
        self.edge_output_callback = edge_output_callback
        if _layout is None:
            # Standalone callers still receive complete declaration admission.
            schedule = tuple(exchange_schedule)
            declared = LocalMPCapabilities(
                1.0,
                0.0,
                max((point[0] for point in schedule), default=0) + 1,
                schedule,
                1.0,
                1.0,
                fields=fields,
                stages=stages,
            )
            _layout = _CompiledExchangeLayout.from_capabilities(declared)
        self._layout = _layout
        self.exchange_schedule = _layout.exchange_schedule
        self.fields = _layout.fields
        self.stages = _layout.stages
        self._next_exchange = 0

    def assemble_edge_gradients(
        self,
        selected_candidate_ids: torch.Tensor,
        edge_gradients: torch.Tensor,
        neighbor_epoch: int,
        length_to_angstrom: float,
        need_virial: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Use native selected-edge assembly; return owner-only model-unit outputs."""
        if not callable(self.edge_output_callback):
            raise RuntimeError("native edge output assembly is unavailable")
        if not isinstance(selected_candidate_ids, torch.Tensor):
            raise TypeError("selected candidate IDs must be a tensor")
        if selected_candidate_ids.dtype != torch.int64:
            raise TypeError("selected candidate IDs must be int64")
        if (
            selected_candidate_ids.ndim != 1
            or selected_candidate_ids.device != self.device
            or not selected_candidate_ids.is_contiguous()
        ):
            raise ValueError(
                "selected candidate IDs must be contiguous on the runtime device"
            )
        self._validate(edge_gradients, selected_candidate_ids.numel())
        if edge_gradients.shape[1] != 3:
            raise ValueError("edge gradients must have width 3")
        if not isinstance(neighbor_epoch, int) or neighbor_epoch <= 0:
            raise ValueError("neighbor epoch must be a positive integer")
        if not math.isfinite(length_to_angstrom) or length_to_angstrom <= 0:
            raise ValueError("length conversion must be finite and positive")
        forces = torch.empty(
            (self.num_owned, 3), dtype=torch.float32, device=self.device
        )
        virial = (
            torch.empty((3, 3), dtype=torch.float32, device=self.device)
            if need_virial
            else None
        )
        status = self.edge_output_callback(
            selected_candidate_ids,
            edge_gradients,
            neighbor_epoch,
            length_to_angstrom,
            forces,
            virial,
        )
        if status not in (None, 0):
            raise RuntimeError(
                f"SPONGE native edge output assembly failed with status {status}"
            )
        return forces, virial

    def _validate(self, rows, expected_rows):
        if not isinstance(rows, torch.Tensor):
            raise TypeError("exchange input must be a tensor")
        if (
            rows.ndim != 2
            or rows.shape[0] != expected_rows
            or rows.shape[1] <= 0
        ):
            raise ValueError("exchange input has the wrong row count or width")
        if rows.dtype != torch.float32:
            raise TypeError("exchange input must be FP32")
        if rows.device != self.device:
            raise ValueError("exchange input is on the wrong device")
        if not rows.is_contiguous():
            raise ValueError("exchange input must be contiguous")

    def _validate_output(self, source, output):
        self._validate(output, source.shape[0])
        if output.shape != source.shape:
            raise ValueError("exchange output has the wrong shape")
        if output is source or (
            source.numel()
            and output.numel()
            and source.untyped_storage().data_ptr()
            == output.untyped_storage().data_ptr()
        ):
            raise ValueError("exchange output must not share input storage")

    def _validate_schedule(self, layer_id, width):
        if (
            self._next_exchange >= len(self.exchange_schedule)
            or (layer_id, width) != self.exchange_schedule[self._next_exchange]
        ):
            raise ValueError(
                "exchange layer or feature width differs from declared schedule"
            )

    def forward_exchange(
        self, owned_features: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        self._validate(owned_features, self.num_owned)
        self._validate_schedule(layer_id, owned_features.shape[1])
        result = _ForwardExchange.apply(owned_features, self)
        self._next_exchange += 1
        return result

    def exchange_stage(
        self, stage_id: int, owned_fields
    ) -> dict[str, torch.Tensor]:
        """Copy named owned columns and transpose only differentiable inputs."""
        if stage_id not in self.stages:
            raise ValueError("stage is not declared for named field exchange")
        stage = self.stages[stage_id]
        if set(owned_fields) != set(stage.field_names):
            raise ValueError("stage input field names differ from declaration")
        columns = []
        for name in stage.field_names:
            field = self.fields[name]
            rows = owned_fields[name]
            self._validate(rows, self.num_owned)
            if rows.shape[1] != field.width:
                raise ValueError(
                    "stage input field width differs from declaration"
                )
            columns.append(rows if field.differentiable else rows.detach())
        packed = columns[0] if len(columns) == 1 else torch.cat(columns, dim=1)
        local = self.forward_exchange(packed, layer_id=stage_id)
        result = {}
        offset = 0
        for name in stage.field_names:
            field = self.fields[name]
            rows = local[:, offset : offset + field.width]
            result[name] = rows if field.differentiable else rows.detach()
            offset += field.width
        return result

    def forward_exchange_into(
        self, local_features: torch.Tensor, output: torch.Tensor, layer_id: int
    ) -> None:
        """Fill a distinct slab when the caller owns the autograd backward."""
        self._validate(local_features, self.num_owned + self.num_ghost)
        self._validate_output(local_features, output)
        self._validate_schedule(layer_id, local_features.shape[1])
        output[: self.num_owned].copy_(local_features[: self.num_owned])
        _call_transport(self.forward_callback, output)
        self._next_exchange += 1

    def validate_complete(self) -> None:
        if self._next_exchange != len(self.exchange_schedule):
            raise ValueError(
                "adapter did not execute every declared exchange point"
            )

    def reverse_local_gradients(
        self, local_gradients: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate ghost cotangents into owners and return the full slab."""
        self._validate(local_gradients, self.num_owned + self.num_ghost)
        rows = local_gradients.contiguous().clone()
        _call_transport(self.reverse_callback, rows)
        return rows

    def reverse_local_gradients_into(
        self, local_gradients: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Fill a distinct cotangent slab without changing input gradients."""
        self._validate(local_gradients, self.num_owned + self.num_ghost)
        self._validate_output(local_gradients, output)
        output.copy_(local_gradients)
        _call_transport(self.reverse_callback, output)

    def reverse_owned_forces(self, local_forces: torch.Tensor) -> torch.Tensor:
        self._validate(local_forces, self.num_owned + self.num_ghost)
        if local_forces.shape[1] != 3:
            raise ValueError("coordinate gradients must have width 3")
        rows = local_forces.clone()
        _call_transport(self.reverse_callback, rows)
        return rows[: self.num_owned].clone()
