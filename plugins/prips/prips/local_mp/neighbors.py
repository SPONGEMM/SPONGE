"""Acquire borrowed device CSR only when SPONGE publishes a new epoch."""

import torch

from .contracts import CSRNeighborView


class RuntimeNeighborCache:
    """Cache runtime-owned arrays, never an independently searched graph.

    Consumers must discard graph state borrowing the previous view before
    using a changed epoch. This cache cannot guard tensors retained elsewhere
    or replace the runtime's producer/consumer stream synchronization.
    """

    def __init__(self, sponge):
        self.sponge = sponge
        self.view = None
        self._domain_layout = None

    def invalidate(self):
        self.view = None
        self._domain_layout = None

    def get(
        self,
        num_owned: int,
        num_ghost: int,
        device: torch.device,
        before_acquire=None,
    ) -> CSRNeighborView:
        epoch_callback = getattr(self.sponge, "_mlip_neighbor_epoch", None)
        view_callback = getattr(self.sponge, "_mlip_neighbor_view", None)
        if not callable(epoch_callback) or not callable(view_callback):
            self.invalidate()
            raise RuntimeError(
                "runtime CSR requires the PRIPS local-MP v1 neighbor callbacks"
            )
        try:
            epoch = int(epoch_callback())
        except Exception:
            self.invalidate()
            raise
        layout = (num_owned, num_ghost, torch.device(device))
        if self.view is not None and self.view.neighbor_epoch == epoch:
            if layout != self._domain_layout:
                self.invalidate()
                raise RuntimeError(
                    "domain layout changed without a neighbor epoch change"
                )
            return self.view
        # Never retain a previously borrowed view after observing a new epoch,
        # including when acquisition or validation of the replacement fails.
        if before_acquire is not None:
            before_acquire()
        self.invalidate()
        if epoch <= 0:
            raise RuntimeError("runtime MLIP neighbor graph is unavailable")
        published_epoch, owned, edges, rows, sources, shifts = view_callback()
        if published_epoch != epoch:
            raise RuntimeError(
                "runtime neighbor epoch changed during acquisition"
            )
        if owned != num_owned or edges < 0:
            raise ValueError(
                "runtime CSR metadata disagrees with domain counts"
            )
        view = CSRNeighborView(
            published_epoch,
            torch.utils.dlpack.from_dlpack(rows),
            torch.utils.dlpack.from_dlpack(sources),
            torch.utils.dlpack.from_dlpack(shifts),
        )
        if view.source_local_index.numel() != edges:
            raise ValueError(
                "runtime CSR edge count disagrees with source tensor"
            )
        view.validate(num_owned, num_ghost, layout[2])
        self.view = view
        self._domain_layout = layout
        return view
