"""Owner-unique local message-passing interface for SPONGE PRIPS local-MP v1."""

from .contracts import (
    CSRNeighborView,
    ExchangeStage,
    FieldSpec,
    LocalMPCapabilities,
    LocalMPModelAdapter,
    OwnedPrediction,
)
from .deployment import LocalMPDeployment, load_deployment
from .exchange import LayerwiseGhostExchange
from .resident import ResidentLocalMPDriver

__all__ = [
    "CSRNeighborView",
    "ExchangeStage",
    "FieldSpec",
    "LocalMPCapabilities",
    "LocalMPModelAdapter",
    "LocalMPDeployment",
    "load_deployment",
    "OwnedPrediction",
    "LayerwiseGhostExchange",
    "ResidentLocalMPDriver",
]
