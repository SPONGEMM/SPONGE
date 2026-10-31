"""CPU deployment objects and explicit system type inputs for local-MP."""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from .contracts import LocalMPModelAdapter


class LocalMPDeployment(ABC):
    """Reusable CPU model/configuration owned by an external model package.

    Serialize with ``torch.save``. Do not retain system atom arrays, prepared
    adapters, CUDA tensors or runtime state in deployment objects.
    """

    @abstractmethod
    def create_adapter(
        self, type_names: tuple[str, ...]
    ) -> LocalMPModelAdapter:
        """Create an unprepared adapter for the ordered model type labels.

        Labels may repeat when several source types map to one model species.
        Integer atom types index this tuple. Device preparation belongs to the
        resident driver after SPONGE selects a device.
        """
        ...


def load_deployment(model_path, types_path, *, type_map=None):
    """Return an external adapter and CPU int64 global atom type indices.

    Paths are used as supplied; the caller resolves mdin-relative paths.
    Only load trusted deployment files: Torch deserializes Python objects.
    """
    with open(types_path, encoding="utf-8") as source:
        types = json.load(source)
    if not isinstance(types, dict):
        raise ValueError("types must be a JSON object")
    type_names = types.get("type_names")
    if (
        not isinstance(type_names, list)
        or not type_names
        or any(not isinstance(name, str) or not name for name in type_names)
        or len(set(type_names)) != len(type_names)
    ):
        raise ValueError("type_names must be unique nonempty strings")
    atom_types = types.get("atom_types")
    if not isinstance(atom_types, list) or any(
        type(index) is not int or not 0 <= index < len(type_names)
        for index in atom_types
    ):
        raise ValueError(
            "atom_types must be a one-dimensional integer array indexing "
            "type_names"
        )
    if type_map is not None:
        if not isinstance(type_map, Mapping) or any(
            name not in type_names or not isinstance(label, str) or not label
            for name, label in type_map.items()
        ):
            raise ValueError(
                "type_map must map declared source names to nonempty strings"
            )
        type_names = [type_map.get(name, name) for name in type_names]

    return load_deployment_with_types(model_path, type_names, atom_types)


def load_deployment_with_types(model_path, type_names, atom_types):
    """Internal entry for validated, in-memory structure labels and indices."""
    deployment = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(deployment, LocalMPDeployment):
        raise TypeError("model file must contain a LocalMPDeployment")
    adapter = deployment.create_adapter(tuple(type_names))
    if not isinstance(adapter, LocalMPModelAdapter):
        raise TypeError("create_adapter must return a LocalMPModelAdapter")
    return adapter, torch.tensor(atom_types, dtype=torch.int64, device="cpu")
