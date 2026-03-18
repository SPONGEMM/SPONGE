"""
prips.Sponge
===============
a virtual plugin interface module for communicating with the molecular dynamics simulation package SPONGE

Example - 0
---------------
```python
print("Hello World")
input("press any key to continue...")
```

Example - 1
---------------
```python
from prips import Sponge
Sponge.set_backend("cupy")
print(Sponge.md_info.crd)
input("press any key to continue...")
```

Example - 2
---------------
```python
from prips import Sponge
Sponge.set_backend("jax")

def Calculate_Force():
    print(Sponge.md_info.frc[:, 2])
```
"""

from typing import Literal

def set_backend(backend: Literal["numpy", "jax", "cupy", "pytorch"]) -> None:
    """
    Set the backend for PRIPS.

    Parameters
    -----------
    backend : Literal["numpy", "jax", "cupy", "pytorch"]
        Backend name used to convert DLPack tensors to framework tensors.
        The ``jax`` backend is read-only in PRIPS.
    """

class MD_INFORMATION:
    "Basic information class for MD"
    class system_information:
        @property
        def steps(self) -> int:
            "The number of the current step"

    @property
    def atom_numbers(self) -> int:
        "The number of atoms"

    @property
    def crd(self) -> SpongeDLPackTensor:
        "The atom coordinates with shape (atom_numbers, 3)"

    @property
    def frc(self) -> SpongeDLPackTensor:
        "The atom forces with shape (atom_numbers, 3)"

md_info = MD_INFORMATION()
" Basic information instance for MD "

class NEIGHBOR_LIST:
    @property
    def index(self) -> SpongeDLPackTensor | None:
        "Neighbor index matrix with shape (atom_numbers, max_neighbor_numbers)"

    @property
    def number(self) -> list[int] | None:
        "Neighbor count for each atom"

    @property
    def max_neighbor_numbers(self) -> int | None:
        "The allocated max number of neighbors for one atom"

neighbor_list: NEIGHBOR_LIST | None
" Neighbor list interface, available after SPONGE finishes initialization "

class DOMAIN_INFORMATION:
    @property
    def atom_numbers(self) -> int | None:
        "Local atom numbers in the current PP rank"

    @property
    def ghost_numbers(self) -> int | None:
        "Ghost atom numbers in the current PP rank"

    @property
    def pp_rank(self) -> int | None:
        "PP rank id of the current domain"

    @property
    def atom_local(self) -> SpongeDLPackTensor | None:
        "Global atom ids for local and ghost atoms"

    @property
    def atom_local_label(self) -> SpongeDLPackTensor | None:
        "Global-to-local membership flags, stored over max_atom_numbers"

    @property
    def atom_local_id(self) -> SpongeDLPackTensor | None:
        "Global-to-local index map, stored over max_atom_numbers"

    @property
    def crd(self) -> SpongeDLPackTensor | None:
        "Local plus ghost coordinates with shape (atom_numbers + ghost_numbers, 3)"

    @property
    def frc(self) -> SpongeDLPackTensor | None:
        "Local plus ghost forces with shape (atom_numbers + ghost_numbers, 3)"

dd: DOMAIN_INFORMATION | None
" Domain decomposition interface, available after SPONGE finishes DD initialization "

class CONTROLLER:
    "IO controller"
    def printf(self, *values: any, sep: str = " ", end: str = "\n") -> None:
        """Print the values to the screen and the mdinfo file"""

controller = CONTROLLER()
" IO controller instance"
