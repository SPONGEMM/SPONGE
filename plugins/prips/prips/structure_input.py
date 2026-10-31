"""Read a CPU-only initial structure for the native pure MLIP entry point."""

import os
from numbers import Integral


def _check_poscar(path):
    from ase.data import atomic_numbers
    from ase.io.formats import open_with_compression

    with open_with_compression(path, "rt") as source:
        for _ in range(5):
            source.readline()
        symbols = source.readline().split()
        if not symbols or any(
            symbol not in atomic_numbers or symbol == "X" for symbol in symbols
        ):
            raise ValueError(
                "POSCAR requires an explicit chemical element line"
            )
        counts = source.readline().split("!", 1)[0].split()
        coordinate_mode = source.readline().strip().lower()
        if coordinate_mode.startswith("s"):
            source.readline()
        for _ in range(sum(int(count) for count in counts)):
            source.readline()
        velocity_mode = source.readline().strip().lower()
        if velocity_mode.startswith(("d", "l")):
            raise ValueError(
                "POSCAR direct or lattice velocities are not supported; "
                "provide Cartesian atomic velocities"
            )


def _check_cif(atoms):
    for occupancy in atoms.info.get("occupancy", {}).values():
        if len(occupancy) != 1 or next(iter(occupancy.values())) != 1.0:
            raise ValueError("CIF requires full occupancy and ordered sites")
    for tag in ("_atom_site_disorder_group", "_atom_site_disorder_assembly"):
        values = atoms.info.get(tag, [])
        if not isinstance(values, (list, tuple)):
            values = [values]
        if any(str(value) not in (".", "?", "0") for value in values):
            raise ValueError("CIF disordered sites are not supported")


def read_structure(path, *, format=None, frame=0, cell=None, pbc=None):
    """Return Cartesian Angstrom, amu and optional Angstrom/ps CPU arrays.

    ``cell`` (three lengths) and ``pbc`` only supplement missing file data.
    Element labels follow first occurrence; atoms retain ASE's read order.
    Imports remain local so loading this module does not load any model code.
    """
    import numpy as np
    from ase import units
    from ase.io import read
    from ase.io.extxyz import key_val_str_to_dict
    from ase.io.formats import filetype

    if isinstance(frame, bool) or not isinstance(frame, Integral):
        raise TypeError("structure frame must be one integer")
    path = os.fspath(path)
    selected_format = filetype(path) if format is None else format.lower()
    if selected_format not in ("xyz", "extxyz", "vasp", "cif"):
        raise ValueError("structure format must be xyz, extxyz, vasp or cif")

    # Read both XYZ variants with extxyz so explicit metadata is never lost.
    if selected_format == "xyz":
        selected_format = "extxyz"
    explicit_pbc = selected_format in ("vasp", "cif")
    kwargs = {}
    if selected_format == "extxyz":

        def parse_properties(comment):
            nonlocal explicit_pbc
            properties = key_val_str_to_dict(comment)
            explicit_pbc = "pbc" in properties or "Lattice" in properties
            return properties

        kwargs["properties_parser"] = parse_properties
    elif selected_format == "vasp":
        _check_poscar(path)
    else:
        kwargs.update(store_tags=True, fractional_occupancies=True)

    atoms = read(
        path,
        format=selected_format,
        index=int(frame),
        parallel=False,
        do_not_split_by_at_sign=True,
        **kwargs,
    )
    if not len(atoms) or np.any(atoms.numbers <= 0):
        raise ValueError("structure requires atoms with chemical elements")
    if selected_format == "cif":
        _check_cif(atoms)
    if atoms.constraints:
        raise ValueError(
            "structure constraints are not mapped; use explicit SPONGE "
            "constraint input through the existing advanced entry point"
        )
    if np.any(atoms.get_initial_charges() != 0):
        raise ValueError(
            "nonzero initial charge requires a conditioning-aware advanced "
            "model entry point"
        )
    if np.any(atoms.get_initial_magnetic_moments() != 0):
        raise ValueError(
            "nonzero initial magnetic moments require a conditioning-aware "
            "advanced model entry point"
        )
    if "vel" in atoms.arrays or "velocities" in atoms.arrays:
        raise ValueError(
            "custom velocity columns have no standard units; provide ASE "
            "momenta or use initial_velocity instead"
        )

    box = np.asarray(atoms.cell, dtype=np.float64)
    if cell is not None:
        supplied_cell = np.asarray(cell, dtype=np.float64)
        if supplied_cell.shape != (3,):
            raise ValueError("structure cell must contain three box lengths")
        supplied_cell = np.diag(supplied_cell)
        if np.any(box) and not np.array_equal(box, supplied_cell):
            raise ValueError("structure cell conflicts with the file cell")
        box = supplied_cell
    if (
        not np.isfinite(box).all()
        or np.any(np.diag(box) <= 0)
        or np.any(box != np.diag(np.diag(box)))
    ):
        raise ValueError(
            "structure requires a positive axis-aligned orthogonal cell"
        )
    boundaries = atoms.pbc
    if pbc is not None:
        supplied_pbc = np.asarray(pbc)
        if supplied_pbc.shape != (3,) or supplied_pbc.dtype != np.bool_:
            raise ValueError("structure pbc must contain three booleans")
        if (explicit_pbc or np.any(boundaries)) and not np.array_equal(
            boundaries, supplied_pbc
        ):
            raise ValueError("structure pbc conflicts with the file boundary")
        boundaries = supplied_pbc
    if not np.all(boundaries):
        raise ValueError("structure currently requires all-periodic boundaries")

    positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
    masses = np.ascontiguousarray(atoms.get_masses(), dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("structure positions must be finite")
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("structure masses must be positive and finite")
    velocities = None
    if atoms.has("momenta"):
        velocities = np.ascontiguousarray(
            atoms.get_velocities() * (1000 * units.fs), dtype=np.float64
        )
        if not np.isfinite(velocities).all():
            raise ValueError("structure velocities must be finite")
    symbols = atoms.get_chemical_symbols()
    type_names = list(dict.fromkeys(symbols))
    indices = {symbol: index for index, symbol in enumerate(type_names)}
    return {
        "positions": positions,
        "cell": np.ascontiguousarray(box),
        "masses": masses,
        "velocities": velocities,
        "type_names": type_names,
        "atom_types": np.array([indices[s] for s in symbols], dtype=np.int32),
    }
