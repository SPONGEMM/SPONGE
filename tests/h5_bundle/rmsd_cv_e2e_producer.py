"""Create native and converted RMSD fixtures using a producer's own Python."""

import importlib
from pathlib import Path
import sys

import h5py
import numpy as np


def main():
    package, output, rotate_text = sys.argv[1:]
    xponge = importlib.import_module(package)
    importlib.import_module(package + ".forcefield.amber.ff14sb")
    converter = importlib.import_module(package + ".io_bundle")
    root = Path(output)
    molecule = xponge.get_peptide_from_sequence("AA")
    if package == "XpongeCPP":
        molecule.set_box_padding(15.0)
    else:
        molecule.box_length = [50.0, 50.0, 50.0]
        # Keep the whole peptide inside the box so periodic wrapping is not
        # conflated with reference serialization or restart behavior.
        center = np.mean(
            [(atom.x, atom.y, atom.z) for atom in molecule.atoms], axis=0
        )
        shift = 25.0 - center
        for atom in molecule.atoms:
            atom.x += shift[0]
            atom.y += shift[1]
            atom.z += shift[2]
    xponge.save_sponge_input_bundle(molecule, "system", root / "seed")
    with h5py.File(root / "seed/system_restart.spgr.h5") as handle:
        positions = handle["/particles/all/position/value"][0]
        box = handle["/particles/all/box/edges/value"][0]
    # Deliberately noncontiguous and unsorted, with an asymmetric deformation.
    selection = np.asarray([7, 1, 11, 4])
    reference = positions[selection].copy()
    reference += np.asarray(
        [[0.2, 0.4, -0.2], [-0.3, 0.1, 0.5], [0.1, -0.6, -0.2], [0.3, 0.2, 0.1]]
    )
    reference += np.asarray([1.5, -2.0, 0.75])
    protocol = xponge.SpongeProtocol(
        collective_variables=(
            xponge.ProtocolCollectiveVariable(
                name="rmsd_cv",
                type="rmsd",
                atom_indices=tuple(map(int, selection)),
                reference_coordinates=tuple(map(tuple, reference)),
                rotate=rotate_text == "true",
            ),
        )
    )
    for name, weight in (("native", 2.0), ("baseline", 0.0)):
        case = root / name
        xponge.save_sponge_input_bundle(
            molecule, "system", case, protocol=protocol
        )
        # Print and bias configuration goes through the existing /cv/config
        # route, while the RMSD definition and reference remain fully native.
        with h5py.File(case / "system_protocol.spgp.h5", "a") as handle:
            config = handle.require_group("/cv/config")
            text = h5py.string_dtype()
            config.create_dataset(
                "section/name", data=["print", "restrain"], dtype=text
            )
            config.create_dataset("section/key_offset", data=[0, 1, 4])
            config.create_dataset("section/count", data=2)
            config.create_dataset(
                "key", data=["CV", "CV", "weight", "reference"], dtype=text
            )
            config.create_dataset(
                "value",
                data=["rmsd_cv", "rmsd_cv", str(weight), "0.2"],
                dtype=text,
            )
            assert handle["/cv/rmsd_cv/coordinate"].shape == (4, 3)
        with h5py.File(case / "system_restart.spgr.h5") as handle:
            assert (
                "/parameters/restart/references/cv/rmsd_cv/coordinate"
                not in handle
            )
        (case / "mdin.bundled.spg.toml").write_text(
            'mode = "minimization"\ncutoff = 8.0\n'
            'input_h5_topology_path = "system_topology.spgt.h5"\n'
            'input_h5_protocol_path = "system_protocol.spgp.h5"\n'
            'input_h5_restart_path = "system_restart.spgr.h5"\n'
            'input_h5_restart_load = "structural"\n'
        )
    converter.convert_bundle_to_legacy(
        root / "native", root / "legacy", prefix="system"
    )
    np.savez(
        root / "oracle.npz",
        positions=positions,
        box=box,
        reference=reference,
        selection=selection,
    )


if __name__ == "__main__":
    main()
