# H5 Input Matrix Fixtures

These fixtures seed the input-side equivalence matrix for legacy SPONGE inputs,
native bundled H5 inputs, and bundled H5 inputs that carry legacy sidecar
bindings.

## Layout

- `core_structural/legacy_input`
  - Minimal normal-mode legacy input with coordinate, velocity, topology, and
    protocol/restart sidecar files.
- `core_structural/bundled_input`
  - The same case converted with XPONGE, with
    `/parameters/sponge/files/legacy_sidecars` removed from the H5 files and no
    external `legacy_sidecars/` directory.
- `core_structural/bundled_input_with_legacy_sidecar`
  - The same case converted with XPONGE, preserving H5 legacy sidecar key/path
    tables and bundle-local sidecar files.
  - `bundle/mdin.override_conflict.spg.toml` intentionally adds a legacy
    `mass_in_file` override to exercise conflict handling against H5 sidecar
    injection.
- `full_contract_rerun/legacy_input`
  - Broad rerun-mode legacy input generated from XPONGE's full converter
    contract sample. It covers trajectory input and most topology,
    protocol/restart, custom-force, enhanced-sampling, and force-field
    sidecar contracts.
- `full_contract_rerun/bundled_input`
  - Sidecar-stripped copy of the broad rerun-mode bundled case. It keeps typed
    H5 topology/protocol/restart/trajectory inputs but removes H5 legacy
    sidecar key/path tables and external `legacy_sidecars/`.
- `full_contract_rerun/bundled_input_with_legacy_sidecar`
  - XPONGE conversion of the broad rerun-mode case, preserving sidecar tables.

## Regeneration

The H5 files were generated with XPONGE's converter:

```sh
PYTHONPATH=/home/youmans/sidereus/XPONGE \
  /home/youmans/sidereus/XPONGE/.venv/bin/python \
  -m Xponge legacy-to-bundle <legacy_input> -o <bundled_output>
```

Use `core_structural` for the first three-way smoke matrix. Use
`full_contract_rerun` to extend contract coverage before enabling the complete
normal/rerun x legacy/bundled-in x legacy/bundled-out x VDS matrix.

Topology identities are canonical SHA-256 hashes of the logical datasets under
`/atoms/`, `/residues/`, `/forcefield/`, and `/manybody/`; forcefield identities
cover `/forcefield/` and `/manybody/`. Use the SPONGE v2 dataset encoding
(path, logical dtype, shape, payload), prefixed by `topology.spgt.h5`. Sidecar
bindings and identity metadata are excluded so both bundle variants share an
identity. Propagate identity updates to protocol, restart, trajectory, and mdin
references together. Atom-order and restart-state identities remain separate.
