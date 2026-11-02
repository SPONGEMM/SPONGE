# TIP3P execution-matrix fixture

This fixture freezes the bundled half of the CPU rank-1 A/B execution matrix.
It was generated from `benchmarks/validation/thermostat/statics/tip3p_water`
with XPONGE `legacy-to-bundle`. The topology and protocol were last regenerated
with XPONGE commit `2c0dc3e`, whose residue contract is typed-only. The legacy
branch continues to use that source tree; `initial_velocity.txt` is shared
semantically with the bundled restart.

`common/` contains topology, protocol, and legacy sidecars shared by all
matrix cases. The two geometry directories contain only the restart state:
the nonorthogonal box uses angles 80, 100, and 110 degrees. Runtime thermostat
seeds vary by replica, while the initial structural and velocity state remains
fixed so legacy and bundled branches start from the same semantics.

`manifest.json` pins the generated binary inputs used by the gate. Regenerate
and review those hashes whenever the converter or input schema intentionally
changes.

The bundled topology has exactly one residue owner: `/atoms/residue_index` and
`/residues/atom_offset` are present, while `residue_in_file` is absent from the
active legacy-sidecar table and from `common/legacy_sidecars/`.
