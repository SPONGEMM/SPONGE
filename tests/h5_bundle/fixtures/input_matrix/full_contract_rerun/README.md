# Virtual atom contract

This two-atom I/O fixture uses `2 1 0 0 0 0.5 0.5`: a type-2 virtual
atom with target 1 and three references to source 0. Repeated source indices
are valid for this affine construction, so the refreshed target coincides
with source 0. This is an I/O fixture, not a realistic molecular model.

A target must not occur among its own sources. Self-dependencies and cycles
are invalid: SPONGE constructs virtual sites in dependency order; it does not
solve implicit coordinate equations. The former record `2 1 0 1 0 0.5 0.5`
violated this rule and used the target's previous position.

Keep the legacy record, sidecar, native H5 virtual-atom payload, topology and
forcefield hashes, dependent bundle lineage, and runtime coordinate assertions
consistent when changing this fixture. Nontrivial type-2 geometry and force
redistribution are tested separately in the virtual-atom tests.
