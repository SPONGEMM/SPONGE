# Dihedral stress correction

The periodic dihedral force kernel used two incorrect signs in its virial sum. This affects the six-component configurational stress whenever dihedral forces contribute. It can therefore affect pressure control and stress-based viscosity measurements even when the energy and Cartesian forces are correct.

With j as the local image origin, let a=ri-rj, b=rk-rj, c=rk-rl and g=dE_drj_part. The computed forces are fi, fj=g-fi, fk=-fl-g and fl. Their virial is

```
fi ⊗ a + fk ⊗ b + fl ⊗ (b-c)
= fi ⊗ a - g ⊗ b - fl ⊗ c.
```

The corrected expression uses the force kernel's bonded displacement images. The l image is reconstructed along the bonded chain; it need not be the independently shortest j–l image. The existing tensor storage symmetrizes off-diagonal products and the stress observer applies its existing factor convention. Each dihedral remains counted once on its first owned atom.

`test_dihedral_stress_runtime.py` evaluates an independent scalar torsion potential, differentiates all Cartesian coordinates and all nine homogeneous strains, and compares the actual executable's forces and six stress components. It checks a nonplanar configuration, reflection, axis permutation, and a translated chain crossing the periodic boundary. It fails on the old executable's stress and passes after the correction. This is a numerical regression, not a liquid-viscosity acceptance test.

The external TraPPE-UA candidate gate additionally covers 24 torsion angles, three bond angles, and single-/double-molecule LJ and full potentials. All 31 configurations pass the declared SPONGE and LAMMPS static tolerances after the correction; the old SPONGE executable passed only seven. Cartesian force arrays are unchanged across the fix in all 31 cases. The source/model inputs, raw before/after results, tolerances and limitations are archived outside the repository under the project's `notes/evidence/hexane-static-20261028` directory.

Re-evaluate previous stress-derived results for models containing dihedrals before using them scientifically. The branch's earlier LJ atom-fluid and rigid SPC/E water gates contain no dihedral potentials and are not evidence for, or invalidated by, this particular correction. Constraint virials, liquid sampling, model prediction accuracy and experimental agreement require separate validation. CPU tests and CUDA compilation were checked; GPU runtime was unavailable on this host.
