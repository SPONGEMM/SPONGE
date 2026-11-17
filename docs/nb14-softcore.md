# Native nb14 Lennard-Jones soft-core

This optional extension softens only native 1–4 LJ pairs. Ordinary exclusions and
Coulomb scaling are unchanged. The exporter freezes the selected window into
C12_window = w*C12 and C6_window = w*C6, with delta = alpha*(1-w)^p*sigma^6.
The engine computes

    U_LJ = C12_window/(r^6+delta)^2 - C6_window/(r^6+delta).

Delta is measured in angstrom^6. Compute sigma from the unweighted physical
parameters or supply an explicit physical soft-core length. This interface does
not derive sigma, double-scale coefficients, impose complementary A/B weights,
or implement dynamic lambda updates / TI chain-rule derivatives.

## RAW input

Keep the existing nb14/nb14_extra files and set `nb14_softcore_in_file` to an
optional text file with the total pair count followed by `atom_i atom_j delta`
rows. Indices are zero-based. Rows must exactly follow the internal native order:
all nb14_extra rows first, then nb14 rows. Atom identities are checked, including
orientation; negative/nonfinite offsets, wrong counts and extra tokens fail.
For a named nb14 instance use the same module prefix. Retain zero-weight pairs.

## H5 input

Add optional `/forcefield/nb14/softcore_delta` or
`/forcefield/nb14_extra/softcore_delta` with shape [pair_count], matching that
root's atoms rows. Values must be finite and nonnegative. The existing roots
remain mutually exclusive. H5/RAW ownership must not be mixed.

Existing coefficient conventions remain unchanged: RAW nb14_extra and H5
nb14_extra store C12/C6; canonical H5 nb14 materialized three-column params store
12*C12 and 6*C6. Missing delta retains the old kernel path. An explicitly supplied
array, including all-zero arrays, enables zero-safe evaluation of inactive pairs.

## Evaluation and validation

Energy, Cartesian force and virial come from the same potential. An inactive LJ
pair returns exactly zero before division. In the soft-core path an exactly zero
Coulomb product is skipped before inverse-distance evaluation; nonzero Coulomb
remains unsoftened. At full physical coupling delta=0, a nonzero LJ pair is still
singular at zero separation, as required by the physical potential.

The array is carried through host/device upload and domain-local pair remapping.
A target-state rerun must load its full topology and delta array; changing only
lambda_lj is insufficient. Endpoints, independent finite-difference derivatives,
RAW/H5 parity, malformed input and kernel smoke tests are provided under
`tests/numerics/test_nb14_softcore*`. These are engineering checks, not an FEP
sampling/convergence or binding-affinity validation.

## Companion frozen ordinary-LJ controls

The independent ordinary-LJ mode is opt-in. It is separate from the native
nb14 offset interface above; neither interface enables the other.

For RAW input, `LJ_soft_core_independent_sterics_in_file` accompanies
`LJ_soft_core_in_file`. Its header is `atom_count weight_A weight_B`, followed
by exactly `atom_count` ordered rows `atom_index branch`. Membership codes are
0 for common/environment atoms, 1 for A-only atoms and 2 for B-only atoms.
Membership must be explicit, including physical atoms whose LJ coefficients
are zero. Endpoint LJ type indices are not membership labels.

For native H5, store `atom_branch` (integer shape `[atom_count]`) and `weights`
(shape `[2]`, A then B) under
`/forcefield/lj_soft_core/independent_sterics`. Both datasets are required when
the group exists. The ordinary LJ endpoint tables remain unweighted. Weights
are finite in [0,1] and need not sum to one.

For pairs involving A-only atoms, use the A endpoint coefficients with weight_A;
for B-only atoms use the B coefficients with weight_B. Both use
delta = soft_core_alpha*(1-weight)^soft_core_powfer*sigma^6. Sigma is derived
from the corresponding unweighted physical pair using the existing fallback
and minimum sigma settings. Common/environment-only pairs use the existing
`lambda_lj` core coordinate to interpolate the two hard LJ energies. A-B pairs
are excluded and their PME exclusions must be present in the input.

Current-charge direct Coulomb is evaluated exactly once, independently of the
two steric weights. A branch with any nonzero current charge must have its
steric weight exactly one. Long-range C6 coefficients use the same branch
weights. Domain-local remapping carries explicit membership in a separate
array; the legacy packed atom layout is unchanged.

This is a frozen-window periodic-boundary interface. SITS and the legacy
single-coordinate TI derivative are rejected. The solvent-specific kernel is
currently disabled in this mode because it assumes endpoint-A LJ parameters;
all ordinary pairs use the general kernel. Existing inputs without the optional
membership data retain the legacy path. Reruns must load the target window's
membership, both weights, current charges, core coordinate and nb14 offsets.

The PME exclusion correction skips zero-charge pairs before division and uses
the finite erf(beta*r)/r limit at exactly coincident charged pairs. These guards
prevent excluded A/B overlaps from producing NaNs; they do not soften physical
nonexcluded Coulomb or change its parameters.

## Ordinary core-pair restoration

Pairs changing between ordinary, excluded and native 1–4 categories require
explicit routing. Put the pair in the union exclusion list, retain its weighted
native14 fraction in the native14 channel, and supply its ordinary fraction
separately. Never append ordinary restoration rows to native14 input.

RAW `fep_core_ordinary_extra_in_file` contains a pair count and rows
`atom_i atom_j weighted_C12 weighted_C6 Coulomb_weight`. Native H5 stores
`/forcefield/fep_core_ordinary/atoms` (integer `[n,2]`) and `params` (`[n,3]`)
with the same physical C12/C6 and Coulomb-weight convention. Conversion to
internal derivative coefficients (12*C12, 6*C6) happens once on input. This
H5 group is independent of `/forcefield/nb14`; both may contain a fraction of
the same union-excluded pair. H5 and RAW cannot both own the ordinary table.
Softcore offsets are not supported in this common-core restoration channel.

Inside the ordinary cutoff the channel adds the weighted full LJ and Coulomb
pair. Outside it adds only the weighted `q_i*q_j*erf(beta*r)/r` contribution
removed by the PME exclusion correction. Its analytic force and virial use
the same potential. Current atomic charges are used once with the supplied
Coulomb weight; LJ coefficients are already weighted by the exporter.

This requires periodic PME with reciprocal and exclusion terms enabled,
union-excluded pairs, no SITS and one PP rank. It is a frozen-window interface,
not a dynamic-lambda derivative implementation. The ordinary restoration
numerics tests cover weighted native14 coexistence, cutoff and periodic-image
behavior, H5 input validation and RAW/H5 rerun energy/force parity.

## Machine-readable FEP capability query

`SPONGE --fep-capabilities-json` returns `sponge.fep-capabilities.v1` JSON and exits before controller, MPI, input-file, or GPU initialization. It advertises the native nb14 softcore, independent ordinary sterics, and core ordinary restoration input contracts. The declared scope is fixed-window energy/force/virial/rerun with the current PBC/PME, single-PP-rank, no-SITS/no-TI restrictions; it does not certify scientific convergence. Callers should bind this response to the actual executable SHA256 rather than infer capabilities from a version string or filename. Historical inputs do not require the query.

## Repeated multi-pair smoke

`tests/numerics/test_nb14_softcore_repeated.py --sponge /path/to/SPONGE --output /new/result/directory` compares legacy input, an explicit zero offset and a nonzero offset. Defaults are 256 independent pairs, 500 iterations at dt=0, three repetitions with rotated mode order. It checks analytic forces and first/last-frame invariance and records executable SHA256, total process time and the engine Calculate_Force timer. CPU and CUDA are separate runs.

The timer covers the complete force region, not isolated nb14 latency. This zero-time-step diagnostic does not measure MD throughput, moving-domain decomposition, shared-atom accumulation contention or FEP convergence. Physical charged coincidences remain singular; the extension softens LJ only. No support for finite unsoftened Coulomb at r=0 is claimed.

## Shared atoms and local pair packing

`tests/numerics/test_nb14_shared_runtime.py --sponge /path/to/SPONGE`
checks shared-atom accumulation, nonzero native14 electrostatics, input atom/pair
permutations, and all six RAW pressure components against a double
finite-difference energy oracle. An otherwise identical zero-nb14 control
removes ordinary/PME forces and pressure; tolerances are fixed in the fixture.
It uses 10 zero-time-step iterations, not equilibrium sampling. CPU and CUDA
are separate runs. Pass `--output /new/directory` to retain evidence.

`python tests/numerics/test_nb14_local_remap.py --build-dir build-cpu`
uses that CPU build's compile flags and directly exercises the production
local-pair packing function. It checks home/ghost filtering, atom indices,
C12/C6/charge-scale/delta row identity, repeated order/parameter updates, absent
softcore fields, and an empty domain. This optional ELF-toolchain fixture does
not simulate MPI communication or GPU packing. It complements the actual
CPU/CUDA force tests; input permutations alone are not DD validation.
