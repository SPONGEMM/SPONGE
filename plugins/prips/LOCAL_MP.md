# External short-range model adapters

`prips.local_mp` supplies model-independent contracts and execution. Model
implementations belong to their own projects and depend on this API; PRIPS
neither imports model libraries nor chooses a model by name. A user plugin
exports an instance as `local_mp_adapter` and the global integer `atom_types`.
For the rationale behind runtime changes and the C++/Python boundary, see the
[maintainer design guide](../../docs/local-mp-design.md).

## Scope and responsibilities

SPONGE owns integration, atom ownership, domain migration, periodic geometry,
neighbor candidates and PP MPI transport. One PP rank holds one model replica
on its assigned CUDA device. The current production transport exchanges FP32
features; there is no separate PyTorch distributed process group. This API
supports finite-depth, finite-range models whose cross-domain hidden state can
be expressed as fixed-width fields on owned atoms, with additive owned outputs.
Hidden state attached to variable directed-edge rows is not exchanged by this
version. Native edge-gradient force assembly is an output service, not an
edge-hidden-state transport. Such architectures require a separate protocol
extension. Long-range and iterative services are explicitly unavailable.

The external adapter loads and admits a supported complete model, declares its
cutoff/skin and ordered exchange schedule, maps atom types, builds model inputs
from SPONGE's graph, and computes unique-owned energy and derivatives. It must
retain all energy terms of that supported model, not only a selected network
subgraph. Model-specific compilation and acceleration stay with the adapter.

## Native service and Python frontend

The native local-MP service owns domain generations, borrowed graph publication,
and the communication/edge-output execution workspace. The main MD loop still
owns graph storage, halo refresh policy and force ordering; DD owns the halo map
and transport. The plugin loader negotiates capabilities and forwards ABI calls
to the native service. Shutdown releases the service before controller teardown.

PRIPS bridges native buffers and completion semantics to Python tensors. Its
model-independent Python driver manages adapter inputs and autograd exchange;
model code remains external. Exchange returns only after native work completes.
This split does not introduce an asynchronous communication contract or a second
transport runtime. The service tables use C linkage behind the existing C++
plugin loader; the complete loader interface is not a language-neutral C ABI.

## Public entry points

Import `LocalMPModelAdapter`, `LocalMPCapabilities`, `FieldSpec`, `ExchangeStage`,
`CSRNeighborView`, `OwnedPrediction`, `LayerwiseGhostExchange` and
`ResidentLocalMPDriver` from `prips.local_mp`.

Do not import underscore-prefixed implementation helpers or copy PRIPS
contract classes into a model project.

## Geometry and lifetime

`build_resident_inputs(domain, atom_types, neighbor_view, device)` receives:

| Domain field | Meaning |
| --- | --- |
| `num_owned`, `num_ghost` | Local rows are owned first, then ghosts. |
| `atom_local` | Local-row to global-atom index map. |
| `positions` | Owned plus ghost coordinates, already in declared model length units. |
| `cell`, `pbc` | Current cell in model length units and periodic axes. |
| `domain_epoch` | Layout identity; a changed epoch invalidates local mappings even if shapes agree. |

The CSR has one row per owned receiver, source indices into owned-plus-ghost
rows, and integer periodic image shifts. `row_ptr` has shape `(num_owned + 1,)`,
`source_local_index` has shape `(E,)`, and `image_shift` has shape `(E, 3)`; all
three are contiguous device int32 tensors. Its tensors are borrowed from the
runtime and are valid only for that published neighbor epoch. Do not mutate
or retain an old graph after rebuild. Candidate edges include skin; adapters
select their model's actual cutoff, including any type-dependent cutoffs,
without performing another geometric neighbor search.

Coordinates are fixed throughout each force call. The normal
`update_positions` receives model units; optional
`update_positions_from_sponge` receives borrowed Angstrom coordinates and must
not mutate them. Keep autograd-owned storage separate from borrowed runtime
coordinates. A domain or neighbor epoch change rebuilds resident inputs. Native graph
publication also rebuilds for cell changes or when motion reaches the declared
skin threshold (half the skin); zero skin requests rebuilds each force call.

## Native boundary

Local-MP plugins initialize through `Initial_Local_MP_V1` with one
`SPONGE_LOCAL_MP_API_V1` table (`api_version = 1`). CPU hosts may leave CUDA
callbacks null and reject active local-MP setup; CUDA hosts require exchange
and owned submission callbacks. Edge-gradient models additionally require
selected-edge assembly. Its unchanged PRIPS v2 base supplies ordinary
simulation access. The local-MP table supplies
capability registration, rank/device/domain and cell state, output needs,
CSR neighbors, stream-scoped feature exchange, selected-edge assembly, and
owned-output submission. No local-MP raw energy or virial output buffer is
exposed. Exchange completes producer-stream dependencies and native writes
before returning; caller tensors remain alive throughout each call.
Ordinary PRIPS v2 plugins continue to use `Initial_Stable`.

## Adapter API

The plugin exports `local_mp_adapter` (a `LocalMPModelAdapter` instance) and
`atom_types` (one integer per global atom, in SPONGE input order). The driver
preserves those values as int64 and maps them through `atom_local` on each domain
rebuild. Atom types are adapter-defined identifiers, not necessarily atomic numbers.

```python
LocalMPCapabilities(
    cutoff, skin, interaction_depth, exchange_schedule,
    length_to_angstrom, energy_to_kcal_mol,
    feature_dtype=torch.float32, fields=(), stages=(),
    requires_long_range=False, requires_iteration=False,
)
FieldSpec(name, width, scope, lifetime, differentiable)
ExchangeStage(stage_id, field_names)
OwnedPrediction(energy, forces, virial)
```

Cutoff and skin use model length units. Cutoff must be finite and positive;
skin finite and nonnegative; both conversion factors finite and positive.
The schedule is an ordered tuple of unique `(layer_id, width)` pairs with
`0 <= layer_id < interaction_depth` and positive widths. It lists actual
exchange points and need not have one entry per interaction layer.
Capabilities must remain unchanged after registration.

Implement these methods (signatures omit `self`):

```python
@property
def capabilities() -> LocalMPCapabilities: ...
def prepare_model(device: torch.device) -> None: ...
def validate_atom_types(global_atom_types: torch.Tensor) -> None: ...
def build_resident_inputs(domain, atom_types: torch.Tensor,
                          neighbor_view: CSRNeighborView, device: torch.device): ...
def update_positions(resident_inputs, positions: torch.Tensor) -> None: ...
def evaluate_owned(resident_inputs, exchange, need_energy: bool,
                   need_virial: bool) -> OwnedPrediction: ...
# Optional override; default detaches and converts before update_positions:
def update_positions_from_sponge(resident_inputs,
                                 positions_angstrom: torch.Tensor) -> None: ...
```

`prepare_model` loads one replica per PP rank. `build_resident_inputs` receives
local owned-plus-ghost types; `validate_atom_types` checks **all global types**
against the prepared model, including on empty PP ranks. `evaluate_owned` returns energy and virial exactly when their flags are
true, otherwise `None`; forces are always required, including shape `(0, 3)`
on empty ranks. See the [source contract](prips/local_mp/contracts.py).

## Model preparation

Before the first force callback, native SPONGE calls the PP-only setup hook.
The driver runs `prepare_model` and `validate_atom_types(global_atom_types)`
once. Native aggregates preparation success before any feature exchange; a
failed rank stops the MPI world. PM ranks do not prepare models or enter this
PP collective. There is no new per-step collective or WORLD barrier.

Registration checks the communication contract across PP ranks: cutoff/skin,
interaction depth, exchange layout, units and feature dtype. The driver also
requires these declarations to remain unchanged during preparation. This
check does not compare model weights, model configuration or global type
contents. Deployment and external adapters must ensure every PP rank uses the
same model, configuration and global type ordering. Models with different
weights but identical communication declarations are not detected by SPONGE.
Model correctness requires independent E/F/virial validation.

`validate_atom_types` raises for unsupported input types; a type-independent
model may implement a no-op. No model-content fingerprint is required. The
driver refuses force calls before successful local preparation. The named
setup/error hooks leave the PRIPS v2 base unchanged; host and PRIPS must be
built from the same checkout.

## Feature exchange and empty ranks

All PP ranks, including zero-owned and zero-edge ranks, execute the same declared
forward and reverse schedule. Do not return early on empty input. Connect empty
outputs to autograd so backward still participates in distributed exchanges.
`exchange.forward_exchange(owned_features, layer_id)` accepts contiguous CUDA
FP32 `[num_owned, width]` and returns owned rows followed by ghost rows. Autograd
backward applies the transpose: ghost cotangents accumulate into their owners.
`exchange.exchange_stage(stage_id, owned_fields)` provides the named-field path.

For explicit custom backward implementations, `forward_exchange_into(
local_features, output, layer_id)` and `reverse_local_gradients_into(
local_gradients, output)` use distinct, nonaliasing contiguous local slabs.
These APIs do not install an autograd backward for the caller. Reverse hidden
features in the corresponding reverse order; never skip ranks with empty rows.

## Shared native edge output

An adapter that computes `g = dE/dv` calls the native edge-output service:

```python
forces, virial = exchange.assemble_edge_gradients(
    selected_candidate_ids, edge_gradients, neighbor_view.neighbor_epoch,
    capabilities.length_to_angstrom, need_virial,
)
return OwnedPrediction(owned_energy if need_energy else None, forces, virial)
```

`selected_candidate_ids` is a contiguous device int64 array identifying the
current candidate CSR edges, in the same order as the contiguous FP32 `[E,3]`
gradients. Filter candidates by the actual model cutoff on every force call;
retain their IDs even when the CSR epoch is unchanged. A vector is
`v = positions[target] - positions[source] - image_shift @ cell` in model units.
The common native operation adds `+g` at the source and `-g` at the target
in a separate, reusable MLIP local-plus-ghost force workspace. It clears this
workspace for each call and sums its ghost forces back to their owners exactly
once using SPONGE's existing domain transport. It does not reverse the mixed
classical/ML force buffer. It computes `W = -sym(sum(v outer g))` only when
virial is requested, using block partial reductions. Do not scatter or reverse
these forces again in the adapter.

Outputs are caller-owned FP32 tensors in model units. With
`a = length_to_angstrom` and `b = energy_to_kcal_mol`, the driver converts
energy by `b`, force by `b/a`, and virial by `b`, then adds the predictions
to SPONGE's existing outputs. The adapter returns the physical `3x3` virial
tensor. The native submission service packs SPONGE's six-component storage as
`[Wxx, Wxy + Wyx, Wyy, Wxz + Wzx, Wyz + Wzy, Wzz]`; SPONGE's pressure
output divides the stored off-diagonal sums by two. Adapters must not apply
this packing or double their physical shear components themselves.
The driver submits detached contiguous CUDA FP32 tensors in one
`_submit_owned_outputs(forces, energy, virial)` call; the optional scalar energy
is borrowed as a one-element view. The mandatory `submit_owned_outputs` callback
belongs to `SPONGE_LOCAL_MP_API_V1`. Local-MP initialization rejects a
table missing this callback on a CUDA host. The service adds only
owned force rows and optional local energy/virial, performs no MPI operation,
and returns after all input reads and native writes complete. Inputs must not
alias native destination buffers. Python no longer writes native output buffers.
Native geometry normalizes coordinates and cell
before constructing vectors; its FP32 rounding follows the current CUDA
scalar length conversion. Empty selections
and empty owned ranks still call the service. A periodic self-image contributes
virial while its net atomic force is zero. The callback waits for the current
PyTorch CUDA stream and completes native work and transport before returning.
The `assemble_selected_edge_gradients` callback belongs to the same
`SPONGE_LOCAL_MP_API_V1` table. Missing native capability produces an error;
production has no PyTorch scatter fallback.
Standalone adapter tests explicitly supply an independent test oracle.

This operation assembles the final coordinate forces and optional virial.
Intra-model feature aggregation, owner-to-ghost feature exchange and the
hidden-feature backward remain unchanged. Adapters that differentiate local
coordinates directly keep the separate `reverse_owned_forces()` route and
return the same `OwnedPrediction` contract; they do not call edge assembly.

The implementation boundary is the [optional C ABI](include/sponge_plugin_api.h),
[host callback and negotiation](../../SPONGE/plugin/plugin.cpp),
[native operator](../../SPONGE/neighbor_list/mlip_edge_output.cpp),
[PRIPS bridge](src/main.cpp),
[shared adapter service](prips/local_mp/exchange.py), and
[unit conversion and additive integration](prips/local_mp/resident.py).
An edge-gradient adapter selects current candidate IDs and supplies gradients;
the [native runtime regression](../../tests/prips/local_mp/test_edge_output_mpi.py)
checks the operator against independent analytic arithmetic. Endpoint scatter followed by ghost
ownership return is conceptually aligned with LAMMPS/Kokkos ML-IAP force
assembly. The implementation was written independently; Kokkos code was not
copied. This architectural similarity does not establish a speedup.

On coordinate-only calls, the driver uses the optional
`update_positions_from_sponge(resident_inputs, positions_angstrom)` method.
Its input is borrowed, read-only SPONGE coordinates in Å. The default method
detaches and converts them with the declared length factor, then calls the
`update_positions(resident_inputs, positions_in_model_units)` method. An override
may write directly into adapter-owned, isolated resident storage, but must not
retain or modify the borrowed tensor. A changed neighbor epoch still discards
resident inputs, and domain or cell changes still rebuild with a converted
`DomainView`; the coordinate-only method does not update cached cell geometry.

The generic interface does not distribute arbitrary `torch.nn.Module` weights
or create a `torch.distributed` process group. Each model must provide an adapter with owner-only output and finite-range
semantics through its own upstream integration. CUDA
execution requires a CUDA+MPI SPONGE build and uses SPONGE's existing `D_MPI`
transport. The current CUDA+MPI device communicator spans all MPI ranks, so
allocate one visible GPU per MPI process, including PM ranks in a PP+PM launch;
only PP ranks evaluate the local model.

### Classical neighbor consumption

A pure local-MP Python plugin can declare `uses_classic_neighbor_list = False`
(the literal Python boolean). This is an immutable whole-run promise that its
adapter and all executed code never consume `Sponge.neighbor_list`. PRIPS opts
out only after its local-MP driver is initialized and only when the plugin has
none of the legacy hooks `After_Initial`, `Calculate_Force`,
`Set_Domain_Information`, or `Mdout_Print`. Mixed plugins remain conservative,
even with an explicit false declaration. Missing declarations and invalid
queries also remain conservative. Do not add hooks or change this promise after
initialization; restart with the appropriate declaration instead.

Native plugins can optionally export `int Uses_Classic_Neighbor_List(void)`.
SPONGE queries each plugin after all `After_Initial` callbacks complete. Only
zero opts out; missing exports, other return values, and caught query exceptions
mean classical neighbors may be consumed. The combined plugin requirement is
the OR of all plugin requirements. This does not alter the PRIPS v2 base or local-MP API table.
The declaration contributes to the combined runtime consumer decision. Only
when no active consumer needs the classical table does SPONGE release its
storage after initialization and skip subsequent updates. The shared-halo
policy separately accounts for all active halo consumers.

### Declared named fields

Adapters may append `fields` and `stages` to `LocalMPCapabilities`. An immutable
`FieldSpec(name, width, scope, lifetime, differentiable)` declares columns;
`ExchangeStage(stage_id, field_names)` binds ordered constituents to a scheduled
exchange. Stage IDs and summed widths must match `exchange_schedule` exactly.
Communicated fields have `owned` scope: `exchange.exchange_stage(stage_id,
owned_fields)` packs those owner rows and returns named local (owner plus ghost)
views using the existing forward copy and reverse transpose. Nondifferentiable
fields detach both inputs and outputs. Inputs remain contiguous FP32 tensors;
integer atom types continue through the resident type map.

`local` and `global` scopes can describe resident data but cannot appear in
communicated stages. Lifetimes are `static`, `domain`, and `step`; `iteration` is
reserved and rejected at capability admission. `requires_long_range=True` or
`requires_iteration=True` also fails admission: this interface supplies neither
long-range operators nor iterative execution.

Packed exchange APIs accept `exchange_schedule` without named fields or stages.
All capabilities use the same metadata structure, with empty field and stage
catalogs when none are declared. This metadata is included in capability
consensus: field catalog order is canonical by name,
while stage column order is significant.
Stable layout reuses resident static types; a changed domain epoch invalidates
and remaps the complete local type array even when its shape stays unchanged.

## Responsibilities and further reading

Model projects own complete-model admission, checkpoint loading, reference E/F/
virial tests and supported architecture lists. PRIPS owns the generic runtime.
See the [user guide](LOCAL_MP_USER_GUIDE.md) for deployment and the
[validation guide](LOCAL_MP_VALIDATION.md) for evidence and test selection.
