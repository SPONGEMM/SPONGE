# Local message passing: maintainer design guide

This extension lets an external finite-range neural potential use SPONGE's
domain decomposition without replicating its complete interaction-depth graph
on every rank. SPONGE remains responsible for MD state and communication; the
external adapter supplies the model calculation. The
[adapter contract](../plugins/prips/LOCAL_MP.md) specifies the API, and the
[user guide](../plugins/prips/LOCAL_MP_USER_GUIDE.md) covers deployment.
This document explains the implementation boundaries and the changes to the
existing runtime. It is not a performance or physical-validation report.

## Responsibility and code map

| Layer | Relevant files | Responsibility |
| --- | --- | --- |
| MD lifecycle | [main.cpp](../SPONGE/main.cpp) | Order migration, halo refresh, graph publication, force callbacks and output; invalidate views before changing their storage. |
| Domain ownership and transport | [Domain_decomposition](../SPONGE/Domain_decomposition/Domain_decomposition.cpp), [direct_halo](../SPONGE/Domain_decomposition/direct_halo.cpp), [halo_reference](../SPONGE/Domain_decomposition/halo_reference.cpp) | Keep existing update-group ownership, discover required ghosts, reuse a communication plan, and detect motion that invalidates it. |
| Neighbor geometry | [neighbor_list](../SPONGE/neighbor_list/neighbor_list.cpp), [mlip_neighbor_list](../SPONGE/neighbor_list/mlip_neighbor_list.cpp), [mlip_geometry](../SPONGE/utils/mlip_geometry.hpp) | Reuse spatial buckets to build a separate directed, periodic-image-aware candidate graph. |
| Native local-MP service | [local_mp_service](../SPONGE/local_mp/local_mp_service.cpp), [mlip_edge_output](../SPONGE/neighbor_list/mlip_edge_output.cpp) | Publish borrowed graph views, validate epochs, exchange tensors, assemble edge derivatives, and add owned outputs to MD buffers. |
| Plugin integration | [plugin.cpp](../SPONGE/plugin/plugin.cpp), [sponge_plugin_api.h](../plugins/prips/include/sponge_plugin_api.h) | Negotiate capabilities and setup, preserve the existing plugin interface, and forward local-MP calls to the native service. |
| Python bridge | [PRIPS native binding](../plugins/prips/src/main.cpp), [PRIPS Python entry](../plugins/prips/prips/__init__.py) | Expose native buffers and callbacks to Python and connect the external plugin to SPONGE's lifecycle. |
| Generic tensor execution | [contracts.py](../plugins/prips/prips/local_mp/contracts.py), [resident.py](../plugins/prips/prips/local_mp/resident.py), [exchange.py](../plugins/prips/prips/local_mp/exchange.py), [neighbors.py](../plugins/prips/prips/local_mp/neighbors.py) | Define adapter inputs/outputs, manage resident inputs, connect autograd to native exchange, and convert units. |
| External adapter | Maintained outside this repository | Load the complete supported model, map types, filter the actual cutoff, evaluate it, declare its exchange schedule, and validate supported atom types. |

The C++ service and Python driver are two layers of one execution path. There is
one model replica per particle (PP) rank, inside the existing SPONGE process.
Python does not own a second domain decomposition, neighbor search or MPI
worker pool. This division keeps device memory, MD ordering and transport in
the runtime that owns them, while letting model projects use their own PyTorch
operators, autograd and acceleration without adding model dependencies to SPONGE.
The native service can also be called by a native plugin; Python is its PyTorch
frontend, not the definition of the transport protocol.

## One force evaluation

1. Before the first model exchange, every PP rank prepares its model. Native
   setup aggregates preparation success across PP ranks. Communication
   declarations are checked at registration; identical model inputs are the
   responsibility of deployment and external adapters. PM ranks do
   not load a model. These checks establish communication compatibility and
   readiness, not model correctness.
2. SPONGE migrates atoms when required, refreshes the halo and updates ghost
   coordinates. It builds or reuses the local candidate graph, then publishes
   its view. Coordinates stay fixed throughout the force callbacks.
3. The Python driver rebuilds adapter inputs when domain or graph layout changes;
   otherwise it updates positions in adapter-owned storage. Model weights stay
   resident. Borrowed SPONGE coordinates must not be modified by the adapter.
4. At each declared exchange point, the adapter computes owned feature rows.
   Native transport fills ghost rows from their owners. Autograd's reverse
   operation adds ghost gradients back to owners through the same mapping.
5. The adapter produces unique-owned energy, forces and physical virial.
   Edge-gradient models use the common native assembly service; coordinate
   gradient models use the separate force-return operation. The driver converts
   units and submits owned outputs additively to SPONGE.
6. SPONGE uses its existing energy/pressure reductions and integrators. Empty
   PP ranks still participate in the declared communication schedule and
   contribute zero owned outputs.

The geometry halo covers one interaction radius plus skin. Exchanging hidden
features between layers carries information beyond that radius without storing
an interaction-depth-wide geometric halo. This trades replicated work/storage
for communication at each exchange point and during differentiation. It is not
automatically faster for every model or domain size.

## Why neighbor and graph code changes

Classical neighbor lists may be half lists or apply force-field exclusions.
A neural interaction graph needs all directed sources for each owned target,
explicit periodic images, and its own cutoff and skin. Reinterpreting the
classical table would couple two different physical contracts.

The implementation therefore reuses `NEIGHBOR_LIST::GRIDS` bucket allocation
and population operations, but gives the model graph independent storage. Its
CSR rows belong only to owned targets; source indices address owned plus ghost
atoms. Periodic copies are represented by integer image shifts, not duplicate
global atom IDs. This also permits periodic self-images where required by the
radius. Bucket overflow must trigger growth/rebuild before using its contents.

Candidates include skin. The adapter filters the actual physical cutoff on
every force call, even when the candidate graph is unchanged. A skin edge must
not silently become a physical interaction. Domain layout, cell changes or
motion can invalidate the graph; a shape match alone does not make an old view
valid. Domain and neighbor epochs communicate this lifetime to Python.

Classical storage is released/skipped only after all consumers are initialized
and none requires it. An unknown or legacy plugin remains a consumer. Shared
halo consumers also include bonded/listed/constraint code that does not use a
spatial neighbor table; that requirement is tracked separately from allocation
of the classical table. Without local-MP, the legacy update path is retained.

## Why domain and halo code changes

SPONGE migrates update groups together. A group's owner is determined by its
leading atom; other members can extend outside that owner's geometric domain.
Selecting ghosts only from the leading atom's distance to neighboring domains
does not establish member-to-member coverage for a model graph.

`DIRECT_HALO_PLAN` preserves whole-group ownership and packing. At halo rebuild,
PP ranks exchange bounding-box metadata, query potentially relevant coordinates,
and precisely select groups using member distances. Required groups go directly
to the target rank, including targets beyond the adjacent-domain route. Query
coordinates are temporary, not additional resident ghosts. Whole-group packing
can still retain members that are not themselves model neighbors; mixed
classical consumers also retain their required halo coverage.

This discovery has a cost: global rank metadata, pairwise queries and geometric
work at rebuild. It is not a per-layer global atom gather. Forward feature
copies and reverse additive accumulation reuse the resulting row mapping until
the next rebuild. Transport remains SPONGE's `D_MPI`; on the CUDA backend it
uses the runtime's NCCL device transport. The adapter creates no independent
NCCL communicator or `torch.distributed` process group.

The scheduled DD update interval alone does not bound atom motion. A shared
halo must satisfy the smallest skin among its active consumers. Refresh is
requested when any PP rank reaches half that skin, bounding the relative motion
of a pair by the full skin. Zero skin requests a halo refresh at every iteration's refresh check. The
refresh decision is also broadcast to PM ranks so their subsequent atom
exchange occurs on the same iteration; PM ranks do not join PP motion reduction.

## Output, ownership and synchronization

For an edge vector `v = r_target - r_source - image_shift @ cell` and
`g = dE/dv`, native assembly adds `+g` to the source and `-g` to the target.
It uses a separate model-force workspace, returns ghost contributions once,
and forms the physical virial `-sym(sum(v outer g))` when requested. It must
not reverse SPONGE's mixed classical/model force buffer or let the adapter
repeat the same scatter. There is no Python scatter fallback in this path.

Owned submission adds force rows and rank-local energy/virial without an MPI
reduction of its own. Existing SPONGE reductions then count them once. Physical
3x3 virial is packed into SPONGE's six-component representation with symmetric
off-diagonal sums; pressure output divides these sums by two. This conversion
belongs at the runtime boundary, not in each model adapter.

Native callbacks wait for producer-stream dependencies and complete their work
before returning. Graph publication and MD transitions also use conservative
device fences. Borrowed pointers cannot survive invalidation or rebuild.
These choices make ownership explicit, but do not provide communication/compute
overlap. An asynchronous extension would need explicit completion and lifetime
rules; removing fences alone would not be a correct optimization.

## Other changes to existing runtime code

| Change | Reason and boundary |
| --- | --- |
| [Initial coordinate handling](../SPONGE/MD_core/initial_coordinate.hpp) | Preserve already in-cell FP32 coordinates instead of perturbing them through an unnecessary fractional-coordinate round trip; outside coordinates still receive lattice translations and existing molecular connectivity handling. |
| [Restart output](../SPONGE/io/output_session.cpp) | Gather live DD coordinates/velocities before restart output even when trajectories are disabled. Reading output state must not wrap or modify the running DD coordinates. |
| Empty-domain launch guards in MD, thermostats, constraints and PM code | A valid decomposition can give a rank no owned atoms. Avoid invalid zero-sized launches or stale contributions while retaining required communication. This is not permission for a model to skip its exchange schedule. |
| [MD energy/pressure integration](../SPONGE/MD_core/MD_core.cpp) | Keep model totals separate from per-atom classical accumulation until the existing reductions; preserve physical virial conventions at submission. |

Empty arrays are no-ops for `Scale_List`, so velocity scaling does not need a
separate guard in each thermostat. Local thermostat kernels also skip empty
owned sets, but domain counts are updated before returning. Langevin/Andersen
invocation counters and the Nose–Hoover chain state still advance: an empty
rank can acquire atoms at a later migration. Global reductions and model
exchanges continue on all participating ranks.

The existing PRIPS v2 entry remains available. Local-MP uses one independently
named version-1 service table, rather than changing the old table's layout.
This version is a compatibility identifier, not a retained development variant.

## Support and review boundary

The admitted path is CUDA+MPI, fully periodic orthogonal cells, the native DD
rank/geometry limits, finite-range/depth models, and fixed-width atom-associated
features. Variable directed-edge hidden-state exchange, long-range operators,
iterative solvers and MLIP with the MC barostat are not supplied. Local-MP with
an enabled SITS module is rejected at initialization: SITS currently consumes
total energy before the world reduction, and its enhancement does not include
the separate model virial. This covers every SITS mode; supporting selected
model energy additionally requires its complete force and virial derivatives.
The check does not change the existing classical SITS path. Native edge
force assembly does not imply support for edge hidden-state communication.

Review the numerical and lifecycle invariants above alongside the
[retained regression tests](../plugins/prips/LOCAL_MP_VALIDATION.md). Tests with
generic fixtures do not qualify arbitrary external models, production MD,
mixed force fields or scaling. Those require separate adapter and deployment
evidence. Benchmark/profiler campaigns and model-specific adapters belong
outside this generic interface change.
