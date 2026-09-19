# Generated directional first-integral matrix consumers

The opt-in `first_directional` compiler and `first_directional_execute` owner
provide conventional Cartesian S/T/V and four-center directional contractions
on CUDA. This is an integral consumer plus a small RHF H1/S1 integration, not
a new public method, complete molecular HVP or device-resident response solver.

## Mathematical ownership

`DirectionalMatrixTerm` explicitly declares an output-matrix slot, two output
AO shell slots, an optional fixed external-weight matrix pair, and a signed
coefficient. For one Cartesian integral component it means

```text
output[slot, ao_i, ao_j] += coefficient * weight[ao_k, ao_l] * dI(v).
```

An absent weight pair means one. The compiler does not infer HF, spin,
occupation, exchange fractions or nuclear-gradient signs. The RHF consumer
supplies the same declared J/K terms as its existing CPU first-order path.
Generated primitive evaluation reuses the established S/T/V and weighted-ERI
DAGs. Existing CPU raw-component source output is preserved.

Every mathematical center retains its own derivative slot. The device expands
physical directions through the explicit center-to-atom mapping before
contracting; coincident coordinates never implicitly merge slots. Nuclear
attraction includes the operator nucleus as well as both basis centers.
Generated Cartesian angular factors and staged radial normalization occur
once each. External weights and the direction are fixed primitive inputs;
their electronic/geometric responses must be supplied by downstream assembly.

The supported consumer rejects spherical AO slots and unimplemented DF/ECP or
range-separated operators instead of silently reinterpreting them. Selected
s/p/d/f component programs are representable; numerical qualification of a
particular selected primitive is distinct from validating every high-angular-
momentum molecular endpoint. No general all-f-shell speed/coverage claim is
made by the small RHF integration.

## Resident accumulation and failure behavior

`DirectionalFirstAccumulator` uses the shared TensorIR CUDA context/arena owner.
It retains the external weight matrix, physical direction, all requested output
matrices, a bounded primitive-record buffer and an error flag. Shell programs
with the same runtime/header identity, target and host-compiler ABI may append
to the same accumulator. The actual compiled program identity is verified;
different metadata cannot reuse a cached callable just because dimensions fit.

Only scalar completion/error checks return per chunk. Primitive derivatives
and per-shell matrices are never downloaded for CPU contraction. `finish()`
validates the complete matrix result and publishes an immutable snapshot after
all terms succeed. Invalid inputs, unsupported mappings, numerical overflow or
runtime failure block publication until a successful `reset()`. Reset clears
all accumulators and supports clean replay. Closing an owner is idempotent.

The generated consumer currently evaluates bounded local first-derivative
components before direction contraction; it does not materialize any molecular
coordinate-by-integral tensor. Further output pruning, recurrence sharing and
schedule tuning are separate performance work. There is no runtime coordinate
AD or SCF/DIIS tape.

## Resource and execution boundary

Storage admission counts the device arena, native publication candidate,
Python primitive staging, result copies, input conversion/snapshots and bounded
center staging. Caller-owned metadata, compiled code/library mappings, native
call stacks, compiler working memory and CUDA context/event overhead remain
explicit exclusions. This provider budget is not a simultaneous whole-Hessian
or full-response budget: the RHF J/K plan, SCF state and Krylov allocations have
separate owners and may coexist.

Strict FP64 compilation disables FMA contraction and rejects external NVCC
flag overrides for this path. No mixed-precision promotion is implied. Shared
source/header hashes, native artifact hashes, target identity, compiler resource
reports and actual program counts are available for qualification.

The small `generated_directional_first_order_cuda` integration stages immutable
shell geometry/radial records on the host, while derivative, direction, density
and AO accumulation arithmetic run on CUDA. It returns only two final host
matrices for the existing response layer. AO/MO transforms and GMRES remain on
the host. The source exporter retains its existing 12-Cartesian-AO/four-atom,
all-electron closed-shell RHF limit. Its convenience wrapper is not a retained
production HVP plan.

## Validation and performance scope

Pure tests cover typed terms, component/normalization identity, budget admission,
installed assets and generation without native-library loading. Optional real-
device tests compare against independent native derivatives, including a
selected f-shell quartet with repeated physical centers and signed density
weights. Method tests compare H1/S1 and complete directional density responses
with independently reconverged native finite differences at three step sizes.
Translation, raw symmetry, zero/scaled directions, failed records, ABI mismatch
and clean replay are checked without projection or post-hoc symmetrization.

Expensive CUDA compilation and real-device qualification remain opt-in; routine
CPU tests do not require a GPU. Cold code generation can dominate small systems,
and the conservative component/chunk schedule repeats some primitive work.
This implementation establishes executable correctness and residency, not a
speedup, large-system scalability or production schedule promotion.

See the [decision note](../.agents/notes/implemented/numerics/2026-09-19-directional-first-cuda.md)
and the [Hessian integration boundary](hessian.md#directional-nuclear-rhs-and-density-response).
