# Decision: compiled stationary tensor, AO and grid consumers

Status: implemented
Date: 2026-09-19

## Problem

Merged #512 validated complete CPU RKS gradients but retained scalar/TensorIR
interpretation for source weights/reduction, local AO pullback and grid response.
Its grid response repeated every point/pair traversal for all nuclear coordinate
directions. Removing that overhead must not introduce a second PBE gradient
formula, change the SCF model, hide a reference fallback or mislabel Python/NumPy
orchestration as an entirely native public endpoint.

## Decision

Provide an explicit `execution="native"` diagnostic alongside the unchanged
default reference selector. Lower ordinary stationary TensorIR graphs through a
small checked FP64 CPU subset, reuse the established native XC jet-pullback
emitter, and compose a bounded grid adjoint from the existing local Graph roots.
The grid kernel uses two pair passes per point with atom-sized storage, not one
pass per coordinate. Center validation has its own work count and is included
in admission. Local norm/ratio/log/Becke partials remain compiler-generated;
branch handling, normalized-product reverse composition and atom traversal are
shared native grid-primitive responsibilities, not named-functional drivers.

Keep Python primitive enumeration/scatter and NumPy feature/BLAS/map operations
explicit. Do not enable public forces, claim global resource admission, infer
UKS/DF/ECP/d/f or CUDA support, or close #163 from this consumer slice.

## Numerical and failure invariants

The seven source signs, RKS occupation factors, native SCF-domain point model,
raw atomic quadrature and current D/F/C/epsilon/W lease are unchanged. Source
weights and final reduction consume the same StationaryGradientPlan graphs.
Native code must work with scalar-array, AO-pullback and TensorIR interpreter
entrypoints blocked. Native/reference parity alone is not an independent oracle.

Becke normalization retains log-scaled products and zero-factor counts. A factor
can round to zero while its derivative remains nonzero: erasing that derivative
or dividing the final grid weight by a tiny partition is incorrect. Independent
Decimal displaced-product tests cover that case as well as ordinary branches.
Outputs are staged and published only after all values pass. Dimension, dtype,
byte/work budgets and overflow checks precede memory accesses/allocation.

Generated sources use atomic publication and existing mismatches fail. The
artifact cache retains compiler/source/header/flag identity. The existing CUDA
emitter is not changed to support CPU execution, and the CPU subset rejects
unsupported operations rather than dispatching to an interpreter.

## Evidence

Maintained tests: `test_tensor_cpu.py`, `test_grid_native.py`, both execution
routes in `test_dft_complete_cpu.py`, and existing stationary/grid/XC/tensor
regressions. All-coordinate, three-step finite differences rebuild grids and
reconverge SCF. Tests also cover opaque-state revocation, changed geometry,
translation/permutation, partial tiles, hostile native ABI counts/budgets, late
failure/recovery and source generation without runtime/compiler access.

Independent review on asymmetric STO-3G water with the 24/8/16 native grid gave
native-versus-independent total maximum errors about `1.65e-11` (PBE) and
`1.26e-11` (LDA) Eh/bohr. Native-versus-reference differences were below `9e-16`.
An additional 75-digit Decimal direct-product objective agreed with the native
grid adjoint near `2.3e-16` on ordinary small fixtures. These are bounded-fixture
correctness observations, not arbitrary-domain or GPU claims.

A preliminary single-core AMD EPYC 7K62 review run compared complete prepared
SCF + snapshot + AO owner + gradient calls, with compiled artifacts warmed and
five interleaved repetitions per mode. Fixed/changed geometry speed ratios were
about 1.61/1.59 for H2 and 1.28/1.25 for water. The validation and final measured
source/library identity belong in the PR evidence; the reusable runner is
`tools/benchmark_stationary_consumers.py`. No isolated-kernel timing is presented
as an overall speedup, and initial shared-cache warmup times are not fair cold
compilation comparisons.

## Rejected alternatives and remaining work

Rejected: copying a PBE derivative into another driver, native wrappers calling
Python/scalar interpreters, dropping grid/Pulay terms, enforcing translation by
projection, promoting an unqualified public force flag, or a speculative fully
general CPU compiler unrelated to the stationary consumers.

Next integrate native orchestration and the remaining NumPy contractions under
one production resource/failure contract, register the verified public RKS
force endpoint, and independently qualify UKS/CUDA. Preserve this complete
reference diagnostic and its independent gates while changing schedules.

## References

Refs #163 and #396; follows merged #512 and its #484/#473/#455 prerequisites.
Current contracts: `docs/stationary_native_consumers.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
Implementation agent: Codex CLI 0.155.0
Implementation model: gpt-6-astra
