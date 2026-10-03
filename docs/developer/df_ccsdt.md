# DF-CCSD(T) same-Hamiltonian definition and factorized path

Status: issue #157 slices A-B, the first slice-C factorized-(T) endpoint, and
C2a's explicit energy-only source facade are implemented. Native Calculator
registration and production performance/memory qualification remain open.

## First supported method definition

The first validated DF-CCSD(T) variant deliberately separates the reference
approximation from the correlation-integral approximation:

- reference: conventional, unscreened, all-electron closed-shell RHF;
- orbitals, orbital energies, Fock matrix, and RHF reference energy: retained
  exactly from that conventional RHF calculation;
- correlation Hamiltonian: density-fitted two-electron integrals only;
- DF metric: the existing square-symmetric thresholded inverse square root;
- precision: FP64;
- frozen orbitals: unsupported in this slice;
- triples: standard canonical noniterative (T), with no local/truncated-triples
  approximation.

The method contract records the conventional reference identity, orbital and
auxiliary basis identities, geometry and AO representation, DF Hamiltonian
identity, metric convention/cutoff/effective rank/conditioning, Fock policy,
precision, frozen-orbital policy, and triples variant. These fields participate
in the contract identity instead of allowing two differently defined DF
Hamiltonians to share a cache/result identity.

Changing the correlation Hamiltonian does **not** recompute or replace the
conventional RHF Fock matrix. This is intentional: the first #157 variant is
“conventional RHF reference + correlation-only DF”.

## Dense same-Hamiltonian oracle

For small qualification systems, the oracle uses the existing DF source and
metric factor to build the whitened MO three-index tensor

```text
B[Q,p,q] = sum(mu,nu,P)
           C[mu,p] C[nu,q] A[mu,nu,P] M^(-1/2)[P,Q]
```

in the conventional RHF MO basis. It then reconstructs exactly one dense
same-DF-Hamiltonian tensor

```text
g_DF[p,q,r,s] = sum_Q B[Q,p,q] B[Q,r,s].
```

That dense tensor is exposed through a validation-only provider and passed to
the already-audited RCCSD and standard (T) equation stack. Consequently,
comparison with a later factorized implementation is a comparison of two
implementations of the **same Hamiltonian**, rather than a comparison of DF
against conventional four-center integrals.

An orthogonal rotation of the retained auxiliary B axis leaves `g_DF` and the
physical RCCSD(T) energy invariant. The slice-A tests exercise that gauge
invariance explicitly.

## Memory scope

The dense oracle is intentionally small-system-only. Before reconstruction it
preflights a numeric budget that includes the caller-owned B tensor, the
immutable owned B copy, the dense four-index result, and a conservative
full-sized contraction temporary. The production slices must not rely on this
`NMO^4` allocation.

The existing `DFProvider` remains responsible for bounded construction of B.
Slice B retains `B_ov` and `B_vv`, charges both arrays against the provider
budget, and retains only the smaller `ovov`, `ovvo`, `oovv`, `ovoo`, and
`oooo` four-index blocks. The full `ovvv` and `vvvv` arrays are never requested.

Every conventional singles/doubles term containing `ovvv` or `vvvv` is removed
from the DF TensorIR feed and evaluated by an exact auxiliary-index reduction.
The resulting singles/doubles correction tensors are injected back into the
same audited CCSD equation inventory. The auxiliary index is reduced one slice
at a time; the largest correction temporaries scale like T2 rather than
`Nvir^4`.

The factorized solver uses the same denominator, damping, DIIS, physical
residual, and fresh expanded-equation acceptance policy as conventional RCCSD.
Its independent expanded recheck also uses the factorized corrections, so an
optimized-path agreement alone cannot certify convergence.

The first Slice-C endpoint now evaluates standard canonical (T) directly from
the same retained `B_ov`/`B_vv` data model. For
`vvov[a,b,i,f] = sum_Q B_ov[Q,i,a] B_vv[Q,f,b]`, W1 reduces Q directly
into the occupied-space W tensor. It therefore forms neither a complete
`ovvv` tensor nor full T3. H2/H2O same-Hamiltonian tests compare the composed
factorized energy with the dense oracle, and an independent random-factor test
compares the factorized (T) correction with the audited dense triples equations.

### C2a executable energy facade

The source-level `df_rccsd_t_energy` facade now composes the already validated
pieces without changing their scientific ownership:

1. run a conventional unscreened CPU RHF export with no DF metric supplied;
2. build and validate the auxiliary metric factor;
3. relabel only the correlation Hamiltonian with
   `correlation_df_reference`;
4. execute the factorized DF-RCCSD + standard factorized (T) path.

The facade reports the full method contract, metric rank/conditioning, DF
provider statistics, and an explicit capability record. Its supported property
set is exactly `{"energy"}`. A force request fails before source validation or
solver work, so the conventional RCCSD(T) force capability cannot be inherited
accidentally.

C2a is deliberately not a native Calculator registration. It currently uses the
existing native RHF snapshot-export bridge and therefore inherits that bridge's
<=12-AO qualification boundary. The next registration slice must give the
native method owner the same correlation-DF Hamiltonian semantics instead of
merely enabling the existing conventional RCCSD(T) owner.

Remaining Slice-C work is C2b native Calculator energy registration plus C3
production performance/memory qualification (and later CUDA promotion); DF
gradients remain #158.

## Compiler-owned virtual residual and response actions

`generativeqc_compiler.cc.df_equations` derives the omitted `ovvv/vvvv`
contributions directly from the expanded conventional RCCSD inventory. It
substitutes one auxiliary slice of `B_ov` and symmetric `B_vv` before planning
binary contractions. New intermediates have at most two virtual axes, including
occupied-rich shapes; a scheduler constraint prevents reconstructing the dense
virtual blocks. An execution owner must accumulate every auxiliary slice once.

`build_df_virtual_response_programs` also generates amplitude JVP/VJP and
factor VJP actions using the existing dense-symmetry TensorIR AD. Amplitude
cotangents accumulate over auxiliary slices; each factor cotangent belongs to
its own slice. These actions cover the virtual correction, not the retained
smaller integral blocks, complete Lambda solution, or nuclear derivative chain.

`tools/generate_df_ccsd_native.py --output-dir <directory>` emits CPU/CUDA
runtime-shape actions, exact scratch-arena queries, equation hashes and operation
counts using the shared native CC emitter. The scratch queries exclude caller
inputs, accumulated outputs, device staging and endpoint state; a complete owner
must admit those as well. Generated outputs borrow the supplied scratch arena.
This generator is an internal building block and does not register a public
DF Calculator method or extend its qualified force domain.

The one-slice virtual actions have at most fifth-degree contraction work and
fourth-degree storage. The complete auxiliary sum adds the auxiliary population
to the work count. DF therefore avoids the dense virtual-integral storage but
does not, by itself, remove the usual sixth-order CCSD or seventh-order standard
triples work. Neither source generation nor small action tests establish large
complete-endpoint performance.

The native action tests use
`GENERATIVEQC_DF_CC_CUDA_TEST=1` for explicit real-device qualification; on the
local scheduled GPU they must run inside the repository's required Slurm job.
`benchmarks/df_ccsdt_large_oracle.py` supplies pinned independent PySCF references
for 230-AO ethane and 264-AO benzene, with explicit symmetric metric whitening,
conventional RHF, same-Hamiltonian DF correlation, and physical residual replay.
Its retained states are validation artifacts, not production inputs.

## Internal native RCCSD solver

`cc::Problem::naux`, `df_bov` and `df_bvv` select an explicit factorized
virtual representation. `ovvv` and `vvvv` must be empty; retained smaller
blocks must describe the same fitted Hamiltonian. The native CPU/CUDA solver
reuses the conventional DIIS and physical convergence policy. It accumulates
all Q slices for every current/trial/replay amplitude state. The compiler derives
the primary schedule in `cc/df_hoist.py` from the shared RCCSD inventory: prepare
amplitude-only tau once, accumulate the virtual contributions to Lvv, Wvoov,
Wvovo and Xv, then contract the complete sums with T2. The virtual ladder remains
inside the Q loop. `tools/generate_df_ccsd_hoisted.py` emits this schedule; no
materialized tensor has more than two virtual axes. This reduces repeated
contraction work without changing the formal leading CCSD scaling.

The planner charges preparation, every Q slice and the retained core. It selects
the reduced schedule only when its scalar contraction-summand count is smaller
and its complete owner storage fits the budget. Otherwise it uses the original
bounded schedule. `SolverOptions::df_auxiliary_reduction=false` forces that
fallback for qualification. Convergence always uses the original expanded
virtual actions and `tools/generate_df_ccsd_core.py` replay, independently of the
primary schedule.

Complete solver admission includes resident factors, accumulated corrections,
both core arenas, one-slice scratch, preparation and accumulated intermediates,
DIIS and retained/final host arrays. CUDA
uploads factors once; borrowed action outputs are consumed on the same stream
before scratch reuse. A sticky arithmetic flag spans all Q slices and the
core. Diagnostics report auxiliary slices, virtual operations, accumulation
calls, prepared/hoisted evaluations and exact scalar contraction summands across
the entire solve, including convergence replay. Summand counts exclude
elementwise operations and are not hardware FLOPs or timing predictions.

Conventional admission rejects the DF representation unless an owner explicitly
opts in. Native CUDA Lambda accepts the factorized representation as described
in [DF response composition](df_ccsdt_gradient.md); molecular triples and force
consumers remain conventional. The separate internal DF triples energy owner
is described below.
This internal supplied-Hamiltonian solver does not register a public DF
Calculator endpoint. Its qualification is
`tests/python/test_df_cc_native_solver.py`. The supplied-Hamiltonian solver is
also checked for 230-AO ethane using `benchmarks/df_ccsd_native_solver_probe.py`:
energy must agree within 3e-9 Eh, every amplitude within 1e-8, and expanded
physical residuals must be below 1e-10. That adapter reconstructs the same packed
AO factors consumed by the independent oracle before the symmetric MO transform;
stored full MO factors are not substituted for the oracle's packed source.
Complete hundreds-AO native CCSD(T) energy and forces still require the remaining
source/response owners.

## Internal native DF triples energy

`cc::triples::evaluate_df_cuda` evaluates standard closed-shell FP64 `(T)` from
supplied Q-major `B_ov/B_vv`, retained `ovoo/ovov`, Fov, T1/T2 and orbital
energies. It requires physically symmetric Bvv pairs and a canonical occupied/
virtual gap above the denominator threshold. It is an internal phase API;
public molecular descriptors and response consumers do not select it yet.

The compiler owns the panel and moment TensorIR in `cc/occupied_triples.py`.
`tools/generate_df_occupied_triples.py` derives direct BLAS products from the
shared GEMM contract and emits a fused scalar epilogue using the shared emitter.
For each `i>=j>=k`, the owner builds six W cubes with twelve GEMMs and computes
V elements inside the epilogue. All six occupied permutations are retained,
including repeats divided by the occupied 6/2/1 multiplicity. The virtual domain
is the full cube: the six original virtual rows have equal complete sums after
dummy-index relabeling. Folding this virtual cube independently, or keeping only
one occupied permutation, changes the energy.

One to three occupied integral panels replace full `ovvv`; six W cubes replace
full T3. Let `T=o(o+1)(o+2)/6` and `P` be the actual panel-build count. Complete
contraction work is `P Q v^3 + 6 T (v^4 + o v^3)` scalar summands; the epilogue
visits `T v^3` points. Standard triples retain seventh-order leading work.
Diagnostics separately report panel/moment GEMMs, epilogue/reduction kernels,
transfers, and admitted/observed storage; summands are not hardware FLOPs.

The owner uploads inputs once and orders every panel producer, W consumer,
epilogue and reuse on one owned stream. The numeric budget includes staged
inputs, panels, moments, reductions, a 4-MiB BLAS workspace and a conservative
96-MiB provider allowance. If the requested panel count does not fit, admission
retries with one panel; an infeasible one-panel plan fails before allocation.
The caller separately charges retained molecular/CC state. Result timing spans
validation, allocation, uploads, computation, readback, drain and destruction.
Any input, budget or device-arithmetic failure leaves results unpublished.

`tests/python/test_df_occupied_triples.py` covers the domain rewrite on CPU and,
with `GENERATIVEQC_DF_TRIPLES_CUDA_TEST=1` plus `GENERATIVEQC_LIBRARY`, real CUDA
energy, work, exact-budget/fallback, repeatability and failure behavior. Real-GPU
checks run in finite Slurm allocations. `benchmarks/df_triples_native_probe.py`
compares large supplied states with the independent oracle; its timings cover
the complete triples phase, excluding RHF, source construction, CCSD and forces.
See the [decision note](../../.agents/notes/implemented/performance/2026-10-03-df-occupied-triples.md)
for the algebraic rationale and qualification evidence.

## Internal native CUDA molecular source

`cc::build_df_source_cuda` builds correlation-only DF integrals from normalized
orbital/auxiliary systems and a conventional physical RHF reference. The
internal `run_rccsd_native_state` entry accepts an optional correlation auxiliary
system to compose native CUDA RHF, this source, and the native DF CCSD solver.
Public descriptors still reject DF; native DF triples, Lambda and forces are
not registered by this entry.
The value source accepts orbital shells through f and auxiliary shells through
g. The shared capability check runs before RHF. Existing through-f values use
Rys quadrature; g auxiliary values use compiler-generated Gaussian moment
polynomials with F0–F10 from the shared FP64 Boys evaluator. The generic source
policy reports `generated_rys_auxiliary_g_polynomial` when g is present; other
math policies are rejected for such an owner. Raw/transformed three-center and
metric derivative requests on this owner are rejected before launch. Legacy
integral exporters and public derivative capabilities retain their f limit.

The DF metadata packer skips SCF warm densities and pair/quartet task tables.
It uses Cartesian metadata plus a separate six-term public g expansion, leaving
the legacy three-term SCF topology unchanged. This is an internal value-domain
extension; hundreds-AO molecular endpoints still require independent
conditioning, energy, residual and amplitude qualification.

The source reuses the generated CUDA three-center/metric evaluator and the
shared cuSOLVER symmetric inverse-root owner, with an explicit relative cutoff.
The internal molecular composition fixes that cutoff at `1e-10`. Each complete
AO row is generated once. The compiler-owned `method/df_mo_source.py` traversal
performs two orbital projections and metric whitening, with
`2 N^3 Q + N^2 Q^2` contraction summands and `N + 2` GEMMs. The source generates
`N^2 Q` raw values; it does not regenerate them for each auxiliary consumer.

`cc/df_source.py` defines independent `oo/ov/vo/vv` factor selection and the
five retained Gram products. `tools/generate_df_cc_source.py` emits the packing
actions, block BLAS traversal, equation hashes and phase capacity queries.
Selection does not assume MO-pair symmetry. Only `B_ov`, `B_vv` and the five
retained integral blocks cross the explicit host-input solver boundary; all
integral generation, whitening, MO transformation and block contraction use
CUDA. No CPU numerical retry is provided.

One stream and one raw row buffer preserve source lifetime through each
projection. Transform temporaries are released before packing; the full MO
factor tensor is released before block scratch is allocated. Failure drains
work before owners are destroyed and publishes no partial result. Admission
includes the caller/reference, source setup, metric owner, transform/packing
arenas and host outputs. The existing source factory reports setup capacity
after construction, so that phase is checked and released before downstream
allocation/publication. Device capacity includes the shared owner's lazy SCF
reservations: it is a conservative bound, not a measured physical peak.

`tests/python/test_df_cc_molecular_source.py` is enabled with
`GENERATIVEQC_DF_CC_SOURCE_CUDA_TEST=1` inside a finite Slurm GPU allocation.
It checks factors/blocks against committed independent raw integrals and live
PySCF g-auxiliary fixtures, including duplicated auxiliary shells and truncated metric rank, and exercises exact
budget admission and transactional failure. Its two-electron molecular CCSD
case uses an independent determinant-space energy oracle. This small-source
qualification does not establish a complete hundreds-AO CCSD(T)/force endpoint.
`tests/python/test_df_auxiliary_g.py` uses `GENERATIVEQC_DF_G_CUDA_TEST=1` for
its CUDA parameter. `tools/validate_df_source.py --cases g-cartesian g-spherical
g-dependent g-cartesian-spherical` checks raw/metric/J/K values, mapping and
tile choices, and derivative rejection using `tests/native/df_value_probe.cpp`.
All GPU runs require a finite Slurm allocation. Large-source conditioning
limitations are retained in the [auxiliary-g decision](../../.agents/notes/implemented/numerics/2026-10-03-df-auxiliary-g-values.md).

See the [native solver decision](../../.agents/notes/implemented/architecture/2026-10-03-df-cc-native-solver.md)
for ownership and auxiliary-work rationale.

## Validation rules

Implementation tolerances and DF fitting error are separate quantities.

1. Raw/whitened B and reconstructed `g_DF` are checked for finite FP64 data
   and identity compatibility.
2. The dense oracle and any factorized path must agree tightly for fixed
   amplitudes, converged RCCSD energy, and (T) energy on the same DF
   Hamiltonian.
3. Auxiliary-gauge-equivalent B tensors must give the same `g_DF` and energy.
4. DF-versus-conventional-four-center differences are reported separately by
   `df_fitting_error`; they are not used to relax same-Hamiltonian numerical
   gates.
5. The conventional RHF Fock/orbital state is preserved and is independently
   identifiable from the correlation DF Hamiltonian.

## Python validation API

The internal validation helpers live in
`tools.generativeqc_cc.df_ccsdt_oracle`:

- `correlation_df_reference` fixes the method contract while preserving the
  conventional RHF state;
- `prepare_same_hamiltonian_dense_oracle` obtains B from the existing
  `DFProvider` and reconstructs the small dense `g_DF`;
- `dense_df_oracle_from_three_index` allows independent/synthetic B fixtures;
- `run_dense_df_ccsdt_oracle` evaluates the trusted RCCSD/(T) equations;
- `df_fitting_error` reports DF-versus-exact integral error separately.

Slice B additionally exposes the internal `solve_df_ccsd` / `PreparedDFCCSD`
validation path. C2a adds `df_rccsd_t_method_capabilities` and
`df_rccsd_t_energy` as the executable energy-only source facade. The latter
requires a live `NativeSource` with an explicit auxiliary basis and records the
conventional-reference/correlation-DF split in every successful result. It does
not register a Calculator method.

## Non-goals after slice B

- no production full-`NMO^4` DF integral storage;
- no native Calculator DF-CCSD(T) method registration in C2a;
- no complete DF-CCSD(T) force or gradient claim (tracked by #158; the
  reusable B-to-A/M reverse edge is documented in [df_ccsdt_gradient.md](df_ccsdt_gradient.md));
- no frozen-core, open-shell, ECP, local, or DLPNO variant.
