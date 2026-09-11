# Arbitrary weighted ERI derivatives

The external-weight consumer computes the derivative of
`sum W[i,j,k,l] (ij|kl)` for fixed, arbitrary real weights. A correlated
cotangent need not factor into HF densities. Weight response belongs to the
caller's Lagrangian/response calculation. The native API never creates a
density and performs no magnitude screening; it skips exact zeros only.

## Execution-path inventory

This inventory follows the actual dispatch in `src/scf/cuda_rhf.cu`, the
architecture registry, and the force queue implementations. Mathematical
source emission alone does not select a production route.

| Consumer/class | Existing execution | Selection and this change |
| --- | --- | --- |
| HF, at most 16 public AOs | Persistent ERI values and their native force path | AO cutoff; does not exercise the direct psss migration |
| Direct HF psss Fock | Fixed queues use handwritten three-component `contracted_eri_cartesian_source_psss`; bounded streaming can use the existing generated Fock row | `kFixedTopologyGeneratedFockExclusionMask` retains the fixed low-order workers; streaming capabilities select generated psss for bounded Fock |
| Direct HF psss force | Handwritten weighted PA/PQ dot products, F0/F1/F2 once per primitive pair product | Default remains handwritten; `VIBEQC_PSSS_WEIGHTED=generated` substitutes only its primitive weighted expression |
| Direct HF psss resident force | Resident bra pairs and primitive-length descriptors | `VIBEQC_PSSS_RESIDENT_BRA`; the generated expression uses the same resident pairs and canonical-orientation scales |
| Bounded direct psss force | Lossless paged exact-class consumer | `VIBEQC_BOUNDED_DIRECT_STREAMING=force` or topology limits; uses the same weighted expression, retains page traversal |
| Direct HF ssss force | Handwritten low-order expression | Remains a future migration candidate; generated Fock support is a separate consumer |
| Direct HF psps/ppss | Common generated production kernels | Already migrated before this issue; manifest, signature policies and queues are retained |
| Other direct HF classes | Manifest-selected generated kernels plus validated native fallbacks | Registry selects by architecture, consumer and class; no manifest rows added here |
| External psss weights | Generated precontracted Hermite DAG | One primitive record carries all x/y/z weights; optional independent fallback |
| External unmigrated s/p/d/f weights | Unscreened Hermite/Dual3 primitive fallback | One explicit component per record, with the caller's actual external weight |

The existing [component ledger](../benchmarks/results/rtx5090-0b6a573-issue-41-current-head-component-ledger.json)
measured psss as the largest exact force class, 137.054 ms per replay, on its
384-AO workload. That measured hotspot motivates this slice. Its timings
are historical and are not claimed as current candidate results.

## Weight and permutation contracts

`python/vibeqc_compiler/integral/eri_weights.py` exposes three explicit adapters:

- `fold_dense_eri_weight` sums every *distinct* ordered weight in the eightfold
  ERI orbit. Same AO indices and same pairs reduce the orbit size. Multiplying
  an arbitrary single weight by eight is invalid.
- `fold_normalized_pair_weight` consumes a full, possibly nonsymmetric matrix
  `Z[I,J]` over triangular pairs. With `s[I]=1` for a diagonal AO pair and
  `sqrt(2)` otherwise, `T[I,J]=s[I]*s[J]*(ij|kl)`. An off-diagonal pair-pair
  contributes `s[I]*s[J]*(Z[I,J]+Z[J,I])`; a diagonal contributes once.
- `hf_eri_weight` is an optional explicit RHF/UHF adapter. Generic execution
  never calls it implicitly.

Orbit folding is directly valid for physical-atom derivatives. Reordering a
raw shell-center derivative also requires reordering the center slots. Four
shell slots stay independent through differentiation and translation recovery,
even when they belong to one physical atom. The response adapter then adds
their derivatives to the requested, sorted physical-atom rows.

## Generator and native boundary

`build_weighted_eri_ir` describes external weights without a direct-HF consumer.
`build_weighted_eri_kernel` shares the existing pair expansions and Coulomb
recurrences, precontracts the Hermite coefficients, factors the scalar, and
then differentiates it with explicit geometry leaf responses. This preserves
psss's combined weighted PA/PQ sums instead of differentiating and cloning
three component recurrences. A lowering accepts at most 64 explicitly selected
Cartesian components; larger classes must be partitioned without truncation.

`prepare_weighted_eri_stream` consumes a `BlockRequest` and a bounded weight
provider callback. Full, padded and partial public layouts are supported. For
spherical shells, callers supply `(ncart,npublic)` projection matrices; weights
are pulled back on the host before streaming. Input primitive coefficients
include radial normalization, and Cartesian angular factors are applied once.
The provider is called once and its numerical weights are frozen. There is no
primitive-product array. A new geometry needs a newly prepared stream.

The internal C++ entry point is
`vibeqc::scf::contract_cuda_weighted_eri_primitives`, declared in
`src/scf/cuda_weighted_eri.hpp`. Link against a CUDA-enabled VibeQC library.
It accepts a bounded span of 208-byte primitive records and returns one
104-byte result per output tile: the explicitly scaled weighted scalar and
four three-component derivatives. Callers can stream multiple spans and add
the returned contributions; each call initializes its own output to zero.
This is an internal correlated-gradient integration boundary, not a new
Calculator method or a complete correlated-gradient implementation.

The input kind identifies either one explicit s/p/d/f Cartesian component or
the full canonical psss x/y/z weight vector. `generated=false` evaluates both
kinds through the independent native fallback. Unit-weight component records
provide raw diagnostics without a full derivative tensor. Nonfinite inputs,
unsupported angular momenta, invalid output indices, and insufficient numeric
budgets fail explicitly. CPU-only builds return `NOT_IMPLEMENTED`.

Capability queries use `backend="cuda_weighted_eri"` for this boundary.
Legacy `backend="cuda"` still rejects arbitrary weights, so an HF emitter
cannot accidentally accept them. Shell slots and operator centers must both
follow `(0,1,2,3)`; physical atom indices may repeat.

## Memory, placement and screening

The host adapter checks its conservative numeric peak before copying primitive
lists, invoking the provider or allocating the full bounded public tile. It
counts weight storage, projection matrices, pullback intermediates, primitive
lists, angular/weight metadata and a yielded record. Python object overhead
and caller-owned caches outside those inputs are excluded.

The native budget separately includes returned host results, device results,
and one reused device upload buffer. Capacity changes only chunk boundaries;
all records are consumed. Caller-owned record spans, CUDA runtime allocations
and implicit kernel stacks are excluded and must be budgeted separately.
The generic fallback has larger implicit stacks than the generated psss path.
Neither diagnostic is a process-RSS or total-GPU-memory guarantee.

No density-derived screen applies to external weights. Introducing a nonzero
threshold would require a conservative bound on the external weights and
derivative integrals, including an accumulated error budget. Existing HF
screening and the distinct Fock/force task domains remain in their own callers.

## Validation and promotion

The independent small-block checks cover all AO equality patterns, arbitrary
nonsymmetric weights, normalized pairs, raw/fused derivatives, coincident
centers, public spherical transforms, and three finite-difference step sizes.
Only representative f shells are used here; the full f matrix belongs to #135.

Run `tools/screen_weighted_eri.py` to compile the materialized and inline-single-use
candidates. Both consume all runtime inputs and write all thirteen outputs.
The native numerical driver is `tools/validate_weighted_eri.py`, paired with
the manual CMake target `vibeqc_weighted_eri_probe`. It checks mixed output tiles,
both execution routes, multiple upload capacities, raw unit weights, repeated
geometries and libcint center derivatives.

`tools/validate_weighted_eri_endpoints.py` compares complete RHF/UHF cluster
energies and forces with fresh plans, fixed/resident/paged scheduling, mixed
batches, cold and warm phases, and explicit changed-geometry replays. Separate
fixed-queue profile runs require nonzero psss primitive work. Native class
counters do not cover paged queues; their endpoints are compared separately. Its optional baseline library
checks the previous binary as well as the retained expression in this binary.

Every numerical GPU run, profiler, sanitizer and endpoint benchmark must use
the Slurm `main` partition, `--gres=gpu:5090:1`, and a finite `--time`.
The generated psss policy remains opt-in. A resource-safe or mathematically
correct candidate is promoted only after non-regressing molecular endpoint
evidence; the old component-cloning candidate and its rejection remain
documented in [shell_codegen.md](shell_codegen.md).

The final [RTX 5090 evidence](../benchmarks/results/weighted-eri-144/README.md)
includes native libcint agreement, memory-sanitizer results, resource records,
and 72 complete RHF/UHF endpoint runs. The new candidate remains opt-in because
its endpoint time ratios cross one.
