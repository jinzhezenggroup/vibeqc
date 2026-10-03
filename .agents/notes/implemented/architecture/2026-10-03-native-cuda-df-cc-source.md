# Decision: compose the native CUDA DF source with conventional-reference CCSD

Status: implemented
Date: 2026-10-03

## Problem

The factorized native solver can converge supplied 230/264-AO Hamiltonians,
but supplied factors do not establish a molecular endpoint. Its source must
generate the same correlation-only DF Hamiltonian without a CPU/PySCF numerical
dependency or regenerating the complete source for every auxiliary slice.

## Decision

Reuse the generated CUDA DF integral source and shared cuSOLVER metric owner.
The staged compiler traversal from #1771 generates each AO row once, applies
two orbital projections, then whitens in the MO basis. The transform costs
`2 N^3 Q + N^2 Q^2` scalar contraction summands. Moving the metric operation
does not change the symmetric inverse-root convention or thresholded space.

Define factor-sector selection and the five retained block Gram products in
TensorIR. Emit dynamic CPU/CUDA packing and direct BLAS traversal through the
existing native CC generator and tensor GEMM contract. Keep `ov` and `vo`
independent during packing, so no symmetry shortcut can conceal an axis error.

The existing solver explicitly owns host factor/block inputs. Download only
`B_ov`, `B_vv`, `ovov`, `ovvo`, `oovv`, `ovoo`, and `oooo` at that boundary.
The source does not allocate `ovvv` or `vvvv`. Downloading inputs is an ownership
boundary, not a CPU contraction or a reference-oracle retry.

An optional auxiliary system on the internal native method entry composes
conventional FP64 CUDA RHF, source preparation and native DF CCSD. It retains
the physical conventional Fock and denominator policy. The composition uses
relative metric cutoff `1e-10`; the lower-level source takes it explicitly.
Public DF descriptors remain rejected until the remaining energy/response
owners are qualified.

## Resource and failure invariants

- One stream and one row buffer ensure source work completes before reuse.
  Temporary orbital projections die before factor packing; full MO factors
  die before retained-block scratch allocation.
- Buffers are declared after the metric owner, so exception unwinding drains
  their stream before the source/BLAS/stream owner dies. No partial result is
  returned on invalid data, source failure or insufficient budget.
- Admission covers reference vectors, caller state, basis metadata, source
  setup, metric factorization, phase arenas, and published host outputs. The
  auxiliary system remains charged during reference and solver phases.
- The shared integral-source API exposes setup capacity after construction.
  Check and release it before downstream allocation/publication, following the
  existing DF bridge contract; do not claim a preallocation bound for that API.
- Shared metric diagnostics reserve lazy SCF storage even though this consumer
  does not run that SCF owner. Numeric/device capacities are conservative
  reservations, not measured physical peaks. Count explicit coefficient and
  factor/block transfers separately from metric staging and scalar audits.

## Rejected alternatives

Reusing the RI-MP2 consumer's complete AO whitening schedule would repeat
expensive source work for the wrong consumer loop. Reimplementing a CC-specific
metric eigensolve would duplicate rank/threshold semantics. A CPU factor source
would defeat native molecular qualification. None is an acceptable shortcut.

Do not combine this source qualification with unmeasured mixed precision or
double buffering. Issues #1763, #1764 and #1765 require compiler-owned schedules,
independent numerical gates and complete endpoint evidence. Keep the present
FP64, single-buffer path as the baseline for those experiments.

## Evidence and acceptance gates

The source test reconstructs full factors and all retained blocks from committed
independent raw integrals and metrics for H2, water, LiH and an f-shell case.
Duplicated auxiliary s shells give `Q != N` and a rank-deficient metric. Compare
every value at absolute/relative tolerance `3e-10`, check semantic work and
transfer counts, accept the reported budget exactly, and reject one byte less
without changing previously published output.

The complete H2 molecular composition is compared with the lowest eigenvalue
of an independent determinant Hamiltonian: two-electron CCSD is exact in that
space. Require correlation-energy error below `3e-9` Eh and freshly expanded
singles/doubles residuals at most `1e-10`, including the rank-deficient case.
This validates the composition, not hundreds-AO native CCSD(T) or forces.

The RTX 5090 qualification passes all ten GPU cases and unfiltered memcheck
reports zero errors. Maximum source/block error is below `2.84e-13`; H2
correlation-energy errors are below `8e-16` Eh. Seventy-three host/compiler/
admission tests and 45 conventional public/force-resource tests pass. Three
live PySCF analytic-oracle tests skip because that optional dependency is
absent. See `benchmarks/results/df-cc-cuda-source-20261003/qualification.json`
for source/binary identity, per-case work, capacity, errors and timing scope.

This qualification disables AOT shells. Loading the full CUDA library exposed
a missing `preferred_streaming_fock_shell_class_mask` definition in the generic
registry stub. Returning an empty mask restores the declared no-AOT fallback;
no scientific routing override or weaker numerical gate is required.

## Revisit when

The full molecular 230/264-AO source/CCSD path is qualified, then compose native
factorized triples, Lambda/orbital/factor/metric response and nuclear forces.
Consider direct device factor handoff when the solver has an explicit borrowed
input contract and independent admission/lifetime tests. Consider overlap only
when larger endpoint timings identify producer/transfer latency as material.

## References

- `docs/developer/df_ccsdt.md`
- `tests/python/test_df_cc_source_blocks.py`
- `tests/python/test_df_cc_molecular_source.py`
- #1763, #1764, #1765, #1769, #1771
