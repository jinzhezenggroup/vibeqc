# Decision: specialize weighted-ERI CUDA fallback by actual quartet angular order

Status: implemented
Date: 2026-09-19

## Problem

The generic external-weight ERI CUDA consumer used one
`primitive_eri_cartesian<12>` Dual3 instantiation for every reference record,
including ssss. On an RTX 5090 (sm_120), even a single ssss shell-quartet call
failed at kernel execution with CUDA `out of memory`. Increasing the explicit
stage budget from 32 MiB through 256 MiB did not change the failure, showing that
the requested record/result arena was not the limiting allocation. The maximum
ffff recurrence instead imposed the order-12 local/thread state on every launch.

## Decision

Use the same invariant already used by direct J/K: the sum of the four shell
angular momenta determines the maximum Hermite/Coulomb order for every Cartesian
component of that quartet. Instantiate reference weighted-ERI kernels separately
for orders 0 through 12, detect the orders present in each uploaded chunk on the
host, and launch only those specializations. Each specialization ignores records
of other orders. High-order Dual3 kernels also reduce block width progressively
(64 lanes through order 4 down to one lane for orders 11-12), so mostly-inactive
lanes do not reserve ffff recurrence state. Generated psss records retain their
existing dedicated kernel.

Keep the public record ABI, weight convention, output layout, chunking, memory
budget and independent Dual3 reference mathematics unchanged. Improve the CUDA
failure detail to retain `cudaGetErrorString` instead of replacing it with an
undifferentiated contraction failure.

## Rejected alternatives

- Raising the explicit stage budget: 32, 48, 64, 96, 128, 192 and 256 MiB all
  failed before specialization; this was not the bounded record/result arena.
- Using the generated psss recurrence for every shell class: only psss has that
  generated callable at this boundary; changing scientific ownership would be a
  separate migration.
- Keeping one order-12 kernel and changing block size: lower occupancy does not
  remove the unnecessarily large per-thread recurrence state and would leave
  low-order work coupled to the ffff implementation.
- Silently falling back to CPU: that would invalidate requested CUDA execution
  and resource evidence.

## Invariants

A record's per-slot Cartesian powers are validated first, so the host-computed
order is in [0,12]. The specialization changes only compile-time storage bounds;
the primitive inputs, Dual3 nuclear derivative, contraction coefficient, public
AO normalization, four independent shell-center outputs, and dense Frobenius
weight convention are identical. Mixed-order chunks may launch several kernels,
but each record contributes in exactly one reference specialization. Generated
psss records remain excluded from the reference kernel when `generated=true`.

## Evidence

On node3, CUDA 12.9.86, RTX 5090 (compute capability 12.0), driver 580.95.05:

- before specialization, ssss and H2/H2O weighted-ERI calls failed with CUDA
  `out of memory`, independent of stage budgets through 256 MiB;
- after specialization, ssss, full H2 and full H2O arbitrary-weight contractions
  executed successfully;
- a nonzero two-center sparse ffff order-12 derivative executes and agrees with
  an independent CPU integral finite difference at steps 1e-4, 3e-5 and 1e-5
  bohr (analytic atom-0 x derivative -0.0241199081939 Eh/bohr);
- arbitrary H2O S/T/V + ERI CUDA contractions agree with the independent dense
  CPU derivative oracle;
- complete H2, H2O, NH3, CH4, Cartesian-d H2 and spherical-d H2 CCSD CUDA
  derivative endpoints meet their retained independent <=1e-6 Eh/bohr analytic
  reference gate;
- dense-vs-shell ERI-weight CCSD endpoint parity passes on H2;
- an impossible one-byte derivative stage budget fails without a CPU derivative
  fallback.

Exact source hashes and final integrated test totals belong in the PR evidence;
these observations authorize no whole-process GPU memory or speedup claim.

## Consequences

Low-order external-weight ERI derivatives no longer inherit ffff thread-local
storage, and high-order sparse launches no longer reserve that state for 64
mostly-inactive lanes. A mixed-order primitive chunk can scan the same records
once for each order present, so this is a correctness/resource fix rather than a
claim of the best launch schedule. Current molecular bridges naturally flush within one shell
quartet, where total angular order is invariant.

## Revisit when

Generated weighted derivative callables cover every production shell class or a
source-owned queue groups records by angular class before upload. Either can
remove the residual mixed-order scanning while preserving this specialization
invariant and the independent reference path.

## References

#144, #153, #532; `src/scf/cuda/weighted_eri_kernels.cu`,
`src/scf/cuda_rhf.cpp`, `tests/python/test_cc_complete_gradient_cuda.py`.

Agent: ChatGPT
Model: GPT-5.6 Sol
