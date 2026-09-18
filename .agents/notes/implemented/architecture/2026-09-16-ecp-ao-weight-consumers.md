# Decision: Generate ECP contracted AO and fixed-weight consumers

Status: implemented
Date: 2026-09-16

## Problem

After #371, the CUDA adapter still defined the component/primitive contraction
and density-weighted energy derivative. These scientific operations were outside
the compiler-owned ECP specification even though their inputs were generated.

## Decision

Lower node displacement, contraction coefficients, local/nonlocal operator
addition and fixed-weight derivative products through the existing scalar Graph.
Emit the component/primitive and full AO reductions alongside the projector
lowering. Native kernels retain indexing, launches, buffer ownership and scatter.
The generated consumer takes arbitrary real AO weights; RHF/UHF density selection
remains method policy in the native caller.

## Rejected alternatives

Do not move the CPU ECP implementation onto generated arithmetic: it must remain
an independent numerical oracle. Do not fuse derivatives into density contraction
in this migration: that changes derivative-buffer lifetimes, raw export behavior
and resource planning, requiring a separate scheduling qualification.

## Invariants

Preserve FP64 storage, component-outer/primitive-inner reduction order, normalized
coefficients, primitive offsets, value-only zero derivative slots and full AO
weight semantics without an occupancy multiplier. Forces are minus the energy
derivative. Keep the one-radial-shell schedule, physical center mapping and
coarse/refined grid gate. No broader ECP or DFT capability is implied.

## Evidence

The native projector test also checks contracted s/p/d values against an
independent long-double factored polynomial/Gaussian oracle, basis-center
derivatives at two finite-difference steps, mixed signed components/primitive
coefficients, coincident nodes, primitive offsets and value-only calls.
Nonsymmetric fixed weights check local/nonlocal force signs against displaced
energy contractions. Existing CPU/Libcint and CUDA complete RHF/UHF tests remain
the integration gates. [Source-bound qualification](../../../../benchmarks/results/ecp-consumers-171/README.md)
records 30 CPU native tests, 43 CPU Python passes with 10 GPU skips, 53 CUDA
Python passes, two CUDA native passes and zero memcheck errors. Complete RHF/UHF
timing and planned peaks match the baseline within this small regression check;
the AO kernel trades eight extra registers for 112 fewer stack bytes. No
statistical speedup or schedule-promotion claim is made.

## Consequences and revisit conditions

The generated header grows while native scientific bodies shrink. Host
normalization metadata, quadrature/harmonic construction and convergence policy
remain explicit and the mixed CUDA adapter stays conservatively scientific in
the ownership ledger. Revisit fusion only with independent full-force and
resource evidence and a demonstrated complete-endpoint need.

The host normalization/grid/harmonic boundary is subsequently superseded by
the [2026-09-17 host-grid decision](2026-09-17-ecp-host-grid.md). This note retains
the historical AO/weight migration rationale and its original measurements.

## References

- Issue #171; PR #371.
- [Current ECP contract](../../../../docs/ecp.md).
- `tests/native/test_ecp_projector.cpp`, `tests/python/test_ecp.py`.
