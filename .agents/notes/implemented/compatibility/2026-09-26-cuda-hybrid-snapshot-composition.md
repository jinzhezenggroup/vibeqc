# Decision: Preserve CUDA hybrid composition in final-state snapshots

Status: implemented
Date: 2026-09-26

## Problem

Public CUDA PBE0 admission exposed a mismatch between the SCF owner and its
snapshot: the final-state identity retained default unit XC scales, while the
writer appended a composition suffix under a CUDA wire version that the reader
interpreted as having no suffix.

## Decision

Copy the actual semilocal scales into the final-state identity. Use CUDA wire
versions 8/9 for composition records, with version 9 also carrying ECP data.
CPU versions 6/7 and pure CUDA versions 3/5 retain their existing layouts.
Select the wire format from the presence of composition data, independently of
the functional's scientific domain version.

## Invariants

Writer and reader must agree on grid, transfer, ECP, and composition suffixes.
Exported scales and exact-exchange coefficients must match the calculator's
resolved MethodIR composition. This transport fix does not authorize additional
force or response methods.

## Evidence

`tests/python/test_cuda_hybrid_snapshot.py` compares CPU/CUDA converged PBE0
RKS/UKS exports, including the native writer, Python reader, and D/F matrices.
Run it through Slurm with `VIBEQC_RESOURCE_CUDA_TEST=1`.
