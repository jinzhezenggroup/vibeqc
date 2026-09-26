# Decision: qualify complete D4 EEQ response on CPU and CUDA

Status: implemented
Date: 2026-09-19

## Problem

The migrated #498 baseline had correct fixed-charge GFN2 D4 CPU/CUDA
mathematics, but generic DFT-D4 requires the upstream EEQ reference model,
the EEQ2019 charge solve, and the complete coordinate response dq/dR.
Canonical r2SCAN-3c additionally changes the D4 zeta profile from 3/2 to 2/1.

## Decision

Retain the fixed-charge D4 evaluator as a common qualification primitive and
add a pinned EEQ2019 provider with differentiated constrained linear solve.
Generate independent EEQ reference/C6 tables for standard D4 and r2SCAN-3c,
then form the complete derivative as

  dE/dR = (partial E/partial R)_q + (dq/dR)^T (partial E/partial q).

The same bounded scalar mathematics is host/device callable. CUDA uses one
worker per molecule for correctness qualification only; this is not a
performance promotion or permission for permanent per-functional kernels.

## Evidence and boundaries

Independent DFT-D4 4.2.0 fixtures cover standard PBE, r2SCAN-3c, asymmetric,
neutral and charged Zn-containing molecules. Multi-step finite differences
validate dq/dR and the complete gradient. RTX 5090 sm_120 tests exercise
complete EEQ-D4, mixed profiles, ragged members and failure isolation;
Compute Sanitizer memcheck reports zero errors.

MethodIR owns an explicit D4Spec and exact r2SCAN-3c D4 manifest identity.
No public Calculator method is registered. Production runtime/provider
integration, generated derivative execution, scalable CUDA scheduling, PBC,
Hessians, and the remaining electronic/basis/gCP r2SCAN-3c pieces remain open.

Agent: ChatGPT
Model: GPT-5.6 Sol
