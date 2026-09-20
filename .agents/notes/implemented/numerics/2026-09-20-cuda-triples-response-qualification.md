# Qualification: bounded generated CUDA triples response

Status: real-device numerical qualification completed for PR #710's bounded slice
Date: 2026-09-20

## Scope and ownership

The compiler owns the standard-(T) tile VJP. The runtime owns one resident tile
at a time and adds its prefix-overlapping cotangents on the host. Corrected
Lambda uses the existing resident CCSD transpose-Jacobian action and its
independently expanded residual gate. This does not register public forces,
orbital response, a full T3 tensor, or a CPU fallback for CUDA execution.

## Review repair

Demand-driven VJPs may eliminate primal inputs. The repair at 1acbd440 binds
uploads to the generated live input set instead of weakening the resident ABI.
The strict fake-owner regressions reject extra/missing uploads for individual
t1/fov responses. The real-device test now compares all eight returned blocks
against the CPU VJP, not only the t1/t2 blocks used by corrected Lambda and
nonzero denominator-response sanity checks.

## Executed evidence

On node3, finite Slurm jobs 10487 and 10490 used the allocated 5090 GPU, CUDA
12.9.86, sm_120, and the source-verified 8ba30c62 implementation. The second
run includes the additional all-eight-block oracle assertions in this change.
Both runs passed all six selected tests without skips: five host contract/math
cases and one actual CUDA water triples-response/corrected-Lambda test.
The final run completed in 84.21 seconds using verified compiler-cache artifacts.
This timing is not a benchmark or a speedup claim.

The device test checks all t1/t2/ovvv/ovoo/ovov/fov/eps_o/eps_v cotangents,
corrected Lambda against the CPU equations, independent Lambda residual norms,
nonzero denominator responses, runtime-device reporting, and the per-tile byte
bound. The reverse-tile and persistent Lambda owners do not overlap.

The test environment used Python 3.11.14, NumPy 2.4.6 and PySCF 2.14.0.
CPU reference setup reused the existing compatible native AO/SCF library;
the CUDA response/Lambda artifacts were generated and identity-checked from
this source tree. Initial setup attempts lacked a development typing dependency
or an explicit nvcc path; those were corrected before the passing device runs.
No numerical tolerance was loosened and no assertion was skipped to pass.

## Admission boundary

This qualifies the bounded response slice, not unrestricted size/domain or
endpoint performance. Repository final-head integration CI remains mandatory.
Nuclear/orbital assembly and public Calculator force admission remain separate.

Refs #154, #155; PR #710.

Agent: ChatGPT
Model: GPT-6 Astra Pro
