# CUDA second-integral RHF HVP consumer

## Decision

Issue #180 now reuses #178's already-generated CPU/CUDA second-derivative provider
for the frozen conventional-RHF skeleton instead of keeping the molecular HVP
consumer hard-wired to a CPU C++ compiler.

The public tools choice is explicit:

- `second_backend="cpu"` remains the default;
- `second_backend="cuda"` requires an explicit `CudaCompilerAdapter`;
- there is no CPU/oracle fallback for a CUDA-selected action;
- scalar HVP, multi-RHS HVP and bounded full-Hessian assembly all propagate the
  same provider selection and program identity.

No RHF Hessian formula was added to CUDA. The existing #178 generated
`weighted_hvp` programs still own S/T/V/four-center second-integral arithmetic.

## Execution boundary

For a CUDA-selected second-integral HVP, the consumer streams bounded shell
primitive/fixed-weight records into `PreparedSecondDerivative`. Primitive
second derivatives and fixed-weight contractions execute on device. Only the
contracted coordinate HVP tile is returned to host; raw integral Hessians and
intermediate AO matrices are not published.

Diagnostics retain the generated program/native-artifact identities, primitive
record and batch counts, host/device numeric peaks and CUDA input/kernel/output
timings. Multi-RHS/full-Hessian diagnostics additionally record the
second-integral phase peak alongside response and relaxation phases.

This is still mixed execution. Response reconstruction currently publishes
D1/W1 on host before optional CUDA relaxation, final molecular assembly remains
host-side, and the full-Hessian output is host-owned. Production-size and public
Calculator capability are not inferred.

## Validation

Implementation code through `5210682e1dd2e8ba8ee42070f9c60277d52874a9`
(the later branch change before GPU qualification was documentation only):

- Ruff check passed.
- Ruff format check passed.
- Focused CPU HVP/block suite: **16 passed**.
- Opt-in tests without a GPU allocation: **16 passed, 4 skipped**.
- RTX 5090, CUDA 12.9, `sm_120`, Slurm job 10290:
  **4/4 passed** for direct CPU/CUDA component equality, complete scalar HVP
  with CPU-compiler substitution forbidden, multi-RHS HVP, and bounded full
  Hessian.
- The CUDA integration tests require positive packed-record uploads and
  contracted-tile downloads while asserting zero raw-Hessian and
  intermediate-matrix downloads.
- A fresh Compute Sanitizer run was **not** claimed: the node currently exposes
  only CUDA 12.4 `compute-sanitizer`, which cannot qualify the RTX 5090
  `sm_120` path, while the local CUDA 12.9 toolkit tree used for compilation
  does not contain a sanitizer executable.

GitHub pre-commit for PR #617 passed. Remaining repository workflows are allowed
to complete independently; this note makes no release or performance claim.

## Follow-on

The next RHF GPU work is no longer second-integral arithmetic. The remaining
all-device seam is the host publication/reconstruction around D1/W1 and the
final molecular HVP/Hessian assembly. Separately, public production-size
Calculator HVP/Hessian capability remains gated.

For the method roadmap, the next scientific expansion is LDA/PBE RKS/UKS
Hessian/HVP: bind real native CPKS state through #179 and add the required
second grid/partition geometric response before extending hybrids, meta-GGA,
RSH, DF or ECP variants.

Refs: #178, #179, #180, #564, #568, #574, #590, #617.

Agent: ChatGPT
Model: GPT-5.6 Sol
