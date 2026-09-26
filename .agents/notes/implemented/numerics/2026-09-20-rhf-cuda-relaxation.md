# Generated CUDA RHF relaxation contraction

## Decision

Issue #180 B1-CUDA/relaxation reuses a method-independent generated
first-integral gradient consumer rather than adding a handwritten RHF force or
Hessian kernel. The generic compiler accepts one or two external AO matrix
weights per term and emits the corresponding primitive-gradient pullback into a
Cartesian atomic vector.

RHF declares only these weights:

- kinetic and nuclear attraction: D1_ij;
- overlap/Pulay: -W1_ij;
- conventional ERI response:
  1/2 D1_uv P_wx + 1/2 P_uv D1_wx
  - 1/4 D1_uw P_vx - 1/4 P_uw D1_vx.

The CUDA runtime owns bounded storage, transport, generated-program dispatch,
atomic output accumulation and failure-atomic publication. It contains no
RHF/Hessian coefficient equation and is classified as runtime in the CUDA
ownership ledger.

## Execution boundary

The selected CUDA relaxation path uploads D1, W1 and P0 once per directional
contraction. Primitive derivatives, Cartesian normalization and AO-weight
products execute on device. Only the final (natoms,3) result is downloaded;
raw derivative tiles and intermediate AO matrices are never downloaded.

CPU remains the default and no silent fallback is allowed. The complete HVP is
still mixed execution: D1/W1 are reconstructed on the host before this upload,
#178 second-integral weighted HVP consumers and final molecular assembly remain
host-side, and the public Calculator Hessian API remains closed.

Block/full-Hessian memory accounting treats the relaxation arena as a separate
phase peak in addition to the existing response-solver workspace. An
insufficient relaxation budget fails before response/provider work.

## Validation

On the rebased implementation branch based on master commit
d32caac5fc14810188e5fa33586f1b6cb51078b5:

- CPU Release native CTest: 41/41 passed.
- CPU Release focused Hessian suite: 45 passed.
- Generic compiler-contract tests: 5 passed.
- CUDA relaxation budget/preflight plus compiler contracts: 6 passed.
- RTX 5090, CUDA 12.9, sm_120: 4 CUDA relaxation/HVP/block/full-Hessian tests
  passed, including independent CPU relaxation equality and a test that forbids
  CPU relaxation substitution.
- CUDA 12.9 Compute Sanitizer memcheck on the numerical relaxation case:
  application passed and ERROR SUMMARY reported 0 errors.
- Compiler dependency structure: 233 modules, 0 errors.
- Shared SCF structure: 221 modules, 0 errors.
- Evidence retention checks passed.
- CUDA ownership inventory passed after classifying the new runtime header.

The system CUDA 12.4 sanitizer/compiler cannot qualify sm_120 and is not counted
as evidence; qualification uses the installed CUDA 12.9 toolchain.

## Follow-on

The next #180 GPU slice is not another first-relaxation formula. It should move
the remaining #178 second-integral weighted-HVP consumers and, separately, the
host reconstruction/upload seam if an all-device HVP is required. Production
size, public Calculator Hessian/HVP endpoints and DFT Hessians remain
independent gates.

Refs: #178, #179, #180, #564, #568, #574.

Agent: ChatGPT
Model: GPT-5.6 Sol
