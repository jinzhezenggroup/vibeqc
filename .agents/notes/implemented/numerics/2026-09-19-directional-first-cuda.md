# Decision: generated CUDA directional first-integral matrix accumulation

Status: implemented
Date: 2026-09-19

## Problem

#526 implemented shell-local directional H1/S1 and one shared nuclear response
solve, but the first-integral and direction/density contractions remained on
the CPU. #511's CUDA J/K action alone could not make this source stage a CUDA
implementation. Copying every raw derivative or per-shell matrix back to the
host would preserve that gap while disguising the data movement.

## Decision

Introduce a method-neutral `DirectionalMatrixTerm` contract with explicit AO
output slots, optional external-matrix weight slots and signed coefficients.
The compiler reuses the existing generated S/T/V and weighted-ERI primitive
DAGs, applies direction and Cartesian normalization, then emits the declared
weight/matrix contractions. The RHF consumer shares its declared conventional
J/K terms with the existing CPU first-order traversal. No HF equation or
shell-specific integral recurrence is added to native runtime code.

The new native runtime owns one common device accumulator across compiled
shell/component programs. It reuses the TensorIR context/arena/stream/error
owner. Density and physical direction upload once; primitive geometry and
radial records stream in bounded chunks; only final H1/S1 matrices return.
Per-chunk scalar status checks remain explicit. Runtime/header identity,
target and host-compiler ABI are bound across the loaded programs. Failed
inputs, kernels or accumulations block publication until successful reset.

`first_backend="cuda"` plus an explicit compiler is opt-in and independent of
`jk_backend`. Default CPU behavior stays intact. The downstream CPHF AO/MO
transforms and Krylov solve remain host-side, and the public/native state-size
admission is not changed. This completes the directional H1/S1 device source
slice, not a complete GPU HVP or a wholly resident response solve.

## Rejected alternatives

- Copy CPU formula strings into another CUDA Hessian module: duplicates
  scientific equations and makes CPU/CUDA drift difficult to diagnose.
- Return each primitive/shell derivative for host direction/density reduction:
  retains the very host scientific stage this task is meant to remove.
- Substitute density-fitting response into conventional second derivatives:
  changes the Hamiltonian and does not close the exact/direct chain.
- Present small device-buffer accounting as an entire Hessian memory bound:
  the SCF, J/K and response/Krylov owners coexist and have separate budgets.
- Infer GPU-resident CPHF or full Hessian support from device first sources:
  the remaining transform/solver and complete HVP assembly stages are distinct.

## Invariants

Full-Coulomb/Cartesian derivative semantics, explicit signed weights, attraction
operator-center motion, repeated center-to-atom mapping and once-only radial/
angular normalization must remain unchanged. Generic spherical/DF/ECP/range
requests are rejected rather than relabeled. External weights and direction
are fixed primitive inputs; electronic response is supplied elsewhere.

Outputs are not symmetrized to hide defects. Nonfinite terms/accumulations,
invalid records, stale compiled identities, incompatible ABIs and closed owners
fail explicitly. No failed partial output becomes an accepted result; reset
clears the device state for replay. Provider storage admission includes native
and Python numeric staging/publication but excludes compiler/code mappings,
caller metadata, native call stacks and CUDA context overhead.

## Evidence

- New pure compiler contracts plus first-derivative/directional/response CPU
  regression: **93 passed**.
- New CUDA primitive/source/response tests plus existing direct/DF J/K response
  regression on RTX 5090: **26 passed**.
- Compute-sanitizer memcheck for partial/late record failure, numerical overflow
  and impossible-budget/replay: **3 passed, 0 errors, 0 bytes leaked**. This is a
  subset of the device coverage, not an additive count.
- Ownership and pure compiler contract tests: **33 passed**, overlapping the
  CPU count above.
- Four sampled existing CPU first-component emitters (S/T/V and ERI) remain
  byte-identical to base `ec7c9e7d82ea6ec694f24a8e0e1a4fcb8ed89416`.
- The initial independent seven-AO water run measured maximum H1 error
  `3.38e-14` and S1 error `5.56e-17`; its complete CPHF residual was about
  `8.02e-15`. The test suite separately checks three-step native displaced
  Fock/overlap/density/energy-weighted-density differences and forbids CPU
  derivative/J/K/oracle callbacks on the CUDA execution side.
- A selected f/s/s/s contraction uses an independent native analytic derivative
  oracle, signed external weights and repeated physical centers. This is not
  qualification of every large high-angular-momentum molecular endpoint.

New first-source runtime libraries are compiled from the current checkout with
NVCC 12.9.86 for `sm_120`; the device was RTX 5090 with driver 580.95.05 under
finite Slurm allocations. The unchanged SCF/J/K bridge uses a pinned existing
CUDA library, SHA256
`eecfc179ca4ce1098699a395b75fc5f06cffc4c8a14f9ab0c51d973f61c5220c`.
CPU bridge library SHA256:
`61abd1253cfbcf729c73bafea92190af2b7cb822bab80a52feaed352cd2fb335`.
These are distinct provenance scopes; local results do not replace a fresh
whole-native CI build. No Release or external evidence archive is published.

## Consequences and revisit conditions

The conservative schedule still evaluates bounded local gradient components,
repeats primitive work across component tiles and compiles multiple programs.
Cold generation can dominate tiny systems. Correct device residency is not a
performance promotion; optimize output pruning, artifact reuse and source-work
amortization only with matched complete-endpoint evidence. New GPU tests are
opt-in so routine CPU CI does not compile or execute this CUDA workload.

Next #180 work must retain separate gates for response AO/MO and Krylov device
residency, the first-integral weighted contractions needed by relaxation, and
complete second-integral/Pulay/nuclear HVP assembly. The 12-AO/four-atom native
reference-entry limit and all DF/ECP/UHF/DFT/higher-method boundaries remain.

## References

#178, #179, #180, #449, #511, #526;
`docs/first_directional_derivatives.md`; `docs/hessian.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
