# Decision: separate packaged stationary execution targets from compilers

Status: implemented routing; installed GPU qualification remains pending
Date: 2026-09-21

The public all-electron AOT force path previously required NVCC before looking
for the packaged library. A valid artifact could never reach its loader on a
compiler-free installation. Target identity is required for execution, but a
compiler executable is not required for this native seven-source reduction.

Pass an explicit CudaTargetInfo independently of an optional compiler. Admit
compiler=None only with both packaged stationary and native-grid artifacts on
the all-electron path. ECP and JIT consumers still require a real compiler;
compiler/target disagreement rejects. Keep every plan, binary, precision and
code-object check in the existing loader. Missing assets do not invoke JIT.

Two public-call regressions fail before the routing repair. Afterward all 35
selected AOT/lowering/owner/budget/spin tests pass. The orchestration test runs
both JIT and prebuilt branches and makes every compilation function fail in
prebuilt mode. This is host contract validation, not an installed-artifact GPU
numerical run. No scientific equation, numerical gate or CUDA capability changes.
The complete installed NVCC-blocked six-method GPU campaign remains required.

Agent: ChatGPT
Model: GPT-6 Astra Pro
