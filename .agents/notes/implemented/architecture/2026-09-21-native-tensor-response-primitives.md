# Native TensorIR CPU response primitive coverage

## Decision

Extend the generic FP64 `NativeTensorProgram` materializing backend rather than
introducing a CCSD(T)-specific native derivative path.

The backend now lowers the ordinary TensorIR primitives used by generated CC
response graphs: divide, scaled bilinear quotient, transpose/reshape/slice,
broadcast, gather/indexed gather, scatter-add and segment-sum, in addition to the
existing input/constant/add/multiply/einsum/reduce set.

Declared dense tensor symmetries remain metadata, not packed storage. Native
execution validates symmetric inputs with the same numerical gate as the
reference interpreter and then evaluates the dense graph normally.

## Consumer proof

Acceptance compiles and executes generated RCCSD Lambda transpose actions,
Hamiltonian/source pullbacks and bounded standard-(T) tile VJPs through the
generic native backend and compares them directly with the TensorIR interpreter.
A larger `(nocc=2,nvir=3)` generation gate covers the same production graph
families without method-specific native equations.

## Boundaries

- FP64 only; no implicit dtype conversion.
- Unsupported TensorIR primitives remain fail-closed at generation.
- This is a component executor, not yet a native RCCSD(T) `PreparedCalculation`
  or public Calculator capability.
- Method state identity, solver convergence, provider lifetime and transactional
  force publication remain owned by the CC method layer.

Refs #155, #151, #181.

Agent: ChatGPT
Model: GPT-5.6 Sol
