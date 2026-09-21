# Native RCCSD(T) response execution (#155 C slice)

## Decision

Route the qualified CPU RCCSD(T) response/gradient TensorIR through the generic #772 `NativeTensorProgram` backend before exposing public forces. Scientific equations remain owned by the existing TensorIR builders; this slice adds execution orchestration, not another Lambda/Z implementation.

## Boundary

- `BoundCCSDLambda(backend="native-cpu")` owns one `NativeCCTensorExecutor` for its complete response lifecycle.
- Compiled programs are cached by exact TensorIR logical identity.
- Lambda RHS/actions, fixed-orbital parameter VJPs, standard-(T) tile VJPs, raw-Hamiltonian/Fock pullbacks, canonicalization and AO back-transforms share that executor.
- The legacy interpreter backend remains available for independent tooling/oracles; the complete RCCSD(T) CPU force endpoint explicitly selects native CPU execution.
- Public C ABI / `Calculator("ccsd(t)")` force capability remains fail-closed until the C++ method owner publishes this response lifecycle.

## Admission

The generic native CPU TensorIR default stays at 4096 nodes. The CC response owner explicitly requests 8192 nodes, bounded by a 16384 hard maximum. This admits the qualified NH3 final triples-response tile (5258 nodes) without widening unrelated consumers. Existing byte/work budgets remain mandatory.

## Validation

H2O and NH3 complete RCCSD(T) analytic gradients pass the pinned PySCF 2.14.0 acceptance gate at `1e-6 Eh/bohr`. A focused Lambda test makes the NumPy TensorIR interpreter raise on any call and still completes through the native backend. The generic 4096-node rejection and the explicit NH3 8192-node admission are both regression-tested.

Cold NH3 AOT compilation is intentionally not optimized in this correctness slice; compiling many independent response artifacts should be handled as a separate compiler-performance change.

Agent: ChatGPT
Model: GPT-5.6 Sol
