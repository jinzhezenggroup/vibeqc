# RCCSD(T) analytic-force binding (#155 C slice)

## Delivered

- Bind `rccsd_t_force` directly to the merged #746 complete conventional RCCSD(T) analytic-gradient owner.
- Add homogeneous `PreparedRCCSDTForceBatch` / `rccsd_t_batch_forces` with input-order preservation, immutable successful forces, and per-item failure isolation.
- Report the internal facade as energy + forces while keeping `native_public=False`.

## Architecture boundary

The reserved `VIBEQC_METHOD_RCCSD_T` C ABI and `Calculator("ccsd(t)")` remain fail-closed. The complete response graph still contains generated TensorIR einsum/VJP programs used by CC Lambda and orbital response, while the generic native CPU TensorIR emitter currently supports only a smaller elementwise/reduction subset. Promoting the native method by copying the derivative equations into C++ would create a second scientific implementation and violate the compiler-first #155/#181 direction.

The next native-public slice should extend the common native TensorIR/generated-program execution boundary (or introduce an equivalent reusable generated artifact owner), then reuse the same #746 graph. It should not add a CCSD(T)-specific handwritten Lambda/Z stack.

## Validation contract

The force binding does not alter the #746 numerical gates: conventional real closed-shell RHF, all-electron, no frozen core/DF/ECP/open shell, with complete-gradient acceptance against pinned PySCF and the existing `1e-6 Eh/bohr` maximum-error gate.

Agent: ChatGPT
Model: GPT-5.6 Sol
