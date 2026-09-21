# RCCSD(T) analytic-force binding (#155 C slice)

## Delivered

- Bind `rccsd_t_force` directly to the merged #746 complete conventional RCCSD(T) analytic-gradient owner.
- Add homogeneous `PreparedRCCSDTForceBatch` / `rccsd_t_batch_forces` with input-order preservation, immutable successful forces, and per-item failure isolation.
- Report the internal facade as energy + forces while keeping `native_public=False`.

## 2026-09-21 native-energy follow-up

The boundary below records the state when the force binding landed. A later
#155 C slice activates `VIBEQC_METHOD_RCCSD_T` / `Calculator("ccsd(t)")` for
native **CPU energy and homogeneous prepared batches only**. Native/public
forces remain fail-closed, so the force-binding ownership statement and the
compiler-first requirement remain unchanged.

## Architecture boundary

At the time this force-binding slice landed, the reserved `VIBEQC_METHOD_RCCSD_T` C ABI and `Calculator("ccsd(t)")` were still fail-closed. The complete response graph used generated TensorIR einsum/VJP programs for CC Lambda and orbital response, and copying those equations into CCSD(T)-specific C++ would have created a second scientific implementation in conflict with the compiler-first #155/#181 direction.

That prerequisite was subsequently addressed by #772 and the native-energy follow-up above. The remaining force promotion must still reuse the #746 graph and common generated/native TensorIR execution rather than add a handwritten CCSD(T)-specific Lambda/Z stack.

## Validation contract

The force binding does not alter the #746 numerical gates: conventional real closed-shell RHF, all-electron, no frozen core/DF/ECP/open shell, with complete-gradient acceptance against pinned PySCF and the existing `1e-6 Eh/bohr` maximum-error gate.

Agent: ChatGPT
Model: GPT-5.6 Sol
