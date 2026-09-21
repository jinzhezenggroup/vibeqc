# Native RCCSD(T) CPU energy owner (#155 C slice)

## Delivered

- Activate the existing `VIBEQC_METHOD_RCCSD_T` ABI id for energy-only native CPU execution and homogeneous prepared batches.
- Reuse the qualified native RCCSD reference/solver state; no second RCCSD equation stack is introduced.
- Generate the standard canonical `(T)` CPU evaluator directly from the audited literal inventory in `tools/vibeqc_cc/triples.py`.
- Retain canonical occupied/virtual orbital energies in the composed owner without changing the standalone RCCSD solver problem or its memory contract.
- Bound triples scratch to `13 * nocc^3` doubles and never materialize full T3.
- Publish separate triples energy, virtual-domain work count, workspace and inventory identity in the additive correlation diagnostic.
- Fail closed for native CUDA, analytic forces, density fitting, frozen core, ECP and open-shell references.

## Numerical acceptance

On the public H2O/STO-3G endpoint, the native total RCCSD(T) energy differs from the committed pinned reference by approximately `2.4e-13 Eh`; the `(T)` component differs by approximately `1.7e-15 Eh`. A direct random `(nocc,nvir)=(2,3)` generated-C++ comparison against the audited Python triples evaluator agrees to approximately `5e-17 Eh`.

This slice productizes the already delivered energy science. It does not complete #155: native/public analytic forces still need to connect the merged #746 response/gradient owner through the common native TensorIR boundary from #772.

Agent: ChatGPT
Model: GPT-5.6 Sol
