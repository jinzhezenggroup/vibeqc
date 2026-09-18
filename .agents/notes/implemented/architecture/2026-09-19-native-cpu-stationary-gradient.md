# Decision: native CPU state with complete compiler-plan gradient diagnostics

Status: implemented
Date: 2026-09-19

## Problem

Issue #163's merged source-contraction plan could not consume a current CPU KS
state or produce a complete molecular gradient. A correct handoff must use the
actual native physical F[D], not detached arrays or the orbital frame preceding
the final Fock rebuild. A functional formula alone also omits moving-grid,
partition, Pulay and nuclear terms.

## Decision

Retain CPU RKS D/F under the prepared owner and solve epoch. Canonicalize the
actual Fock only on explicit export and reuse the existing physical final-state
validator before W or a snapshot is published. Extend the opaque snapshot with
a CPU-only version-two payload and explicit host-device sentinel; leave the
CUDA version-one format intact. The same rejection rules apply to replay,
failed requests, changed geometry and relabeling.

Use the existing StationaryGradientPlan/TensorIR weights and complete source
inventory. Generate first S/T/V and ERI derivatives from the existing integral
DAGs, and the small nuclear-pair derivative from the same scalar Graph. The
native record loop owns validation/reduction, not derivative equations.
Normalized primitive records and physical-center scatter remain runtime data.
No new PBE-specific scientific implementation or production oracle is added.

The initial executable chain is deliberately a **complete diagnostic**. Native
SCF, AO jets, XC point coefficients and generated integral derivatives compose
with interpreted plan weights/reduction, AO pullbacks and Becke JVPs. s/p RKS is
qualified first. Public forces, full native lowering, UKS, DF/ECP, d/f gradient
coverage, CUDA and higher derivatives remain unqualified here.

## Quadrature failure and rejected alternatives

A first full asymmetric-water run rejected points whose tiny Becke weight
rounded to zero differently in the native value and generated-response paths.
Recovering the raw measure by dividing the final weight by the partition is
ill-conditioned and loses valid boundary derivatives. The CPU export now
reconstructs the atomic measures from the native quadrature routine on demand;
the gradient uses raw measure times generated partition derivative directly.
Energy-only grid execution retains no extra point-sized array.

Rejected alternatives: dropping grid response, loosening the force gate,
projecting the total force to enforce translation, relabeling a CUDA result as
CPU, adding an entire HF force, differentiating SCF iterations, or introducing
a permanent handwritten PBE gradient assembler. None fixes the actual contract.

## Invariants

- D/F/C/epsilon/occupations/W and the energy describe the same current native
  state and quadrature; reconstruction labels do not establish ownership.
- All seven signed sources are included exactly once; force is minus gradient.
- Source generation has no public runtime/PySCF dependency. External programs
  are independent test oracles only.
- Small native snapshots cannot be promoted into unqualified methods/backends.
- Compiler input is atomically published before hashing and compilation;
  concurrent calls must never see a truncated hash-named source. Reuse includes
  compiler, source, header and flag identity through the existing cache.
- Partial derivatives or failed late tiles cannot publish a complete result.

## Evidence and reproduction

`tests/python/test_dft_complete_cpu.py` compares the total and seven sources
with independent PySCF analytic derivatives, and every nuclear coordinate with
three fully recomputed/reconverged central differences. Asymmetric molecules
make the individually large point-motion and partition terms visible rather
than hiding errors behind symmetry. Tests also vary tile shapes and atom order,
translate the molecule, replay warm state, revoke malformed native requests,
inject a late primitive failure and block all oracle imports in a fresh process.
The same final-state tests preserve the separately gated CUDA snapshot cases.

A separate review fixture, with water coordinates `(0.13,-0.12,0.08)`,
`(0.16,1.40,1.17)`, `(-0.21,-1.32,1.21)` Bohr, native STO-3G records and a
16/6/12 radial/polar/azimuth grid, gave maximum total errors about `1.91e-11`
(PBE) and `1.90e-11` (LDA) Eh/bohr against PySCF 2.14.0 full analytic grid
response. Maximum individual-source errors were below `5.84e-11`; translation
defects were below `1.5e-14`. These are small-fixture correctness observations,
not general accuracy/performance or final-head GPU claims. The maintained test
suite supplies the reproducible acceptance gates; large transient outputs and
compiled sources are not committed.

## Consequences and revisit criteria

The diagnostic is mathematically complete but not a native production endpoint.
It performs full ordered AO quartet work and `3*Natom` partition traversals,
although intermediate arrays are tiled. Its work/tile report is not a global
peak-memory guarantee, and interpreter/compiler overhead is real.

Next lower the same plan contractions, XC pullbacks and partition response to
native CPU/CUDA execution, add production resource/capability qualification,
and expand angular/spin domains under independent gates. Do not rewrite the
scientific derivative equations for each named functional.

## References

Refs #163 and #396. Builds on #455, #473 and #484.
Current contract: `docs/ks_diagnostics.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
Implementation agent: Codex CLI 0.155.0
Implementation model: gpt-6-astra
