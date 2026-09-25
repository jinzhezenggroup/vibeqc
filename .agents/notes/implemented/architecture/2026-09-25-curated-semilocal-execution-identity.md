# Decision: Use one typed curated semilocal execution identity

Status: implemented
Date: 2026-09-25

## Problem

Prepared CPU and CUDA KS execution independently interpreted integer semilocal
selectors, while the same codes also entered retained execution identities.
That duplicated scientific execution identity and could drift as generic
compiled functionals are introduced.

## Decision

Use `dft::SemilocalFamily` as the single curated native execution identity.
Keep the existing stable numeric codes at serialized/device boundaries and
reject unknown codes before execution. CPU prepared execution and CUDA KS use
the typed family directly.

Composed B3LYP and WB97M-V paths retain their dedicated exchange/nonlocal
composition checks. The common curated dispatcher covers only LDA, PBE, and
r2SCAN. CUDA admission remains limited to those same three families.

## Invariants

- Stable codes remain LDA=0, PBE=1, r2SCAN=2, B3LYP=3, WB97M-V=4.
- Domain versions remain 1 for LDA/PBE/r2SCAN, 2 for B3LYP, and 3 for WB97M-V.
- Unknown serialized codes fail closed.
- CUDA admission remains LDA/PBE/r2SCAN only.
- This refactor grants no new method or backend capability.

## Evidence

PR #1273 rewires CPU prepared execution, CUDA KS construction, final-state
validation, and route-selection regressions through the shared identity.

## Revisit when

Revisit when generic compiled semilocal programs replace curated-family
selection at the prepared execution boundary.

## References

Issues #926, #934, #396, #1119 and #1121; PR #1273.

Agent: ChatGPT
Model: GPT-5.6 Sol
