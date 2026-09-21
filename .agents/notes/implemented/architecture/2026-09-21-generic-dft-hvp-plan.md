# Decision: derive DFT Hessian/HVP eligibility from MethodIR primitives

Status: implemented (compiler planning slice)
Date: 2026-09-21

## Problem

Issue #180 still described DFT Hessians as a later method-level milestone even
though MethodIR, native LDA/PBE CPKS, XC feature Hessians and second-integral
HVP primitives already exist. Implementing PBE, B3LYP and r2SCAN Hessians as
separate scientific pipelines would duplicate the compiler-first architecture
of #181/#396.

## Decision

Add a data-only `StationaryHVPPlan` that derives the second-order source
topology from a resolved MethodIR plus the existing stationary mean-field
envelope. The first admitted planning domain is direct, all-electron, real FP64
LDA/GGA RKS/UKS using the qualified native SCF point model.

LDA and GGA share one source inventory:

- one-electron;
- Coulomb;
- XC AO motion;
- XC grid-point motion;
- XC partition-weight motion;
- overlap/Pulay;
- nuclear repulsion.

The method graph supplies active `rho`/`sigma` ingredients. The plan records
the shared #179 CPKS, XC feature-Hessian directional action, and #178 weighted
second-integral HVP contracts without executing or promoting them.

Any additional active primitive fails closed unless its second-order rule is
registered. This first slice therefore rejects full/range-separated exchange,
tau/meta-GGA, nonlocal correlation, DF and ECP rather than silently inheriting
Hessian support from energy or gradients.

## Evidence

- new compiler tests: 12 passed;
- related MethodIR/stationary-gradient/typecheck regression: 92 passed total;
- Ruff check and format: passed;
- `git diff --check`: passed.

Public Calculator DFT HVP/Hessian remains unsupported until native geometric
directional consumers and complete molecular validation land.

Agent: ChatGPT
Model: GPT-5.6 Sol
