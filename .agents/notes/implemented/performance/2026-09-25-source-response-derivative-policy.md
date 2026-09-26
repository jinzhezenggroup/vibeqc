# Decision: explicitly qualify source-only response derivative scheduling

Status: implemented
Date: 2026-09-25

## Problem
The source-only occupied producer clears the fitted view. The bridge previously
used the absence of resident raw/fitted storage as a reason to select generic
full-pair derivatives, confounding producer and consumer comparisons (#445).

## Decision
Add `VIBEQC_DF_SOURCE_DERIVATIVE_SCHEDULE=auto|qualify`. The default is unchanged.
`qualify` admits an already validated full-rank occupied source to the same
work/target profile as a fitted source. It does not bypass final-state,
representation, numerical, diagnostic, or unknown-target gates. All dependent
pair/compact/signature policies see the same promotion decision.

## Evidence
The host policy matrix exhausts all 32 combinations. CUDA numerical/performance
qualification is outstanding; no automatic promotion or speedup is claimed.

## Revisit when
Retain independent physical E/force and ragged/changed-geometry evidence with all
producer/consumer selectors and work counters pinned before changing the default.

## References
#1078, #445, #459. Agent: ChatGPT. Model: GPT-5.6 Sol.
