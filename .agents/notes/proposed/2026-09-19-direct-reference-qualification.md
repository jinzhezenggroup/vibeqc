# Proposal: qualify an explicitly tighter direct-force reference

Status: proposed; historical external failures remain failures
Date: 2026-09-19

## Problem

PR #427 retains two failed 96-AO paired force comparisons against a stock
reference using an orbital-gradient threshold of 1e-9. Independent CPU
reconstruction diagnoses a larger stock error than native error. An electronic
residual threshold is not a nuclear-force error bound, so tightening a setting
alone is insufficient evidence of correctness or of a stable reference.

## User-authorized experiment

Preserve the native source/library, exact geometries, 100-cycle budget, native
controls and every historical energy/force/speed gate. The historical campaign
and all failed samples remain immutable. Sweep stock 96-AO convergence from
1e-9 through 1e-10, 1e-11 and 1e-12, recording repeated forces, physical Fock
residuals, convergence flags and independent CPU errors. The predeclared
qualification margin is 3e-12 Eh/bohr for both CPU agreement and force drift
under a tighter control; unqualified or nonconverged results cannot authorize
promotion. Accuracy diagnostics are not endpoint performance measurements.

## Explicit runner contract

The matrix runner accepts repeatable `--reference-gradient-tolerance AO=TOL`
arguments, for example `96=1e-11`. It rejects nonfinite/nonpositive values,
duplicate or unused AO sizes, and any loosening of historical reference
convergence. No default changes. Summary records contain both historical and
effective thresholds and an explicit policy label. The 192-AO settings remain
historical when only the 96-AO size is overridden. All four endpoints and every
one of their seven interleaved pairs remain part of the follow-up campaign.

A passing follow-up is a result under a new, disclosed reference policy, never
a retroactive change to the old failed campaign. Complete endpoint timing must
include the extra stock SCF work and keep iteration-matched and ordinary
statistics distinct. This work neither replaces the stock provider with CPU
calculations nor introduces a CPU dependency into production.

## Boundaries

No force-threshold relaxation, dropped repeats, source substitution, general
GPU4PySCF superiority claim or release publication is authorized by this change.
Refs #206 and PR #427.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Follow-up diagnostic: full-density stock Fock rebuilding

The tolerance-only sweep already contains nonconverged repeats on the first
two exact geometries at 1e-11/1e-12, so it does not qualify a new reference.
Its partial records and the decision to stop unproductive remaining work are
retained separately. An additional Slurm step did not start; the idle original
allocation was released rather than held indefinitely.

A separately identified sweep tests the supported stock `direct_scf=False`
option, which disables incremental density/Fock accumulation but keeps the
same GPU4PySCF provider. The matrix runner exposes this as repeatable
`--reference-full-fock AO` controls, forwarded only to those AO-size reference
arms. Summary and per-point records explicitly identify the incremental/full
policy. The default remains incremental; no native setting changes. Promotion
still requires the declared CPU agreement/stability checks and all paired gates.
The added reference work is part of its endpoint, not hidden setup.
