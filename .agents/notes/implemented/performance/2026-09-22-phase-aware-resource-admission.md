# Decision: account serialized resource phases by live set

Status: implemented
Date: 2026-09-22
Updated: 2026-09-26

## Problem

Resource estimates must follow actual allocation lifetimes. Serialized setup,
SCF, XC and force phases do not necessarily overlap, while prepared CUDA force
owners can remain live across later replays. Treating every phase as
simultaneously transient overstates memory; treating a retained owner as retired
understates it.

An accompanying experiment promoted automatic single-system dense RHF DF from
the bounded source-backed owner to a fully materialized resident owner whenever
the total automatic envelope could fit that owner.

## Decision

Keep the phase/lifetime accounting fixes:

- CPU serialized transient work is charged by its maximum live phase.
- Prepared CUDA force host/device arenas are persistent across replays and are
  charged separately from later transient setup/SCF work.
- Public DF value/response budgets remain explicit replay identity and must stay
  within the caller-visible total.
- A bound raw DF response owner is not retired while force response still
  borrows it.

Do **not** promote the default automatic DF route to the materialized resident
owner. Automatic positive resolved DF allowances continue to use the bounded
source-backed route. The provider-derived preferred peak and total-envelope
resident-owner override were removed from PR #970 after device acceptance
showed a reproducible large warm-start convergence regression.

Explicit user budgets, numerical thresholds, force definitions and the existing
bounded planner remain unchanged.

## Source-backed device residency is not host-owner promotion

Automatic DF has two independent decisions: the resolved allowance can keep
the source-backed CUDA J/K value owner device-resident, while the production
route still assembles source-backed metadata instead of a materialized host raw
tensor. Removing the device-residency floor does not prevent host-owner
promotion; that is governed by the preparation route. It only makes large
source-backed plans repeatedly stream and regenerate their value panels.

With the same RTX 5090/CUDA 12.9.86 Release configuration and identical
freshly generated 768-AO checkpoints, removing this floor left changed-warm
at three SCF iterations but raised its seven-sample median complete endpoint
from 2.045308 s to 7.148424 s. The diagnostic force trace changed from one
to six resident-whitening panels and from two to twelve factor GEMMs; the
source-backed flag stayed true. Restoring the device floor returned the median
to 2.049543 s with three iterations, one panel and two GEMMs. Energy and force
agreed with the original source-backed run to 0 and 1.2e-13 respectively.
The resource-qualified device floor avoids both the measured work
amplification and the separately rejected materialized-host convergence path.
Explicit caps and tight-device streaming remain bounded by their original
rules.

## Device evidence for withdrawing automatic resident promotion

On RTX 5090 / CUDA 12.9.86, shared-checkpoint comparisons used identical
base-generated checkpoint bytes and unchanged SCF tolerances.

- 96 / 192 / 384 AO passed the existing 2% same-work screen.
- At 768 AO changed-warm, base/source-backed required 3 SCF iterations while
  the resident candidate required 8 in every retained sample.
- Median complete endpoints were 2.045572 s versus 3.300133 s, a +61.33%
  regression.
- The separate evolving-density campaign also showed a +11.42% 768-AO
  changed-warm slowdown.
- Independent energy/force and public-budget correctness gates still passed.

This establishes a work-count/convergence-path change, not ordinary timing
noise. The resident-owner optimization remains a separate #439 investigation;
it must not be recovered by relaxing the 1e-12 energy or 1e-10 density
convergence gates or by adding an AO-count magic threshold.

## Remaining resource acceptance

The integrated resource work retains the independently qualified public
24/32/64 MiB energy/force/energy replay checks, including the genuine 24 MiB
streamed-force case, and the retained CUDA force-owner lifetime accounting.

The separate response-budget override behavior remains tracked in #1027.

## References

- #439
- #940
- #970
- #1027

Agent: ChatGPT
Model: GPT-5.6 Sol
