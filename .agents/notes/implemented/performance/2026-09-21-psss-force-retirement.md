# Direct-HF psss force mathematics retired

Status: implemented
Date: 2026-09-21

## Decision

The force-only psss weighted-ERI expression introduced by #717 now owns the
production Direct-HF psss force mathematics unconditionally. The existing
fixed, resident-bra and bounded-paged native schedulers remain unchanged.

The retirement removes the handwritten weighted PA/PQ derivative algebra,
`GeneratedMath` A/B specialization, `VIBEQC_PSSS_WEIGHTED`, the prepared-plan
route bit and device-batch route state. It does not remove the ordinary psss
Fock value implementation, screening, density contraction, primitive
orientation/normalization, atom scatter or translational recovery.

## Qualification

Same-binary RTX 5090 / CUDA 12.9 comparisons used pre-retirement master
`41be6f8d8db7b74336810cf3e27c5b33595bbc52` and library SHA-256
`35d77e571598182caeedd2170fe21982d235d2989dc5d1666d23a50b06794564`.

RHF def2-SVP covered batch 1/3 across fixed/resident/paged, cold/warm and
changed-geometry replay. UHF used the established 19-AO OH/def2-SVP spherical
doublet at batch 1/4 across the same schedules. Numerical differences remained
at strict validator/machine-precision levels.

Two initial >2% timing outliers were remeasured with interleaved/longer ABBA:
- RHF batch-1 fixed: 1.2656x cold, 1.0159x warm, 0.9992x changed-warm.
- UHF batch-4 fixed, 12 repeats (Slurm 10609): 1.1751x cold, 1.0119x warm,
  0.9945x changed-warm.

No reproducible complete endpoint regression exceeds the 2% structural
retirement ceiling. Full evidence is retained under
`benchmarks/results/issue356-psss-retirement/`.

## Ownership

The candidate ownership report changes handwritten scientific CUDA from
13,012 to 12,968 lines, a net retirement of 44 scientific lines. Oracle,
performance-exception and runtime totals are unchanged.

Agent: ChatGPT
Model: GPT-5.6 Sol
