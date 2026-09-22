# Rejected: incomplete Direct-HF factorial acceptance

Status: rejected
Date: 2026-09-22

## Problem

The #669 closeout audit must separate the historical dirty-source 3.161731 s
receipt, the controlled 5.605620 s baseline, and current implementation controls.
Treating nearby numerical paths or archived timings as independent qualification
would hide an incomplete endpoint gate.

## Decision

Stop the current Direct-HF campaign and keep #669 open. Two 384-AO timing-only
processes at `0d89ab6fb6219d9af6641d5cfafe0b43f8387206` failed fresh independent
GPU4PySCF reference convergence at energy `1e-12`, gradient `1e-10`, screening
`1e-14`, max 200 cycles. Protocol v1 also primed each arm only at initialization;
control-transition order changed subsequent timings. Those receipts cannot
support numerical acceptance or a clean factorial speedup.

## Rejected alternatives

- Reusing README reference arrays: the 48-atom geometry differs; the matching
  96-atom record uses a looser reference gradient threshold.
- Calling current forced rebuild an exact pre-#698 reconstruction: later
  source changes mean it is only a current implementation control.
- Reporting corrected protocol v2 as qualified: it primes every transition,
  but only a partial 768-AO run was executed before cancellation.

## Invariants and revisit conditions

Preserve full energy-plus-force requests, independent numerical gates, physical
final-state checks, and distinct historical/strict profiles. Revisit after an
independently converged reference and per-transition priming are available, then
complete seven interleaved pairs and two independent processes at each required
key size, with the required operator-work and broader coverage ledger.

## Evidence

[Acceptance audit](../../../benchmarks/results/acceptance-closeout-20260922/README.md),
including `direct-disposition.json`, exact timing-only records, both runner
versions, source/library identities, and hashes of local raw receipts.
