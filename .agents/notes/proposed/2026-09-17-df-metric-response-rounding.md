# Stabilize practical-auxiliary metric response arithmetic

Status: proposed (no production numerical change)
Date: 2026-09-17

## Evidence

This supplements the practical-auxiliary force-accuracy investigation in #430.
Slurm 9906 captures actual final native densities, whose independent CPU forces
agree with the tight oracle within 1.91e-12 for OH and 1.63e-12 for water. The
failed RHF CPU adapter is retained; CPU reconstruction reuses its saved arrays.

Slurm 9907 gives identical explicit independently computed weights to the OH
CUDA derivative consumers. Their assembled force agrees with libcint within
3.38e-14, isolating response generation as the next target.

Slurm 9908 calls the unchanged production CUDA response with independent A/M
and the native density. It reproduces a 2.85e-10 force error. Applying M^-1 to
the factors before forming quadratic products reduces the combined discrepancy
to 6.55e-16 using exactly the same CUDA metric eigensystem. Recomputing both
raw quadratic products and the spectral map in host extended precision gives
8.17e-13; extending the map alone leaves 6.38e-11. Thus rounding before and
within the metric reverse map has an independently reproduced force impact.

## Candidate direction and invariants

For verified full-rank metrics, form the inverse-applied Coulomb and exchange
factors first, then their metric adjoints. Preserve all directions and verify
the combined A/M force; individual component discrepancies can cancel when
the response uses a consistent inverse.

Do not extend the full-rank identity to truncated metrics. Their spectral
Frechet map must include retained/discarded subspace motion and preserve the
rank-crossing guard. Bound all auxiliary/AO panels and account for extra raw
reads, products, transfers and scratch before choosing a production design.

The original native endpoint still uses its own integrals, whereas the
response-only bridge uses independent libcint integrals. This experiment
isolates an arithmetic failure mechanism; it does not claim exact decomposition
of the endpoint's original 1.24e-10 OH discrepancy. Water's response mechanism
also requires independent verification before a universal attribution.

## Acceptance boundary

The #430 clean failures are unchanged and #206 remains open. No new clean
retries, threshold relaxation, precision promotion or performance admission
follows from untimed diagnostics. Before promotion, a candidate must pass the
original complete-force gates, practical unequal auxiliaries, UHF, bounded
fallbacks and rank-deficient/rank-crossing tests.

The complete retained evidence and CPU audit are in
`benchmarks/results/issue206-response-diagnosis/README.md`.
