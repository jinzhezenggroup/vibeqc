# Verified RHF energy-state qualification

> **Checkout retention (2026-09-21):** `qualification.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue311-rhf-energy/qualification.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue311-rhf-energy/qualification.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

Implementation source: `69f2c912e4dcf0c33d48b633c6a3c9c6c6f6240a`. The manifest pins the
matching scientific source and native CUDA library. Final qualification used
Slurm job 9494 on `main` with one RTX 5090 and a finite 20-minute allocation.

All 27 CPU native suites, 85 Python protocol/resource/ownership checks (19
CUDA-only skips), eight new GPU selection tests, six GPU native suites and
145 GPU regression checks pass. Snapshot memcheck reports zero errors/leaks;
all repository hooks pass. The actual RHF energy/rebuild/force-transition case
also passes memcheck with zero errors/leaks under Slurm job 9495.

The eight cases cover Cartesian/spherical water, batch one/four and resident/
8 MiB tile budgets. Independent owners exercise candidate reuse, forced device
rebuilding and forced reference rebuilding across cold, warm, changed geometry
and force/energy transitions. Energies and complete forces agree with independent
CPU DF within 1e-9 Eh and 1e-8 Eh/Bohr; failed neighbors recover independently.
The full-force transition still uses its existing qualified consumer.

Actual traces show four/five corrections per cold item, zero for unchanged
warm reuse and one for each forced warm rebuild. Reuse still evaluates current
physical F once; forced warm rebuilding evaluates it twice. Changed-geometry
and frozen-seed transitions may require five corrections. All work is retained;
these intrusive traces are not clean timing or a speedup claim.

The initial four-correction budget failed honestly. The bounded consumer now
allows sixteen while preserving all numerical thresholds; the original failure
log is separate from final qualification. `qualification.zip` retains 137 files,
including 120 host traces, in 205,565 bytes. Every member was restored
and compared byte-for-byte before retention.
