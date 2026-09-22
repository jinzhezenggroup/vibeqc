# Issue 434 fixed-density operator diagnosis

This record completes the failed Slurm 9948 capture for the preserved
`octamer-b1-dense-tight / changed / item0` state without rerunning SCF,
changing the density, or relaxing the strict `1e-12` residual gate.

The old diagnostic failed before Fock capture because it supplied a nonzero
occupied rank capacity to a dense DF value-storage plan. The corrected probe
uses `DfPairStorage::Dense` with rank capacity zero. Slurm 10499 then
completed against the exact frozen candidate-v14 library
(`source_identity=2edd5723...`, library SHA-256 `54491bce...`).

## Result

At the identical saved density, the native operator passes the strict gate:
commutator `8.3877e-13`, projector maximum `6.8967e-13`. The independent
operator still fails: commutator `1.0781e-12`, projector maximum
`1.4336e-12`.

Cross-composition localizes the difference to Fock rather than overlap.
Native Fock with independent overlap passes, while independent Fock with
native overlap still fails. The maximum native/independent Fock difference is
`1.5522e-12` at AO element `(10, 12)`; there the Coulomb difference is
`-1.5626e-12`, versus `1.13e-15` from Hcore and `9.28e-15` from
`-0.5 K`.

The native/independent raw three-center and metric maxima are
`1.8492e-12` and `3.1317e-12`, respectively. A common CPU Coulomb
recomposition sees raw-only and metric-only perturbations of
`9.27e-13` and `7.99e-13`. That split is diagnostic rather than an
attribution below `1e-12`: the common recomposition itself differs from
the independent/native J implementations at roughly that scale.

Therefore the preserved failure is not evidence of stale final-state
selection or an overlap/eigensolver mismatch. The next numerical slice should
focus on reproducing the independent/native RI-J contraction arithmetic from
the same raw A and metric, before changing SCF convergence behavior.

## Reproduction tooling

`benchmarks/df_fixed_density_probe.cpp` dumps native overlap, Hcore, metric,
raw/whitened three-center values, J, K, Fock and the unchanged density.
`benchmarks/issue434_fixed_density.py` verifies fixture hashes, executes the
probe only inside a Slurm GPU allocation, compares the native snapshots with
the independent fixture, evaluates both residuals and both cross-compositions,
and records the raw/metric Coulomb sensitivity.

No generated binary snapshots or the 262 MiB raw fixture are committed here.
The compact result records the frozen fixture/library identities needed to
audit the run.

Agent: ChatGPT
Model: GPT-5.6 Sol
