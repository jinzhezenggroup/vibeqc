# Full-rank response and physical force-state evidence

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

This numerical repair applies metric inversion to linear factors before quadratic
products and validates the current physical-Fock projector before complete forces.
The frozen v14 library was used in Slurm 9946, 9947 and 9949. One subsequent
comment-only correction is bound explicitly in `review-source-binding-v1.json`.
No executable DF code changed after qualification. Integration with upstream
`d756162` adds its separately qualified ECP host-grid work; the shared ownership
snapshot is regenerated. `integration-v1.json` binds that source separation and
confirms the same PR ownership delta against the new base.

The 19-cell stock campaign completed with **11 warm wins, 3 losses, 2 inconclusive
results and 3 numerical failures**. This does not close #206 or establish universal
superiority. Every raw cell, qualifier, sample and failed verdict is retained.

## Warm endpoint results

Times below are milliseconds, median of seven interleaved synchronized repeats per
engine, using engine-local frozen post-cold densities and equivalent untimed
priming. These are complete ordinary energy/force endpoints with normal convergence
work, not fixed-iteration or matched-operator-work speedups. Native E/D requests
are 1e-12/1e-10; screening is 1e-14. Stock uses its normal cuTENSOR provider and
each cell independently verifies matching full retained auxiliary rank.

Predeclared statistics: both relative MAD <= 3%, reduction >= 2%, paired bootstrap
95% lower bound > 2%, 10,000 resamples with NumPy default_rng(206). Larger variable
and failed cells cannot supply a favorable performance claim.

| Cell (AO count, batch, endpoint) | VibeQC ms | Stock ms | Verdict |
| --- | ---: | ---: | --- |
| equal-96-b1-energy | 9.923 | 68.719 | win |
| equal-96-b4-energy | 27.641 | 274.698 | win |
| equal-192-b1-energy | 39.971 | 80.229 | win |
| equal-192-b4-energy | 155.158 | 319.324 | win |
| equal-96-b1-forces | 127.552 | 265.269 | win |
| equal-96-b4-forces | 493.654 | 1063.206 | win |
| equal-192-b1-forces | 163.722 | 351.029 | win |
| equal-192-b4-forces | 650.292 | 1403.440 | win |
| equal-384-b1-energy | 296.993 | 136.419 | loss |
| equal-384-b1-forces | 742.003 | 614.453 | loss |
| equal-768-b1-energy | 1011.736 | 1231.658 | variable; inconclusive |
| equal-768-b1-forces | 2145.342 | 2140.090 | tie/inconclusive |
| equal-OH/UHF-b1-forces | 9.376 | 242.269 | win |
| equal-ammonia-b1-forces | 9.829 | 237.251 | win |
| practical-96-b1-forces | 615.435 | 272.376 | loss |
| practical-96-b4-forces | 2380.214 | 1096.540 | FAILED numerical gate |
| practical-192-b1-forces | 3364.475 | 582.350 | FAILED numerical gate |
| practical-192-b4-forces | 13556.235 | 2339.546 | FAILED numerical gate |
| practical-OH/UHF-b1-forces | 19.624 | 204.431 | win |

`equal` uses the original same-basis workloads. `practical` explicitly supplies
cc-pVDZ/cc-pVDZ-JKFIT; historical case names provide geometry/spin only. B4 stock
comparisons scale each geometry, unlike the identical-geometry independent batch
regressions. The original gates are preserved: equal 96 E/F 3e-11, equal 192
E 1e-10/F 5e-10, larger/OH/ammonia E 1e-9/F 1e-8; practical E/F 3e-11.
Stock gradient tolerances are 1e-9 (96), 1e-8 (192), and 1e-10 otherwise.

The cached CPU reference matches the practical 192 B1 geometry and bases exactly.
Across seven saved samples, maximum native force error is 7.524e-13 and stock
error is 1.4375e-10. This diagnoses one failed comparison; its paired verdict
remains failed. The cached reference cannot qualify scaled B4 geometries.

## Numerical qualification and limits

- Slurm 9946: native final-state, device-validation, snapshot and occupied-response
  tests passed; CUDA memcheck reported zero errors; all 138 Python tests passed.
  The 23 practical configurations retain 92 endpoints across cold, warm, changed
  and changed-warm phases. Original 1e-10 (40 endpoints) max E/F errors are
  2.274e-12/2.5615e-11; tight 1e-12 (52 endpoints) 2.387e-12/4.3302e-13.
- Slurm 9947: dense/packed 192 B1, 96 B4 and 192 B4 retain 144 endpoints, 72 at
  each density request. All pass the unchanged E/F 3e-11 gates. All **72 original
  1e-10 endpoints** pass independent CPU residual gates. **24 tight 1e-12 octamer
  endpoints fail** independent commutator/projector gates; maximum defects are
  1.0795e-12 and 1.4493e-12. These remain failures and are a separate follow-up.
- Work ledgers count current-F solves and promoted probes without duplicate
  solves: original group 158 Focks/solves, 86 corrections, 80 probes, 8 promotions;
  tight group 164/92/82/10. Intrusive timing is excluded from clean comparisons.
- Small 128-MiB v14 tests do not qualify the larger v14 constrained-memory boundary.
  Earlier v12 low-memory results cannot qualify the changed force-state selection.
- Repeated cold/changed timing, full independent residual coverage, direct-path
  failures, and complete process memory/operator-work accounting remain for #206.
  No failure, tolerance or correction cap is waived by this PR.

## Reproduction and retention

Run from the repository root with a Python containing NumPy:

```bash
python benchmarks/results/issue206-stable-response/replay.py
```

This CPU-only command verifies every archive/member hash, extracts to a temporary
directory, recomputes all warm verdicts, and compares the result with the retained
summary. The archives preserve exact source scripts, scientific arrays in JSON,
input/geometry/basis identities, frozen build manifest, numerical audits, named
test receipts and historical failures. `manifest.json` binds compressed and
uncompressed bytes. Archive inclusion never implies numerical admission.

`sources-evidence.zip` contains the frozen changed sources, patch, build cache,
build identity and ownership reports. Reconstruct production from the reported
base and frozen sources; use the normal repository CMake build. The runners in
`history-evidence.zip` preserve exact historical commands and paths. Adapt only
environment/library/output locations for new runs, retain the numerical settings,
and use a new output directory. Each GPU run must use finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 bash -lc '<qualification or clean comparator command>'
```

Keep Slurm device visibility unchanged. Never overlap clean timing with builds,
profilers or intrusive tracing. The exact clean controller has 19 predeclared
cells; it exits nonzero when any cell fails and retains all completed samples.
Production has no CPU oracle dependency. Saved density arrays, full component
traces, failed-run stdout and binaries remain in the local artifact inventory;
Git retains compact scientific records and hashes, not the multi-GB build tree.

Ownership versus e3ea91d: handwritten scientific roles +583/-40 (scientific
+578/-40, oracle +5/-0), runtime +9/-0, no unchanged-line reclassification.
Generated capabilities are unchanged. Spectral rank-deficient, bounded-source
and scalar/serial routes remain necessary fallbacks/oracles. The decision note
records their retirement conditions and every material rejected alternative.
