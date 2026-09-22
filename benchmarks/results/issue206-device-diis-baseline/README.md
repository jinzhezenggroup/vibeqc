# Initial repeated DF matrix after device DIIS (#206)

> **Checkout retention (2026-09-21):** `evidence.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue206-device-diis-baseline/evidence.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue206-device-diis-baseline/evidence.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

Five interleaved warm repeats per engine cover 96/192 spherical AOs, batch
one/four, energy and complete forces. All repeat errors satisfy the declared
1e-9 Ha / 1e-8 Ha/Bohr gates. An independent preflight records exact basis data,
metric spectra and the actual stock GPU4PySCF factor ranks for all ten input
geometries; both engines retain full auxiliary rank. The engines take different
SCF iteration branches in every pair, so these timings are explicitly unmatched.

| AOs / batch | VibeQC warm energy | GPU4PySCF warm energy | VibeQC warm forces | GPU4PySCF warm forces |
| --- | ---: | ---: | ---: | ---: |
| 96 / 1 | 0.01207 s | 0.06894 s | 0.26101 s | 0.26494 s |
| 96 / 4 | 0.03486 s | 0.27495 s | 1.03338 s | 1.06062 s |
| 192 / 1 | 0.06596 s | 0.08069 s | 3.25547 s | 0.35278 s |
| 192 / 4 | 0.23755 s | 0.32247 s | 13.03361 s | 1.39956 s |

The 192-AO force gate remains unmet: the ordinary ratios exceed nine, and no
shared iteration branch was demonstrated. The runner's successful numerical
exit is not performance acceptance. Maximum paired energy error is 1.535e-11
Ha and maximum force error is 8.396e-11 Ha/Bohr. Cold samples, all raw repeats,
per-item iterations/results, fixed warm-density policies, provider/basis details
and hardware metadata are retained. No compilation, reference preflight or
component profiling overlaps the clean endpoints. This artifact does not claim
whole-process memory acceptance or completion of the stricter direct-path gates.

A separate intrusive component run identifies host response-weight construction
in the default (zero DF subbudget) plan: at 192 AOs, its exclusive region is
2.361 s of a 3.322 s warm full-force call. Host raw-value gathers add 0.145 s,
metric/force orchestration about 0.154 s, and derivative contraction 0.587 s.
A 512-MiB source request already uses device response weights, but its warm call
rebuilds preparation: raw value work is 1.100 s and one-electron derivative
generation 0.871 s of 3.231 s total. These observations motivate reuse of the
existing GPU response-weight implementation with the default retained data;
they do not justify a new scientific kernel or dropping response terms.

The clean run uses GPU4PySCF 1.8.1, CuPy 14.2.0, cuTENSOR 2.3.1, PySCF 2.14.0,
FP64 and one RTX 5090. Exact versions and finite Slurm job IDs are recorded in
`summary.json` and the archive. Native library SHA-256 is
`44c140b4d46b69ea7846d8aa2d9aa8cc3bec8c198908fd9e830e9981f78d4bc7`,
the same frozen library as `../issue308-device-diis/`. The measured source head
is `f3dc1bdcfcd2fb397fc8a7082fab8d79de45aa92` (documentation only after the source
commit qualified there).

`evidence.zip` contains every raw result, rank preflight, trace and runner.
Every member was restored and compared byte for byte; its SHA-256 is
`dda7de93473601eeb67a080d46bf3ffd586c21739d6ba67031c75a7289ede8a4`.
Changed-geometry and larger integration/ablation measurements, explicit memory
and transfer accounting, direct regression checks and the #206 performance
acceptance remain open. No tracker is closed by this baseline.
