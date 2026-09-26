# Host DF response traversal (#283)

The largest force component in the zero-budget compatibility route was the
host response-weight matrix products. Interchanging independent column and
reduction loops makes the inner traversal contiguous. Every output still
accumulates reduction indices in ascending order. Equations, spin coefficients,
metric response, derivative terms, scratch sizes and fitting policy are unchanged.

The three unprofiled pairs per case compare pristine `640a81e` with `8697318`
on one Slurm-assigned RTX 5090, Release/CUDA 12.9.1/sm_120. The baseline archive
is in `../issue283-component-baseline`. Exact library/source hashes and complete
arrays remain in the archives; `summary.json` collects the derived results.

| AO | energy before / after (s) | energy + force before / after (s) | endpoint speedup | force increment before / after (s) |
|---:|---:|---:|---:|---:|
| 96 | 0.454683 / 0.461434 | 1.055345 / 0.956040 | 1.104x | 0.598335 / 0.491201 |
| 192 | 5.555643 / 5.558586 | 12.432925 / 9.700111 | 1.282x | 6.886455 / 4.141359 |

Each table cell is its own sample median. Paired iterations remain 34/39 and
energies agree exactly. Maximum force changes against the pre-optimization
recorded arrays are 5.69e-14 / 6.76e-14 Eh/bohr (gates: 1e-10 Eh, 1e-9 Eh/bohr).

Separate diagnostic runs locate the improvement in exclusive host response
time: 249.460 to 144.199 ms (96 AO), and 5123.100 to 2385.912 ms (192 AO).
Named components/synchronization cover 96.3–96.6% of warmed 96-AO increments
and 95.2–95.5% of all 192-AO increments. First-pair initialization gives 140.7%
for the first 96-AO sample; this is retained as variation and does not pass
the attribution gate. Profiled intervals are not used for endpoint speedups.

The full #206 energy-plus-force matrix has five interleaved warm samples for
every endpoint. All numerical gates pass, but **all iteration branches are
unmatched**: VibeQC takes 2/3 iterations (96/192 AO), GPU4PySCF takes 1.
These medians therefore make no matched cross-engine speed claim.

| AO / batch | VibeQC warm median (s) | GPU4PySCF warm median (s) | maximum energy error (Eh) | maximum force error (Eh/bohr) |
|---:|---:|---:|---:|---:|
| 96 / 1 | 0.478228 | 0.245457 | 1.43e-12 | 4.46e-11 |
| 96 / 4 | 1.903936 | 0.980147 | 5.46e-12 | 5.49e-11 |
| 192 / 1 | 6.456374 | 0.331116 | 1.76e-11 | 8.72e-11 |
| 192 / 4 | 25.771654 | 1.327074 | 1.43e-11 | 1.15e-10 |

The recorded metric diagnostics describe the value/J/K plan and exclude force
staging. They are not a whole-endpoint peak-memory measurement. This slice
does not complete #283: generated resident/low-memory response optimization,
complete resource evidence and integration measurements remain.

Validation includes the four real-GPU native suites (direct HF, DF numerical
and gradient reference checks, CUDA Fock provider and composition), the paired
force probe and the independent GPU4PySCF matrix. No new equations require a
duplicate implementation-mirroring test.

Restore the raw records into a new directory, from the repository root:

```bash
python -m tools.unpack_evidence benchmarks/results/issue283-host-response \
  --output build/issue283-host-response-restored
```

Member hashes, archive hash and source commit are in
`raw-evidence.manifest.json`; restoration and direct byte comparisons passed.
Commands embedded in the probe execution metadata and matrix manifest used
finite `srun --partition=main --gres=gpu:5090:1` allocations. To reproduce the
probe, run `benchmarks/issue206_df_force_probe.py --repeats 3 --library LIBRARY
--output NEW_JSON` inside such an allocation, once per frozen source/library.
Add `--component-trace-dir NEW_DIR` only for the separate diagnostic run.
Run `benchmarks/issue206_df_matrix.py --run --repeats 5 --library LIBRARY
--output-dir NEW_DIR` in a 25-minute allocation with the pinned benchmark
environment. Keep `PYTHONPATH` pointed at the measured source's `python/`.
