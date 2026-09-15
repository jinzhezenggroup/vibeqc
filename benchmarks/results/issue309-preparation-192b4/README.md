# #309 192-AO batch-4 preparation ablations

Five interleaved baseline/candidate pairs per clean endpoint on n3's RTX 5090,
submitted through Slurm job 9600. The source is exact `master` `b4a18af5` with a
Release CUDA 12.9.1 / sm_120 build, AOT disabled, RHF spherical def2-SVP and
the matching auxiliary basis under a 1 GiB DF allowance. Separate profiled
runs retain the actual host/core/overlap call traces; their timings are not used below.

| AO / batch | Candidate | Clean warm endpoint | Baseline (s) | Candidate (s) | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 192 / 4 | lazy-core | energy | 0.396184 | 0.346756 | 1.143 |
| 192 / 4 | lazy-core | force | 9.728078 | 9.674247 | 1.006 |
| 192 / 4 | overlap-cache | energy | 0.397341 | 0.332073 | 1.197 |
| 192 / 4 | overlap-cache | force | 9.730552 | 9.657290 | 1.008 |
| 192 / 4 | combined | energy | 0.396251 | 0.281664 | 1.407 |
| 192 / 4 | combined | force | 9.725850 | 9.603197 | 1.013 |

All nine runner invocations passed. SCF iteration/retry branches match, energy
and complete-force replay gates pass at 1e-9 Eh and 1e-8 Eh/Bohr, and the
profiled runs retain cold construction, unchanged replay, changed-geometry,
and restored-geometry samples. The energy endpoint shows significant gains for
lazy-core, overlap-cache and their combination; force timings are dominated by
the force endpoint and remain below the runner's 2% significance threshold.
Changed-geometry samples are retained so cache-rebuild cost is not hidden.

This closes the missing 192-AO/batch-4 timing domain of #309. It does not close
#309: 384-AO batch 1/4, constrained-memory and broader spin/representation
performance domains remain. It also does not make an independent GPU4PySCF parity claim.

The raw archive contains every result/manifest, original JSONL trace text and
`reproduction.sh`. `raw-evidence.manifest.json` binds the archive and every
member by byte count and SHA-256. Restore and verify with:

```bash
python -m tools.unpack_evidence benchmarks/results/issue309-preparation-192b4 \
  --output /tmp/issue309-preparation-192b4-evidence
```

Reproduction requires the source commit and library identity recorded in
`summary.json`; execute the restored script through a finite RTX 5090 Slurm
allocation rather than directly occupying the login node.
