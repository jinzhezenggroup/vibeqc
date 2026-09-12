# True batch energy-only matrix (#282)

`PreparedBatch.execute(properties=("energy",))` omits all response work in both
the native fleet and the paired GPU4PySCF benchmark. Five interleaved warm
samples per engine/case on one Slurm-allocated RTX 5090 give these medians:

| AO / batch | VibeQC energy (s) | GPU4PySCF energy (s) |
|---|---:|---:|
| 96 / 1 | 0.232582 | 0.064248 |
| 96 / 4 | 0.911080 | 0.259372 |
| 192 / 1 | 3.231899 | 0.077399 |
| 192 / 4 | 12.984029 | 0.307123 |

Every stored warm pair passes the 1e-9 Eh energy gate (maximum error
2.786e-11 Eh). All force values/errors are null. Cross-engine iteration
branches differ, so these are labeled endpoint measurements rather than
iteration-matched speed claims. The declared 1-GiB DF sub-budget retains the
conservative force-capacity allowance. Value-plan diagnostics are preserved
but do not measure total peak memory or opaque CUDA library retention.

The frozen native library is from implementation 2084eee85; SHA-256 is
`e2d22094d9a99bdd3660e7f140f1de6c1382215b4f0ad15bfe739deb3f43faa0`.
It precedes response-panel cache integration, which does not execute in this
energy-only endpoint. The archive preserves 19 exact files: raw matrix,
commands, source patch, CMake cache and validation logs. An initial GPU run
detected unwanted nuclear derivatives; its failed log is preserved alongside
the passing fix verification (116 GPU tests plus four CUDA native suites).
CPU validation includes 17 native suites and 74 Python tests, with 14 optional
skips. All hooks pass. Every archive member was restored and byte-compared.

```bash
python -m tools.unpack_evidence benchmarks/results/issue282-energy-only \
  --output /tmp/issue282-energy-only
python benchmarks/results/issue282-energy-only/audit.py
```

The archive's matrix manifest records the finite Slurm job and complete
reproduction commands. `summary.json` records all-pair numerical gates,
timing branch metadata, resource diagnostics and archive identity.
