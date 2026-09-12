# Generated raw response panel reuse (#283)

On one Slurm-allocated RTX 5090, retaining the response's current raw panel
reduces full-panel generation from three passes to one with the same scratch.
Five fresh energy/energy-plus-force pairs per binary and case, interleaved in
ABBA order without concurrent compilation (job 9349), give:

| AO | Before endpoint (s) | After endpoint (s) | Speedup | Before force increment (s) | After (s) |
|---:|---:|---:|---:|---:|---:|
| 96 | 1.321514 | 1.015272 | 1.3016x | 0.727010 | 0.422020 |
| 192 | 9.094814 | 7.474947 | 1.2167x | 4.560533 | 2.959096 |

These are medians of fresh solves. Iterations match before/after (34/39);
energies are identical and maximum force change is 9.60e-14 Eh/bohr. The
earlier job 9346 overlapped compilation and is retained as preview evidence.

Executed response traces reduce raw AO slice generation 288→96 and 576→192;
scratch stays 14,758,992 and 115,632,272 bytes. The 192-AO response takes
3829.993→2211.223 ms, with raw generation 2462.331→852.578 ms. Named host
components and measured synchronization explain 99.27%/97.37% of its force
increment. Initial 96-AO pairs overattribute the increment because energy-only
initializes libraries first. Their records remain in the archive; the second
warmed diagnostic pairs give valid 98.84%/98.89% coverage. Instrumented
component timings and uninstrumented endpoint medians are separate probes.

The final 1-GiB declared-budget force matrix includes five warm repeats per
engine at each size. VibeQC medians are 0.795468/2.533044 s at 96 AO, batch 1/4,
and 6.372424/24.969428 s at 192 AO. Every raw pair passes 1e-9 Eh energy and
1e-8 Eh/bohr force gates. Cross-engine iteration branches differ, so these
measurements do not establish an iteration-matched GPU4PySCF speed comparison.
Value-plan peaks exclude response staging and opaque library retention.

The 192-AO fixed-density J/K plus complete two-electron gradient takes
2.187696/9.866391/98.251761 s at 512/128/32 MiB. The full transformed tensor is
56,623,104 bytes and cannot fit 32 MiB. The constrained results agree with
resident J/K/gradient within 4.58e-16/9.55e-18/5.43e-19. This probe is not a
total SCF force or a passing low-memory full-SCF timing.

Native CUDA suites (4), GPU resource/derivative tests (51), host/ownership
tests (13), hooks and memcheck (zero errors) passed at checkpoint 2ca01977.
The cache preserves raw values, metric discarded-space response and all
derivative terms. Exact library hashes, reconstruction patches, CMake caches,
commands, raw samples, traces and validation logs are archived. Restore into a
new directory with:

```bash
python -m tools.unpack_evidence benchmarks/results/issue283-response-panels \
  --output /tmp/issue283-response-panels
python benchmarks/results/issue283-response-panels/audit.py
```

The archive contains 133 files; every member was hash-checked, restored and
byte-compared. `summary.json` records the compressed archive identity and
limitations. Remaining integration work is tracked in #282, #283 and #284.
