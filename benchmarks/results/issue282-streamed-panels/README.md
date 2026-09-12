# Generated residency, streamed J/K and preparation capacity (#282)

The complete 96/192-AO batch-1/batch-4 force matrix now completes with a
1 GiB declared DF sub-budget. All four value plans retain transformed data.
Previously the 192-AO cases failed before SCF because one-electron preparation
reserved an unused direct-ERI quartet table. Matrix-only packing and quadratic
preflight accounting remove that false rejection.

| AO / batch | VibeQC warm median (s) | GPU4PySCF warm median (s) | maximum energy error (Eh) | maximum force error (Eh/bohr) | value-plan peak bytes |
|---:|---:|---:|---:|---:|---:|
| 96 / 1 | 1.107352 | 0.244536 | 6.42e-12 | 3.53e-11 | 33,515,540 |
| 96 / 4 | 3.777844 | 0.977027 | 5.92e-12 | 3.81e-11 | 62,345,312 |
| 192 / 1 | 7.987801 | 0.330866 | 9.33e-12 | 7.84e-11 | 241,760,732 |
| 192 / 4 | 31.535691 | 1.332272 | 2.79e-11 | 7.19e-11 | 441,898,880 |

Five interleaved repeats per engine preserve fixed post-cold density snapshots.
Every cross-engine iteration branch remains unmatched. These endpoints do not
establish a matched speedup, and generated response remains slower than the
optimized compatibility route. Value-plan peaks exclude force staging and
opaque library retention; 1 GiB is a declared allowance, not a measured total
peak. Metric ranks, condition numbers, convergence and all arrays are archived.

Earlier matrix commands did not set the comparison tool's optional numerical
error limits. `summary.json` therefore records an independent all-repeat audit
of the original arrays at 1e-9 Eh and 1e-8 Eh/bohr, including the prior host
response and 8-GiB resident matrices. All pass, with errors much smaller than
those limits. Run `audit.py` here to reproduce the new matrix audit directly
from its archived arrays; exit status alone is not the numerical evidence.

The fixed-density 192-AO probe compares resident storage and two streamed
allowances using the same symmetric density. It measures J/K plus the complete
two-electron derivative, not the total SCF force:

| DF allowance | resident | J call (s) | K call (s) | J/K + E2 derivative (s) |
|---:|:---:|---:|---:|---:|
| 512 MiB | yes | 0.016799 | 0.003626 | 3.826 |
| 128 MiB | no | 1.451 | 5.017 | 11.138 |
| 32 MiB | no | 1.609 | 86.196 | 99.154 |

Maximum resident/streamed errors are below 4.58e-16 for J, 9.55e-18 for K,
and 4.75e-19 Eh/bohr for the E2 gradient. The full transformed tensor alone
requires 56,623,104 bytes, exceeding the 32-MiB allowance. J uses two raw
passes (113,246,208 logical bytes), with no transformed tile production. At
32 MiB, K produces 192 transformed panels and reuses each for its second GEMM.
At 128 MiB it produces seven panels using 49 raw generations. Executed-stream
traces expose the remaining recomputation cost; capture-only records are not
replay measurements.

The first skinny raw-GEMM experiment was cancelled after producing 36,864 raw
launches per K build. The fused skinny fallback restores a workable fixed-density
path, but the corresponding 32-MiB full SCF still timed out at ten minutes.
Both attempts are retained and are not passing total-energy/force gates.

The fixed-density binary is `578fe16eaf78ea94ee1b65828398c3fd032d1bc82dedd270b58b289ba26b26b6`.
The final preparation/matrix binary is
`58a79cc40d189f092c21d0a5aa2fbb4f02232d3bd56a1c6f97d4f12ec59b5961`.
Native changes are committed in `7e28059` atop resident checkpoint `787304d`;
the archive retains the exact pre-commit patches, untracked helper for older
experiments, and final CMake configuration. Binaries remain locally frozen;
reconstruction source and SHA-256 identities are the retained provenance.

Validation: four CUDA native suites, 51 GPU resource/DF derivative tests,
50 GPU one-electron tests, 121 host checks and all hooks passed. The prior
streamed checkpoint also passed compute-sanitizer memcheck with zero errors.
All GPU work used finite Slurm jobs with scheduler-provided device visibility.
The first final-matrix attempt loaded an older cuSOLVER through GPU4PySCF and
failed before SCF; its logs are retained. The successful rerun explicitly used
`/group/software/cuda-12.9.1/lib64` first in `LD_LIBRARY_PATH`, with
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.

Forty original files (21,475,127 bytes) are stored in a 3,166,222-byte archive.
All restored members were compared byte for byte. The exact-hash evidence-policy
exception covers this compressed archive because the full parity arrays exceed
the ordinary 1-MiB review guard even after compression.

```bash
python -m tools.unpack_evidence benchmarks/results/issue282-streamed-panels \
  --output build/issue282-streamed-panels-restored
python benchmarks/results/issue282-streamed-panels/audit.py
```

Use a fresh restore directory. The matrix manifests preserve the exact endpoint
commands; the full driver used `--memory-budget-bytes 1073741824 --repeats 5`
inside a 30-minute allocation. `df-fixed-panel-probe.py` is also archived.
Energy-only batch measurements, generated response optimization and CUDA
occupied-factor K remain separate integration work; #282–#284 remain open.
