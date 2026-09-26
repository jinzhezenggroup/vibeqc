# Source-bound ECP projector review qualification

This fresh RTX 5090 record supersedes the incomplete provenance of the
historical RTX 4090 measurements for the merge qualification gate.

- Candidate: `5d7eb81d86aa9be0a3b6f0b9e1462b645aec1046`, **dirty: false**,
  Slurm job 9624.
- Legacy-projector baseline: `5129a8db666efcf3dedacc8a24f2dbf205b96bd7`,
  **dirty: false**, Slurm job 9629.
- CUDA 12.9.86, sm_120, Release, AOT shells disabled; one host BLAS thread.
- Every GPU command used a finite `main` allocation with `--gres=gpu:5090:1`
  and preserved Slurm device visibility.

The baseline includes unrelated KS final-state work; these conventional
RHF/UHF ECP endpoints execute neither DFT nor density fitting. Full source-file
hashes, binary/build identities, command vectors, raw samples and kernel
resource dumps are retained for both runs. Only untracked CMake output under
`build-review/` was excluded when recording source dirty state.

| Endpoint | Baseline warm median / ms | Generated warm median / ms | Energy error / Eh | Force error / Eh/bohr |
| --- | ---: | ---: | ---: | ---: |
| RHF | 1020.313 | 949.274 | 2.66e-15 | 6.08e-13 |
| UHF | 1017.608 | 943.216 | 3.33e-16 | 8.12e-16 |

Three synchronous warm samples per endpoint are retained. These independent,
sequential small-fixture timings are a regression check, not a statistical
speedup or default-selection claim. All complete planned host/device peaks
match between baseline and candidate. Matrix error against independent
Libcint is 3.47e-12 Eh; all radial powers, s/p/d projectors,
center derivatives and value-only cases pass their independent tests.

Candidate validation: **53 Python tests passed**, both native ECP tests passed,
and Compute Sanitizer memcheck reported **0 errors**. The original CPU ECP
implementation remains the independent numerical oracle/fallback.

Restore and verify the archive with `python -m tools.unpack_evidence
benchmarks/results/ecp-projector-171/review-qualification --output /tmp/ecp-review`.
The restored `native_validation.py` records provenance and runs the checks.
After building each recorded source with the flags above, invoke it inside
`srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00`,
passing the source path, a new output path, and `ecp` (or `ecp-baseline`).
Adjust its Python environment path for reproduction.

This validates the bounded radial/projector migration. Broader ECP domains
and remaining issue #171 work stay open. No historical dirty state was guessed.
