# Issue #367: CUDA RI-MP2 resident contractions

This evidence covers the production CUDA RI-MP2 energy path introduced for #367.
The CPU RI implementation remains the numerical oracle. Measurements were taken
on node3 with one NVIDIA GeForce RTX 5090 (32,607 MiB), driver 580.95.05 and
CUDA 12.9 (Build cuda_12.9.r12.9/compiler.36037853_0). CPU BLAS/OpenMP threads
were fixed to one.

The benchmark is benchmarks/ri_mp2_cuda_endpoint.py. It uses one H2O geometry
in bohr and the orbital basis as the auxiliary basis, measures a fresh endpoint
and complete same-calculator replays, and then captures one separate CUDA
component trace. Trace/progress instrumentation is never pooled into the clean
endpoint timings.

The clean timing record in results.json contains three replay samples and was
captured immediately before the final resource-accounting-only correction. That
correction changed planner accounting/diagnostics, not the full-resident compute
kernels or dataflow used by these three cases. A later final-source five-repeat
attempt was rejected as performance evidence because an unrelated issue #370
Slurm job held the only RTX 5090 at 100% utilization; that raw run is retained as
resource_summary_final.json only for final-source byte/work counters; timing fields are omitted.

| basis | AO/aux | CPU cold (s) | CUDA cold (s) | CPU replay median (s) | CUDA replay median (s) | replay speedup | |Ecpu-Ecuda| (Eh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STO-3G | 7/7 | 0.0333 | 0.1895 | 0.0315 | 0.0241 | 1.31x | 3.41e-13 |
| def2-SVP | 25/25 | 0.3265 | 0.0968 | 0.3265 | 0.0839 | 3.89x | 6.28e-12 |
| def2-TZVP | 48/48 | 3.4238 | 0.2181 | 3.4189 | 0.2117 | 16.15x | 3.25e-11 |

All three CUDA runs select one resident B block and one source pass. The clean
raw timing JSON, all three replay samples, energies and component records are in
results.json. Final-source resource/work counters are retained separately in
resource_summary_final.json; it deliberately contains no wall-clock timings.

## CUDA correlation component ledger

The clean component trace is a separate execution from the timing repeats. Its
outer host scope and CUDA stream event span agree closely; the nested AO-to-MO
scope includes generated raw three-center evaluation, public-basis transform,
metric whitening and the two cuBLAS AO-to-MO contractions.

| basis | RI correlation host scope (ms) | AO-to-MO stream span (ms) | g/x + energy span (ms) | source passes / row generations | AO-to-MO GEMMs | energy GEMMs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STO-3G | 10.437 | 10.253 | 0.043 | 1 / 7 | 8 | 5 |
| def2-SVP | 51.917 | 51.682 | 0.094 | 1 / 25 | 26 | 5 |
| def2-TZVP | 143.221 | 142.915 | 0.166 | 1 / 48 | 49 | 5 |

The stream spans are not GPU-utilization percentages: a CUDA event interval can
include submission gaps. They demonstrate that the old host AO-to-MO and
auxiliary-Q loops are absent from the correlation critical section. The complete
public endpoint timings above include RHF/reference work and are the performance
acceptance numbers.

| basis | H2D bytes | D2H bytes | retained B bytes | plan device bytes | RI working device bytes | planned RI peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STO-3G | 896 | 468 | 560 | 2,131,837 | 2,760 | 3,775,996 |
| def2-SVP | 10,400 | 5,220 | 20,000 | 2,506,855 | 146,616 | 4,426,022 |
| def2-TZVP | 37,632 | 18,836 | 82,560 | 3,592,515 | 986,760 | 6,101,802 |

D2H consists of the metric/source diagnostic plus the final 16-byte OS/SS
result; transformed B is never downloaded. H2D consists of the metric/scales and
orbital coefficients/energies. The exact counter equality
ri_mp2_transfer_bytes = ri_mp2_h2d_bytes + ri_mp2_d2h_bytes is regression-tested.

## Metric preparation and bounded fallback

CUDA RI-MP2 reuses the existing CUDA DF metric plan rather than implementing a
second eigensolver. The same relative threshold, cuSOLVER eigensystem and
inverse-square-root construction therefore define the Hamiltonian. A separate
def2-SVP diagnostic run with VIBEQC_DF_PROGRESS_TRACE measured the correlation
source setup at 1.415 ms and metric factorization at 0.601 ms; that tracing mode
adds fences, so these values are preserved only in progress_summary.json and are
not mixed with the clean timing table.

The B planner is independently regression-tested with nbf=120, occupied=20,
virtual=100, auxiliary=180 and 2 MiB of fixed state. A 64 MiB allowance selects
the full 100-virtual resident B. An 8 MiB allowance selects virtual_block=27,
j_batch=4 and an 8,364,608-byte peak; a 2 MiB allowance rejects before a B block
can be admitted. The public CUDA RI route also no longer applies the CPU
complete-three-center preflight: H2/STO-3G succeeds at a 12 MiB correlation
budget while the CPU oracle keeps its existing complete-tensor admission rule.

The bounded path retains two device B block buffers and repeats source rows only
when another virtual block is required. Its trace reports virtual_blocks,
source_passes and source_row_generations so work amplification remains visible.

## Numerical and regression gates

- Four independent MP2 fixtures (H2, H2O, LiH and an f-shell HeH case) pass on
  both CPU RI and CUDA RI: 8/8 cases at the existing 1e-11 absolute / 1e-10
  relative OS/SS component gates and 1e-9 Eh total-energy gate.
- CUDA memory/error/public-path regressions, the resident-B trace regression and
  the updated RI fixture suite pass together (11 targeted Python tests).
- The complete public MP2 Python suite passes 37/37, and the CPU native MP2
  contract suite passes.
- The native CUDA MP2 status suite passes, including full-resident, blocked-B and
  impossible-budget planner cases plus the pre-existing CUDA force/OOM checks.
- Both CUDA sm_120 and non-CUDA shared libraries build and link.

The implementation branch is based on master
d32caac5fc14810188e5fa33586f1b6cb51078b5. The final commit hash is the
authoritative source identity. results.json preserves the uncontended timing
record; resource_summary_final.json preserves final-source resource/work
counters without contaminated timing fields; progress_summary.json preserves the
separately instrumented metric preparation ledger.
