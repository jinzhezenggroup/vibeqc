# Issue #598 real-NVIDIA DF resource-policy qualification

This retained archive closes the real-device qualification gate for PR #614.
The measured production source is exact commit `8b2b90ed8c5b161259d7d1b542716cb9c3d2f821`. The final qz/Inspire job
`vibeqc-598-final-8b2b90ed` completed successfully on `NVIDIA GeForce RTX 4090, GPU-39c03f0b-bcec-70ba-3bbb-44c0c8461f41, 595.71.05, 49140 MiB` with a CUDA `sm_89` build.

Raw persistent evidence is retained at `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0598-qual-final-8b2b90ed`. `qualification-summary.json` binds
this compact archive to the raw qualification/build logs, exact runner, and every
cold/warm trace/progress journal by SHA-256.

## Endpoint results

All four endpoints use public DF budget `0`, so they exercise the automatic
workload/device policy. Constrained arms create real CUDA allocations before
prepare; roomy arms allocate an additional 2 GiB after prepare to prove warm
replay does not re-probe live free memory.

| Endpoint | Envelope | Resolved total | Value cap | Response cap | Observed free | Reserved headroom | Value-plan peak | Response scratch | max abs(dE) | max abs(dF) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RHF water-tetramer-def2-svp-spherical | roomy | 63,209,472 | 46,858,240 | 16,351,232 | 50,438,537,216 | 6,358,310,912 | 40,281,841 | 14,758,992 | 3.695e-12 | 9.664e-12 |
| RHF water-tetramer-def2-svp-spherical | constrained | 56,764,416 | 42,080,413 | 14,684,003 | 6,433,996,800 | 6,358,310,912 | 40,281,841 | 14,611,536 | 3.695e-12 | 9.666e-12 |
| UHF oh-def2-svp-spherical-uhf | roomy | 33,554,432 | 22,450,672 | 11,103,760 | 50,321,096,704 | 6,358,310,912 | 3,306,915 | 141,852 | 3.268e-13 | 6.237e-11 |
| UHF oh-def2-svp-spherical-uhf | constrained | 30,025,728 | 20,089,679 | 9,936,049 | 6,398,345,216 | 6,358,310,912 | 3,306,915 | 141,852 | 3.268e-13 | 6.237e-11 |

For every endpoint, resolved total equals value+response split, the value-plan
peak stays within the resolved value cap, force-response scratch stays within
the resolved response cap, and reserved headroom does not exceed observed free
memory. Both constrained arms reduce the automatic budget while preserving the
independent PySCF energy/force gates.

For each roomy arm, actual free memory is reduced by roughly 2 GiB after
prepare. Cold, warm1, and warm2 retain byte-identical resource-policy records,
directly demonstrating frozen prepared-plan identity rather than a second
live-memory probe.

Independent PySCF 2.14.0 DF energies and analytic forces
(`auxbasis_response=True`) are generated fresh before CUDA execution. The
retained machine-checked `qualification` object requires the live CUDA probe,
RHF/UHF force endpoints, roomy/constrained envelopes, bounded value/response
memory, frozen warm replay, and independent numerical gates.

## Reproduction boundary

`qualification_runner.py` is the exact runner used by the successful final-head
campaign. It reuses VibeQC existing DF progress journal, metric diagnostics,
component trace, and independent `cpu_reference`; it does not create a second
resource-policy, cache, or autotuning framework.
