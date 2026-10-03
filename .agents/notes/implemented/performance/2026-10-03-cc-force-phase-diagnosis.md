# Decision: measure completed CC force phases before changing derivative scheduling

Status: implemented
Date: 2026-10-03

## Problem

After complete ERI source orbit folding, the 28-AO prepared CCSD(T) force endpoint
still takes about 13.90 seconds warm. CUDA response execution does not imply that
the triples response is on CUDA. Neither the CPU triples cost nor the visible
per-shell allocation pattern establishes the largest remaining opportunity.

## Decision

Expose optional completed observations through `GENERATIVEQC_DF_PROGRESS_TRACE`.
Reuse the force owner's existing timers and add conditional derivative callback
and consumer timers. No additional CUDA synchronization is introduced. Disabled
traces perform no additional clock reads. Keep the integral equations, primitive
record schedule, stage budgets, output semantics, and force admission unchanged.

The force summary has five disjoint intervals: triples response, Lambda and
parameter weights, raw Hamiltonian, orbital response, and nuclear derivative.
The raw provider and source-read intervals nest inside the raw Hamiltonian.
CUDA source-read time is submission time, not completed integral time. The
triples response backend is explicitly CPU when triples are included.

The derivative scope has four disjoint intervals: rank-two pullback,
one-electron/nuclear-repulsion work, shell callbacks, and remaining AO pullback,
scatter and bookkeeping. CUDA primitive consumer time nests inside the shell
callbacks and includes the existing stream allocation, transfer, execution,
synchronization, and destruction. Record construction is outside that consumer
interval. Publish actual record and consumer counts, not estimates from time.

The completed force scope's own elapsed time only measures journal emission.
Its recorded phase values describe prior execution. The existing completed RHF
scope similarly uses `cuda_completed`; teach the journal reader to accept and
preserve that mode instead of rejecting valid mixed CC/RHF traces. Do not treat
its short emission interval as the CUDA execution interval.

## Evidence

`benchmarks/results/cc-force-phases-20261003/` retains complete endpoint records,
phase observations, independent gates, selected Nsight aggregates, provenance,
and the patch reconstructing the measured native sources on the parent revision.
Later journal-reader/tests/notes changes do not alter that native library.

n1 Slurm job 5453, RTX 5090, CUDA 12.9.1 sm_120 Release, one GPU, 15-minute limit,
one CPU thread per numerical library, 8-GiB endpoint budget. Four calls per case:
cold, warm, warm-repeat, and atom 1 z displaced by 0.01 Bohr. Warm means the mean
of the two unchanged-geometry repeats; preparation is outside the endpoint.

| CUDA case | Complete energy + forces | Triples response | Lambda/parameters | Raw Hamiltonian | Orbital response | Derivative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 AO | 1.211 s | 0.076 s | 0.486 s | 0.029 s | 0.018 s | 0.441 s |
| 28 AO | 13.895 s | 2.644 s | 2.569 s | 0.468 s | 0.483 s | 6.181 s |

The force phase sums exclude the preceding energy endpoint and admission/cleanup.
Within the 28-AO derivative, shell callbacks take 6.079 s, their primitive
consumers 5.929 s, and AO pullback/scatter only 0.100 s. There are 22,155 canonical
shell quartets, 7,137,963 primitive records, and 22,210 consumer calls. The extra
55 calls arise from four-p-shell quartets exceeding the 4,096-record buffer.
Each endpoint uploads 1,484,696,304 primitive bytes and downloads 2,309,840 result
bytes. The 14-AO case has 1,540 quartets, 516,132 records, and 1,546 consumers.
These are executed counts; no screening or integral reduction is claimed.

A separate n2 Slurm job 2116 used one RTX PRO 6000 under the same finite time
limit. Nsight independently observed 88,840 weighted derivative kernels across
four complete 28-AO calls, exactly matching four times the consumer census.
It measured 20.682 s of accumulated derivative kernel duration; 37,264 launches
used only two blocks of 64 threads. The entire four-endpoint CUDA malloc/free API
duration was 0.787 s. Host API and device times overlap and must not be added.
Nsight's device-event tracing warning and shared-node conditions mean these
profile timings are diagnostic, not a comparison with n1 or a speedup claim.

All twelve n1 endpoint calls passed the independent PySCF 2.14 cluster oracle:
energy < 3e-9 Eh, triples < 2e-9 Eh, force < 1e-6 Eh/Bohr, response residual < 1e-9.
Maximum observed errors were 1.43e-12 Eh, 1.06e-14 Eh, and 6.67e-8 Eh/Bohr.
The four profiled n2 calls independently pass the same gates. Public CPU/CUDA
CCSD/CCSD(T) tests pass (40 cases), journal-reader regressions pass, and all
modified post-HF translation units compile with CUDA disabled. Compiler launcher
commands and ccache snapshots are retained with the evidence.

## Consequences and next scheduling experiment

Prioritize bounded multi-quartet primitive batches: the consumer already accepts
multiple output tiles, while the current caller submits one quartet at a time.
Retain distinct four-center derivative slots until scattering to physical atoms,
including repeated atoms. Charge caller records, tile-to-atom metadata, host/device
results, and reused upload capacity to the owning stage. Preserve an explicit
small-budget path; never collect all quartets or a rank-four derivative tensor.

Workspace reuse alone cannot remove the dominant underfilled kernel schedule.
Porting the native triples response to CUDA remains useful but has a smaller
measured ceiling here. Accelerating the AO pullback first is not justified by
its measured 0.100-second contribution. Revisit these priorities after complete
endpoint measurements of a qualified batching change, not after a microbenchmark.

This diagnosis does not broaden force support beyond 28 AOs. The 56-AO degenerate
same-space response still requires the separately documented scientific solution.
