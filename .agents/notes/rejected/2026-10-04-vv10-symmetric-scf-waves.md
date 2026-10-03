# Decision: reject serial-wave triangular VV10 SCF contraction

Status: rejected
Date: 2026-10-04

## Problem and candidate

Full-grid VV10 contributes to every WB97M-V SCF iteration. A private prototype
evaluated each unordered pair once and accumulated both output legs, seeking
to reduce dense pair work by roughly half. This is distinct from the previously
rejected partner-lane/shared-partner staging experiments.

The second leg was extracted from the canonical geometry TensorIR DAG, not a
separately handwritten derivative formula. Serial waves of i tiles gave each
j CTA unique ownership of its row; an ordered drain completed the i rows.
The scheme used no atomics, deterministic reductions and O(N) extra scratch.
Positive-zero-weight rows retained the original ordered evaluation; negative
zero retained the original masking semantics. Only SCF features were tested;
production geometry and endpoint paths were untouched.

## Decision

Do not promote or spend a full endpoint build on this schedule. Reducing pair
count did not reduce execution time: the 32-point tile took about 3.19 times
the ordered baseline at 320,000 points, and the 16-point tile took about
9.61 times the baseline. Both also failed the unchanged large-array serial
agreement gate. Keep the production ordered kernel and its existing bounds.

## Evidence and limits

Finite n1 Slurm 5654 used an RTX 5090, CUDA 12.9.1, sm_120 and -O3. The
allocation completed with exit code zero, meaning the diagnostic collection
completed, not that the retained large numerical failures passed.

The small qualification covered 117 size/domain/tile combinations, deterministic
replays and guard canaries. Eighteen independent 90-digit Decimal energy
finite-difference checks passed. Memcheck and synccheck repeated the small
qualification with zero reported errors.

CUDA-event timing included zeroing, every wave, drains and finalization, with
three rotated-order observations per size/tile. Allocation and H2D were outside
this timer. These are synthetic component timings, not molecular endpoints:

| Points | Ordered median (s) | Triangle 16 median (s) | Triangle 32 median (s) |
| ---: | ---: | ---: | ---: |
| 20,000 | 0.015343616 | 0.126412415 | 0.048417728 |
| 80,000 | 0.207905212 | 1.901021118 | 0.652838745 |
| 320,000 | 3.103060059 | 29.815593750 | 9.900716797 |

The large serial comparison kept rtol=2e-12 and atol=2e-11. At 80,000 points,
tiles 16/32 failed 4/3 output fields; at 320,000 they failed 461/464. No tolerance
was relaxed and no observation was excluded. There was no independent
large-array high-precision oracle, so this does not establish which reduction
is closer to the exact result.

At 320,000 synthetic all-nonzero points, ordered/triangular algorithmic pair
counts are 102,400,000,000 / 51,200,160,000. These are counts for the constructed
dense domain, not measured molecular active-pair counters. Extra algorithmic
scratch is bounded by 8*(6*N+96), or 15,360,768 bytes at that size. The harness
allocated this scratch for every variant, so the formula is not a process-peak
memory measurement. Tiles 16/32 used 40 registers, 12,288/49,152 shared bytes and
no spills; the ordered kernel used 58 registers and no shared memory.

Verified ccache 4.5.1 was invoked, with command and before/after statistics
retained. The combined compile/link command was counted as uncacheable, not a
hit. Any future probe must split cached compilation (`ccache nvcc -c`) from
linking; this run's statistics must not be presented as cache reuse.

## Provenance and revisit conditions

Ignored `.artifacts/vv10-symmetric-scf-probe-20261004/` retains the generated
source, generator, schedule, runner, raw checks/timings, sanitizer logs, Slurm
receipt, compiler command, cache statistics and hashes. The prototype was made
from integration f5287b355, native input identity
`ebdf07921464440e085b2925a1bd061ba9090788485d1cab01e3cada7122814c`.

- Canonical geometry program:
  `56d9a50e9d50aed17edcdfad06a961edc8004e0fe43161c070657a6fc02eea30`.
- Symmetric program:
  `67ee0d85fb528b957e8a39d8dcea46e7b4981f876bad7a0b891b5add920a1dc4`.
- Generated probe source:
  `381787955470ad60abbb28fc3a14e8325d7c35a5836da48b41f0accce8f7e680`.
- Probe binary:
  `904847883588fac7b0e8b8e72792331b757d5875801f8827ed2db65602aecb4c`.

Revisit symmetry only with a materially different bounded ownership/reduction
scheme that first passes independent numerical checks and wins the complete
component schedule. A halved mathematical pair count alone is insufficient;
any successful successor still requires unchanged molecular E/F gates and
complete cold/warm endpoint timing before promotion.
