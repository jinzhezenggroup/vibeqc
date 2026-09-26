# Decision: one packed fitted DF value owner

Status: implemented
Date: 2026-09-25

## Problem

At 768 AOs and 3712 auxiliaries, the complete dense DF value tensor alone
requires 17,515,413,504 bytes. The existing packed resident representation
retains **both** raw and fitted lower-pair tensors, about 17.54 GB together
before metric, occupied factors, DIIS, solver and response scratch. The
automatic 96-atom value allowance observed on one 5090 at source
`fe534ebf` was 13,685,173,124 bytes. Its bounded streamed SCF consequently
regenerates raw three-center integrals for each J/K call; in the stopped trace
raw generation took
56.76 of 62.52 seconds of completed J/K scopes. This is a data-ownership and
semantic-work problem, not a reason to tune the raw kernel alone.

## Decision

An **explicit** `VIBEQC_DF_VALUE_STORAGE=packed-single` option retains only
the fitted lower-pair B tensor (up to 8,769,110,016 bytes at this shape).
Preparation generates raw lower-pair slices once, transforms each slice into
its permanent B allocation with the same metric eigensystem, and reuses the
existing bounded J/K scratch. SCF J/K consumes B without generating raw
source tiles on every iteration. Force response has no raw owner to borrow:
it must regenerate physical raw and derivative values from the matching
immutable source within its **own** response allowance. Exact final-state and
borrowed occupied-response leases requiring resident raw remain disallowed.
An explicitly selected source-projected occupied response instead validates or
reconstructs canonical factors and regenerates raw within the force budget;
automatic force response continues to borrow B after measuring a regression
for that source-projected alternative. Keep
ordinary double-owner packed and bounded streamed routes unchanged.

## Rejected alternatives

- Raising the 13.69 GB value allowance to 17 GB cannot fit a dense B tensor
  plus complete live scratch. The existing dense admission floor for this
  shape is about 74.7 GB.
- Reusing one raw tile across a J/K call reduces some generation but still
  repeats source work on every SCF iteration and replay.
- Pretending fitted B is a raw owner would corrupt metric reverse response;
  force must take the explicit source-regeneration route.
- Automatically source-projecting occupied force response at 768 AOs reads raw
  only once but costs about 110 s; borrowing B takes about 38–59 s depending
  on the response budget. Semantic raw-pass count alone is insufficient.

## Invariants

- The selected representation is part of prepared-plan/cache identity; a
  mode change cannot replay the previous owner.
- The native planner and actual allocator agree on one or two immutable
  tensors and on the same projection/DIIS reservations.
- Under the explicit single-factor selector, optional automatic occupied U
  may be dropped when B fits but U does not; an explicit occupied request
  continues to fail rather than silently changing the requested algorithm.
- Absent raw storage must not authorize a final-K, force-response or source
  lease that was qualified only for the dual-owner path.
- Unrepresentable shapes and capacity exhaustion fail explicitly; packed
  materialization cannot silently interpret arbitrary host tensors as a
  physical symmetric source.

## Evidence and remaining gates

The shape-only capacity and 96-atom planner tests pass: with a 13.69 GB
value budget and no source metadata, the single-factor owner needs
13,138,165,039 bytes without U; the rank-160 owner needs 16,192,666,939
bytes at Q=128. Full, single-GPU 96-atom def2-SVP/cc-pVDZ-JKFIT comparisons
against GPU4PySCF, 40 host threads, both cold and warm forces, passed
`1e-8 Eh / 1e-7 Eh/Bohr` gates:

| Selector/budget | VibeQC cold/warm | GPU4PySCF cold/warm | Cold/warm iterations (VibeQC; GPU4PySCF) | Peak value-plan device bytes |
| --- | --- | --- | --- | --- |
| Single B, automatic split | 174.34 / 59.72 s | 58.15 / 10.19 s | 20 / 3; 45 / 1 | 11,887,978,601 |
| Single B, explicit 26 GB total | 83.72 / 65.53 s | 57.03 / 10.19 s | 24 / 5; 44 / 1 | 14,942,480,501 |
| Single B, automatic total, 5 GB response | 105.43 / 68.34 s | 57.05 / 12.35 s | 24 / 5; 44 / 3 | 14,942,480,501 |
| Explicit source-projected force, 5 GB response | 156.24 / 119.15 s | 56.97 / 11.27 s | 24 / 5; 44 / 2 | 14,942,480,501 |

The 96-atom single-B owner stores 8,769,110,016 fitted bytes and **zero** raw
resident bytes; every traced SCF J/K call has zero raw-source regeneration.
The default force response took about 43 s and retained a separate 7.74 GB
response allowance; reducing that allowance to 5 GB admitted occupied K
(roughly 0.9 s versus dense packed K roughly 4 s per full iteration) but
slowed the bounded force response to about 59 s. Explicit 26 GB total instead
admitted occupied K while permitting a 37.7 s force response. The warm work
counts do not match the reference, so the ordinary warm ratios are **not**
iteration-matched speed claims. No run establishes a GPU4PySCF speedup.

The explicitly selected source-projected response passed independent energy
and force gates but cost about 110 s of the 156 s cold endpoint and about
119 s warm: 7424 occupied-projection BLAS calls and 3712 raw auxiliary tiles
dominate despite exactly one raw-source pass. It is available for controlled
diagnosis, not selected by `VIBEQC_DF_RESPONSE_SPACE=auto`. Phase repartition
alone is not an endpoint optimization when it shrinks force capacity.

Small-system cold, replay, moved geometry, RHF/UHF fixed-density Fock, dense
fallback, source-projected occupied force and independent energy/force gates
also passed on the allocated 5090. Before broader promotion, qualify
changed-geometry 96-atom endpoints and more representative batch/basis
shapes; retain exact fallback and do not silently select this experiment in
production.

## References

- `src/scf/df_value_storage.hpp`
- `src/scf/density_fitting.cpp`
- `src/scf/cuda/df_plan_setup.cpp`
- `docs/developer/df_occupied_cuda.md`
- `.artifacts/readme-hf-df-aot-20260925/diagnosis/df-96-native.jsonl`
