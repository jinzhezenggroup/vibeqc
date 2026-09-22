# Issue #162 CUDA KS transport evidence (2026-09-14)

This record summarizes the source-bound transport evidence produced from
commit 31372ac6576acd0d499980245633b19803dc2390 (fork branch
codex/issue-0162-d). The raw JSON files remain in the H100 validation
worktree build/ directory and are identified by the hashes below.

## Execution

The existing H100 Notebook vibeqc-mp2-b-20260909 (NVIDIA H100 80 GB,
CUDA 12.9.86, sm_90) built build/cuda/libvibeqc.so and ran the existing
tools/validate_dft_endpoints.py runner for --batches 1 and --batches 2.

Both commands reported: DFT endpoint validation passed: 4 cases.
Each case covers LDA/PBE RKS/UKS and the four phases cold, fixed geometry,
changed geometry, and changed-geometry replay. The runner checked independent
CPU energies, actual CUDA backend, convergence and physical residual < 1e-9,
warm-state behavior, and per-phase cumulative transport plus deltas.

## Raw evidence hashes

| workload | bytes | SHA-256 |
| --- | ---: | --- |
| batch-1 | 12542 | 67fe0c7a0d3c25f1f94d5a0d08fa7dd07ea3b1f6a4797bba62d1063d868a5187 |
| ragged batch-2 | 15390 | 14029a40df74e748ac8f4e7e7e0ca122cf91fe6403989cb3a6b98629c6e26a9b |

The public transport diagnostic reports setup H2D, explicit density H2D,
scalar D2H, explicit matrix D2H, synchronization, iteration, and occupation
stabilization proposal counters. Retired-owner counters are included when a
changed-geometry rebuild exports a last-good density and constructs its
replacement owner. CPU and unavailable/legacy libraries fail closed.
