# Decision: route explicit RCCSD(T) CUDA energy to its native owner

Status: implemented on the #1067 branch; integration qualification remains separate
Date: 2026-09-24

## Problem

The branch implements native CUDA standard-(T) and selects it in RccsdtPrepared, but Calculator retained a CPU-only construction guard from the earlier CPU rollout. Source-matched public H2/H2O CUDA energy tests stopped at that Python guard before executing the new owner. This was an admission mismatch, not a demonstrated triples numerical error.

## Decision and preserved limits

Let the already validated explicit device='cuda' request reach the native RCCSD(T) owner. The existing constructor only accepts cpu/cuda devices, so this does not add automatic backend selection. Keep the native property-boundary rejection of CUDA analytic forces and the uncompiled-CUDA rejection. CPU forces, density-fitting/frozen-core exclusions, correlated-state admission, denominators, numerical tolerances, memory budgets and native equations are unchanged. Default/all-property CUDA requests still fail explicitly; callers request properties=('energy',) for this slice.

## Verification

In Slurm 11579 on RTX 5090, the full tests/python/test_rccsdt_public.py suite passes: 19 tests, no failures or skips. The four explicit CUDA energy cases cover H2, H2O, NH3 and CH4 against the retained independent standard-(T) references, including nontrivial occupied/virtual sizes, physical replay residuals and device/workspace diagnostics. CPU energies/analytic forces, independent H2O/NH3 analytic-gradient checks, multistep force finite differences, nonconvergence, batch replay and reported CPU-force budget tests also pass. Two new tests preserve CUDA-force rejection for default and explicit energy-plus-force requests.

Native library SHA-256: e55f69fc85d9602b4cd8c9d3789ec806a6b427ba36a051e06e26a897f8cf3b38, built from 833af559c8f8a018898c417d5ab39b761900d86d. This repair changes only Python routing, tests and this note; native/generated scientific sources are unchanged. The hash-pinned binary is therefore reused as native support, not described as rebuilt at the later Python commit. Logs/JUnit and the exact job script are retained in /home/jzzeng/reviews/odd-pr-20260924-1238. A missing test-only threadpoolctl dependency was installed into an isolated support directory, not the shared environment. The earlier exploratory run is retained separately; its force-only test request violated the existing mandatory-energy API contract and was corrected to a valid default/all-property request before the final full run.

## Rejected alternatives and remaining work

Do not delete or skip CUDA oracle cases, silently execute CPU triples, relax numerical gates, or advertise CUDA analytic forces. This branch-local energy qualification does not settle conflicts with the evolving canonical RCCSD(T) owner on master, qualify large-system profitability, or close #155. Reconcile integration and rerun its source-matched gates before promoting the merged owner.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro
