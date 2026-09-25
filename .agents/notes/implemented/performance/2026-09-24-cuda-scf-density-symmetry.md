# Decision: contract only unique AO pairs in CUDA SCF density builds

Status: implemented
Date: 2026-09-24

## Problem

The production CUDA RHF/RKS and UHF/UKS density builders launch over the full dense `nbf x nbf` matrix and perform an occupied-orbital dot product independently for `(mu, nu)` and `(nu, mu)`. The ordinary density is mathematically symmetric, so the lower triangle repeats the same coefficient reads and FP64 contraction every SCF density rebuild. These kernels are called by both CUDA HF and CUDA KS iteration paths.

The weighted-density kernels are deliberately excluded: their left-associated `orbital_energy * C_mu * C_nu` expression can change FP64 rounding when the AO operands are exchanged.

## Decision

Keep the existing full-square launch and dense output layout, but let only `row <= column` threads execute the occupied-orbital contraction. Each off-diagonal owner writes the completed value to both dense matrix locations. This avoids a triangular-index decoding path, preserves the current launch API and resource accounting, and removes the duplicated expensive inner loop.

## Invariants

- Occupied-orbital traversal and FP64 accumulation order for each retained AO pair are unchanged.
- RHF keeps the existing exact power-of-two occupation factor `2.0`; UHF/UKS keeps the existing unit occupation expression.
- The full dense matrix remains published before downstream J/K, XC, convergence and force consumers run.
- Inactive systems retain the previous no-write behavior.
- Weighted-density kernels remain full-square to preserve their historical orbital-energy multiplication order.

## Work census

Unique AO-pair contractions fall from `n^2` to `n(n+1)/2` per density matrix:

| nbf | before | after | reduction |
| ---: | ---: | ---: | ---: |
| 384 | 147,456 | 73,920 | 49.87% |
| 768 | 589,824 | 295,296 | 49.94% |

Each retained pair still traverses every occupied orbital, so coefficient-read and contraction work falls by the same pair ratio. Dense output stores remain `n^2` because mirrored values are still published.

## Local GPU microbenchmark

A standalone source-matched CUDA 12.4 harness on the node3 RTX 5090 warmed each kernel and timed repeated RHF density launches with `nocc=nbf/2`. Five paired samples used 50 launches/sample at 384 AO and 12 launches/sample at 768 AO. Median per-launch times:

| shape | baseline | optimized | change |
| --- | ---: | ---: | ---: |
| 384 AO / 192 occ | 0.089456 ms | 0.076017 ms | -15.0% (1.18x) |
| 768 AO / 384 occ | 0.594600 ms | 0.343061 ms | -42.3% (1.73x) |

This is a focused kernel benchmark, not an end-to-end HF/DFT speed claim. Promotion evidence should compare source-matched Release CUDA HF/KS endpoints with identical molecule, basis, convergence, provider and iteration counts, and profile `build_density_kernel` / `build_spin_density_kernel` separately from J/K and XC work.

## Correctness evidence

- A standalone CUDA RHF/UHF harness compared the optimized kernels against independent full-square host formulas on a real RTX 5090 and passed exact element comparisons.
- `tests/python/test_scf_cuda_density_symmetry.py` locks the triangular ownership, mirrored publication, 384/768 work census, and the intentional exclusion of weighted density.
- Existing CUDA HF/DFT/UHF endpoint tests remain the numerical integration gate in CI.

## References

- Issue #1117 — GPU HF/DFT performance tracking and matched-work acceptance policy.
- PR #1210 — analogous generated CPU density optimization; this change targets separate handwritten CUDA kernels.
- `src/scf/cuda/scf_density_kernels.cu` — CUDA density builders.
- `tests/python/test_scf_cuda_density_symmetry.py` — schedule/work-count regression.

Agent: ChatGPT
Model: GPT-5.6 Sol
