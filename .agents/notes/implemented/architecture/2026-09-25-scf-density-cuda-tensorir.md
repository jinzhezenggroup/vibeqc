# Decision: generate CUDA SCF plain density from TensorIR

Status: implemented
Date: 2026-09-25

## Problem

CPU plain-density construction already consumes a checked SCF TensorIR specialization,
while CUDA RHF/UHF retained separate handwritten occupied-orbital contractions in
`src/scf/cuda/scf_density_kernels.cu`. The CUDA kernels had also acquired the
same upper-triangle symmetry optimization independently, leaving one scientific
equation and schedule maintained twice.

## Decision

Add a target-owned CUDA lowerer in
`python/vibeqc_compiler/tensor/scf_cuda.py`. It validates the canonical
`D[b,s,p,q] = sum_i occ[b,s,i] C[b,s,p,i] C[b,s,q,i]` TensorIR topology and
emits the bounded integer-occupation specialization used by production RHF/UHF.

The generated kernel preserves the qualified dense launch, upper-triangle
ownership, mirrored publication, active-item routing and historical FP64 product
order. RHF specializes occupations to exact weight 2; UHF/UKS specializes them
to exact weight 1. Native CUDA retains only the launch adapter for plain density.

Warm-start repair and energy-weighted density remain native. Their rounding,
metric-trace and state semantics are separate and require independent migration.

## Invariants

- CPU and CUDA plain density carry the same shape-independent TensorIR identity.
- Backend lowering does not change the scientific density equation.
- The existing `n(n+1)/2` contraction census is preserved.
- Weighted density is not mirrored or silently migrated.
- No solver, DIIS, J/K, convergence, allocation or stream ownership moves into
  the compiler in this slice.
- GPU endpoint qualification remains required before claiming performance
  equivalence; source ownership is not a latency claim.

## References

- #762 compiler-first production science
- #926 CPU/GPU scientific-path unification
- #1210 generated CPU density symmetry
- #1231 CUDA density symmetry history

Agent: ChatGPT
Model: GPT-5.6 Sol
