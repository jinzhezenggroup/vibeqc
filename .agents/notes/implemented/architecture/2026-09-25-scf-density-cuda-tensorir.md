# Decision: generate CUDA SCF density contractions from TensorIR

Status: implemented
Date: 2026-09-25

## Problem

CPU plain and energy-weighted density construction already consume checked SCF TensorIR specializations,
while CUDA RHF/UHF retained separate handwritten occupied-orbital contractions in
`src/scf/cuda/scf_density_kernels.cu`. The CUDA kernels had also acquired the
same upper-triangle symmetry optimization independently, leaving one scientific
equation and schedule maintained twice.

## Decision

Add a target-owned CUDA lowerer in
`python/vibeqc_compiler/tensor/scf_cuda.py`. It validates the canonical
`D[b,s,p,q] = sum_i occ[b,s,i] C[b,s,p,i] C[b,s,q,i]` TensorIR topology and
emits bounded integer-occupation specializations used by production RHF/UHF plain and energy-weighted density.

The generated plain-density kernel preserves the qualified dense launch,
upper-triangle ownership, mirrored publication, active-item routing and historical
FP64 product order. The generated weighted-density kernel deliberately remains
full-square and preserves the historical orbital-energy/coefficient multiplication
order. RHF specializes occupations to exact weight 2; UHF/UKS specializes them
to exact weight 1. Native CUDA retains the launch adapters.

Warm-start repair remains native because its symmetry repair, metric trace,
normalization and failure-state semantics form a separate stateful contract.

## Invariants

- CPU and CUDA plain density carry the same shape-independent TensorIR identity.
- Backend lowering does not change the scientific density equation.
- The existing `n(n+1)/2` contraction census is preserved.
- Weighted density is compiler-owned but remains full-square with its historical FP64 multiplication order.
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
