# GFN2 electronic compiler cutover preserves fused accumulation

Status: implemented
Date: 2026-09-21

The retired Mulliken/H0 runtime helpers used std::fma. Splitting their product
from accumulation rejects finite cancellation cases: 1e308*2-1e308 has a finite
fused result even though the isolated product overflows. Summing scalar
potentials before applying half-overlap also changes this accepted domain.

The scalar TensorIR C++ emitter now has explicit opt-in fused accumulation for
single-use multiply addends with coefficient +1 or -1. Shared or published
products retain their normal evaluation and finite checks; unrelated callers
keep the default unfused lowering. Generated GFN2 update graphs retain the
original half-integral and sequential accumulation order. The runtime still
consumes generated kernels rather than reclaiming scientific formulas.

Generated input/result finite checks and atomic output publication remain.
The runtime program version is advanced because operation order/fused rounding
is an execution contract. Tests compile the actual generated header with
automatic compiler contraction disabled and compare against independent
std::fma calculations, including finite cancellation and genuine overflow.

No blanket finite-check disablement or low-precision fallback is used. The
existing VJPs are regenerated from the same scalar Hamiltonian graph.

Agent: ChatGPT (Even-PR Review)
Model: GPT-6 Astra Pro
