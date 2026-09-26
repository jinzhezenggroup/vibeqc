# Decision: scale extreme seeded spectral responses before rounding

Status: implemented
Date: 2026-09-20

The inverse-square-root and pseudoinverse Frechet rules multiply a spectral
coefficient by a transformed response seed. Computing that coefficient alone
can overflow or underflow even when the weighted response is representable.
For example, a retained eigenvalue 2^1000 and seed 2^1023 have inverse-square-root
response -2^-478, but the isolated coefficient -2^-1501 rounds to zero.

Preserve ordinary normal-coefficient evaluation. When the coefficient is zero,
subnormal or nonfinite, evaluate the seeded reciprocal products with normalized
mantissas and an accumulated binary exponent, then apply one final scaling.
The CPU custom rule and generated CUDA adapter use the same arithmetic policy;
no threshold, retained-subspace definition, matrix-function identity or
mathematical divided difference changes. Native eigensystem validation and
runtime ownership remain outside the emitted arithmetic.

Exact powers of two independently fix the expected result, without relying on
CPU/CUDA agreement. New retained-eigenvalue cases at 2^-700 and 2^1000 reproduce
both overflow and underflow failures before repair. The native CPU checks and
both explicit GPU cases pass after repair, along with the previous subnormal
and retained/discarded cross-subspace tests.

This does not establish globally range-safe eigensolvers or matrix products:
the surrounding basis rotations still use their existing FP64 reductions.
It also does not promote public RI-MP2 forces or higher-order custom rules.
A full scaled matrix-algebra redesign would require separate independent
qualification; widening thresholds or replacing the expected values would not
repair the missing representable first-order response.

Review validation used RTX 5090 / CUDA 12.9 with ordinary FP64 flags. Both new
GPU extreme cases failed on the original emitted header and passed on the
repaired header; five combined native/GPU range tests completed without skips.
The existing qualification records retain their original source attribution.

Agent: ChatGPT
Model: GPT-6 Astra Pro
