# Compiler-first CPU LDA production point

Status: implemented
Date: 2026-09-21
Refs: #762

## Problem

CPU semilocal production had two scientific owners. The compiler emitted canonical
polarized LDA/PBE E/vxc, while `src/dft/xc_point.hpp` independently maintained
tail-stable Jet formulas used by native UKS and PBE RKS. Directly switching the
existing generated functions is not safe: at ordinary and low densities they
match the production oracle, but at total density around 1e-300 the canonical
generated polarized LDA can overflow and canonical generated PBE can become NaN.

## Decision

Move polarized LDA production E/vxc into a compiler-owned scaled graph first.

The graph uses total-density sixth-root and normalized spin coordinates. PW92 is
evaluated through a bounded positive-density polynomial form with a lexical
log1p(x)/x continuation. Exchange energy and its physical density derivatives
use separate stable powers so an underflowing energy does not erase the finite
rho^(1/3) potential. This is an algebraic/numerical-coordinate change only: no
density cutoff, smoothing, or functional modification is introduced.

Native CPU UKS now calls the generated production LDA function. The native
wrapper still owns finite/nonnegative input admission and analytic vacuum
publication; those are runtime/numerical-policy responsibilities, not a second
scientific formula.

PBE is deliberately not switched in this slice. Its qualified production domain
also includes huge-gradient cancellation and spin-endpoint behavior. The next
migration must express that stable feature formation/pullback through the
compiler before retiring the handwritten PBE owner.

## Evidence

Three independent 450-digit tail references at total density 1e-300, including
both empty-spin endpoints, pass the generated function with the existing
relative gate. The previous canonical generated function was nonfinite there.

Fresh CPU validation on node3:
- vibeqc_dft_tests: passed;
- vibeqc_uks_tests: passed;
- XC expression + compiler-structure Python suite: 131 passed;
- compiler structure: 259 modules, 0 dependency errors;
- Ruff and git diff checks: passed.

No numerical tolerance, public capability, BLAS/provider policy, CUDA path, or
SCF convergence rule was changed.

Agent: ChatGPT
Model: GPT-5.6 Sol
