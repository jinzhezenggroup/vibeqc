# Decision: preserve representable spectral coefficients without overflowing products

Status: implemented
Date: 2026-09-19

## Problem

The native pseudoinverse and inverse-square-root response formed denominator
products before division. Large finite eigenvalues could overflow those products
and silently produce a zero derivative even when the coefficient and final
response were representable. Symmetrizing a large finite seed as (a+b)/2 could
also overflow before multiplication by the small spectral coefficient.

## Decision

Match the shared Python rule's ordered divisions: -1/li/lj and
-1/sqrt(li)/sqrt(lj)/(sqrt(li)+sqrt(lj)). Symmetrize by halving each operand
before addition. Apply the same pseudoinverse arithmetic to generated CUDA.
Preserve retained/discarded cross-subspace terms, branch admission, contraction
ordering and all ordinary scientific tolerances. This is not arbitrary-precision
arithmetic or a guarantee that every possible full matrix product is range-safe.

## Evidence

Power-of-two spectra 2^520 and 2^700 with seeds 2^500 and 2^1023 yield exactly
representable subnormal coefficients and normal responses. The new native test
fails on the original implementation and passes after repair with exact equality.
The generated CUDA pseudoinverse cases also pass on RTX 5090 with NVCC 12.9.1
and --fmad=false; Compute Sanitizer memcheck reports zero errors. These are
rule-level checks, not a new whole-molecule CUDA timing campaign.

Agent: ChatGPT
Model: GPT-6 Astra Pro
