# Decision: preserve provider-independent CPU linear algebra boundaries

Status: implemented
Date: 2026-09-20

## Problem

The scalar GEMM read C for beta=0 and A/B for alpha=0, propagating unused
NaN values unlike the BLAS computation. Zero-inner-dimension output scaling
had the same defect. GEMM/Cholesky matrix extent multiplication could overflow.
The scalar Jacobi eigensolver also overflowed an angle difference for a finite
2x2 input whose eigenvalues and eigenvectors remain representable.

## Decision

Skip mathematically unused inputs for zero factors, reject unrepresentable
matrix byte extents before traversal, and normalize only extreme scalar
Jacobi inputs before iteration. Rescale eigenvalues at publication and reject
nonfinite results. Normal-scale Jacobi arithmetic remains unchanged.

## Evidence and boundaries

Seven independent C++ boundary checks include zero factors, overflow admission,
and analytical eigenpairs/residuals of [[s,0.1s],[0.1s,-s]] for large and small s.
Six fail before repair; all pass afterward. This establishes neither general
arbitrary precision nor a new default provider or endpoint speedup. The probe
also accepts auto/scalar multithread requests without constructing an invalid
ownership plan and reports exceptions rather than aborting.

Agent: ChatGPT
Model: GPT-6 Astra Pro
