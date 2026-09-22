# GFN2 ES3 generated scientific owner

GFN2 ES3 shell science now has one compiler owner. TensorIR defines the shell
energy Gamma3 * q^3 / 3 and production potential Gamma3 * q^2 with the prior
FP64 operation order. Compiler AD also generates dE/dq as an audited witness
that the energy derivative equals the production potential.

CPU and CUDA ES3 runtime files no longer maintain duplicate shell_energy or
shell_potential equations. They retain ragged partition validation, finite
input checks, SCC activity/error propagation, accumulation/reduction, and
publication. The generated wrapper also preserves the previous finite-FP64
fallback contract for extreme caller-supplied values.

The production generated header is registered through the normal build codegen
path and its generator participates in source identity. CUDA ownership records
therefore classify the remaining ES3 translation unit/header as runtime rather
than handwritten scientific code.
