# COSX discrete CPU reference (issue #246 slice A)

This document fixes the mathematics of the first VibeQC COSX reference before
GPU screening, fitting, mixed precision, SCF integration, or provider selection
is allowed to change execution.

## Version-1 model

The caller supplies an explicit quadrature: points R_g in Bohr and volume
weights w_g. The point coordinates and weights are part of the result identity.
The test suite currently uses MolecularGrid only as a reproducible fixture;
COSX does not inherit the XC grid prescription from that use.

For every point, the analytic ESP matrix is

\[
V_{\mu\nu}(g)=\int \chi_\mu(r)\frac{1}{|r-R_g|}\chi_\nu(r)\,dr.
\]

The CPU ESP oracle reuses the normalized contracted Gaussian Hermite/Boys
recurrence already used by the independent nuclear-attraction reference, but
with a unit positive probe and no nuclear charge. Probe coordinates are fixed
external coordinates in this value-only slice.

The one-sided seminumerical two-electron approximation is

\[
(\mu\lambda|\nu\sigma)_{\rm SN}
 = \sum_g w_g\,\chi_\mu(R_g)\chi_\lambda(R_g)V_{\nu\sigma}(g).
\]

For a supplied AO density D,

\[
K^{\rm raw}_{\mu\nu}
 = \sum_{\lambda\sigma}D_{\lambda\sigma}
   (\mu\lambda|\nu\sigma)_{\rm SN}.
\]

Version 1 always returns

\[
K=\frac{1}{2}(K^{\rm raw}+(K^{\rm raw})^T).
\]

There is no overlap/S fitting and no screening in version 1. A request that
changes either of those semantics is rejected instead of being silently
treated as the same approximation.

## Spin convention

For a single alpha or beta density block, the exchange-only energy is

\[
E_x=-\frac{1}{2}\operatorname{Tr}(DK[D]).
\]

For an RHF spin-summed density, it is

\[
E_x=-\frac{1}{4}\operatorname{Tr}(DK[D]).
\]

Thus two identical UHF spin densities D/2 reproduce the RHF exchange energy
for D. The returned K matrix itself is always the positive exchange build; the
RHF Fock uses -K/2 and a UHF spin Fock uses -K for the corresponding spin
density.

## Validation boundary

The native reference tests deliberately separate three errors:

1. Algebra: a compact AO/ESP contraction is compared with an explicit
   four-index discrete contraction on the identical point set.
2. ESP arithmetic: analytic ESP matrices are compared with an independent
   real-space numerical quadrature and must improve under grid refinement.
3. COSX quadrature/model error: the symmetrized seminumerical K is compared
   with the analytic direct-ERI K; refining the integration grid must approach
   direct exchange.

The first comparison is an arithmetic check at fixed discrete semantics. The
second and third are quadrature-convergence checks and therefore do not use
the same tolerance as the first.

## Deliberate non-capabilities

This reference materializes AO values and all point-by-AO-by-AO ESP matrices.
It is a small-system oracle, not a production execution path. It does not:

- register a COSX Fock provider or change AUTO selection;
- perform chain-of-spheres/local screening;
- perform overlap fitting or any other COSX fitting variant;
- provide GPU execution, batching, resource planning, or warm replay;
- advertise analytic nuclear derivatives or DFT hybrid forces;
- claim a crossover against RI-K.

The next #246 slice must preserve these versioned semantics while replacing
the materialized reference tensors with bounded AO/ESP tasks and generated
native device execution.

The durable rationale for sharing the Coulomb recurrence and keeping fitting,
screening, and grid prescription outside v1 is recorded in
[the COSX discrete-reference Agent Note](../.agents/notes/implemented/numerics/2026-09-19-cosx-discrete-reference.md).
