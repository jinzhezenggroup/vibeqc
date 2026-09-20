# Decision: fuse weighted CPU DF derivatives at the generated-integral boundary

Status: implemented
Date: 2026-09-20

## Problem

The generated host lowering removed runtime Jet AD from production s/p/d/f density-fitting
derivatives, but CPU force preparation still materialized complete coordinate-resolved
Coulomb-metric and three-center derivative tensors:

- `dM/dR`: `O(ncoord * naux^2)`;
- `d(mu nu|P)/dR`: `O(ncoord * nbf^2 * naux)`.

That ownership was unnecessary because the SCF force consumer ultimately contracts those
derivatives with fixed final-state energy weights. On the WATER27 tetramer/def2-SVP CPU
endpoint the materialized derivative owner was large enough to remain a significant
runtime and resident-memory cost after the #681 host-codegen/OpenBLAS work.

## Decision

Make the normal CPU density-fitting force path a weighted derivative consumer.

The prepared CPU Fock owner retains ordinary DF value tensors and the orbital/auxiliary
geometry, but does not retain coordinate-resolved DF derivative tensors. At finalization:

1. reverse the fixed-density RHF/UHF DF energy into adjoints of the raw public-basis
   metric `M` and three-center tensor `A(mu,nu,P)`;
2. pull those public-basis weights back through the AO representation transform;
3. evaluate the existing generated s/p/d/f metric and three-center primitive derivatives;
4. multiply each generated derivative by its external weight immediately and scatter
   directly into nuclear coordinates.

The same weighted response is used by the production `PreparedFockPlan` /
`CpuFockProviderView` path and the legacy CPU DF force endpoint. CUDA response ownership
is unchanged.

## Reverse-weight convention

Let `G = M^+` be the thresholded Coulomb-metric Moore--Penrose inverse and
`q_P = sum_mn D_mn A_mnP`. For a Coulomb coefficient `c_J`, the existing energy is

`E_J = 1/2 c_J q^T G q`.

The reverse weights are therefore

- `bar(A_mnP) += c_J D_mn (G q)_P`;
- `bar(G_PQ) += 1/2 c_J q_P q_Q`.

For one exchange density `D`, define `A_P` as the AO matrix at auxiliary index `P`
and `R_P = D^T A_P D`. With the existing HF energy convention the exchange contribution
is

`E_K = 1/2 c_K sum_PQ G_PQ <R_P, A_Q>`.

The implementation uses `scale = 1/2 c_K` and accumulates both appearances of `A`:

- the direct adjoint from the explicit `A_Q`;
- the indirect adjoint through `R_P = D^T A_P D`.

These contractions use the existing CPU dense-linear-algebra provider boundary rather
than vendor symbols in SCF code.

The metric adjoint is **not** formed with the full-rank shortcut
`-G (dM) G`. The accumulated `bar(G)` is mapped to `bar(M)` by the existing
`density_fitting_metric_inverse_response` / symmetric matrix-function VJP. This
preserves the selected truncated-pseudoinverse subspace and the established rank-crossing
semantics at `density_fitting_relative_threshold`.

For UHF, Coulomb uses the total density while alpha and beta exchange adjoints are
accumulated independently with the existing unrestricted exchange coefficient.

## Public-to-Cartesian pullback

Generated host derivative evaluators operate in the Cartesian AO representation.
Public orbital and auxiliary bases may instead be real spherical. The transform is
linear, so the weighted path applies the exact adjoint of the existing public transform
before generated derivative evaluation:

- two-index metric weights pull back through both auxiliary AO transforms;
- three-center weights pull back through both orbital AO transforms and the auxiliary
  AO transform.

The spherical weighted-contraction test compares this pullback directly against a dot
product with the independently materialized public derivative tensors.

## Fallback and diagnostic oracle

The fused generated route is the production path when all orbital and auxiliary shells
have angular momentum `l <= 3`, matching the qualified generated CPU DF domain.

Higher-angular-momentum inputs keep the previous full-tensor derivative implementation
as an explicit correctness fallback at finalization. This fallback is not evidence that
the higher-l generated route is promoted.

For causal A/B validation,
`VIBEQC_CPU_DF_MATERIALIZED_DERIVATIVES=1` restores materialized CPU DF derivative
ownership. This is a diagnostic control, not a separate scientific method.

Provider validation accepts exactly one of two complete derivative sources:

- materialized `dM/dR` and `dA/dR` tensors; or
- no derivative tensors plus bound orbital and auxiliary geometry for weighted response.

Mixed or incomplete ownership is rejected.

## Evidence

OpenBLAS CPU native regression selection on node3:

- `vibeqc_fock_build_tests`;
- `vibeqc_fock_provider_tests`;
- `vibeqc_fock_api_tests`;
- `vibeqc_native_tests` (RHF);
- `vibeqc_density_fitting_tests`;
- `vibeqc_uhf_tests`.

All six passed after the production prepared-provider migration. The density-fitting
tests compare Cartesian RHF/UHF weighted response with the previous materialized oracle
and compare spherical public-basis pullback with full derivative tensors. The Fock
provider test additionally asserts that the normal prepared CPU DF force owner contains
no coordinate-resolved DF derivative tensors.

Paired endpoint measurements used one OpenBLAS/OMP thread:

| endpoint | fused | materialized | speedup |
|---|---:|---:|---:|
| H2O/def2-SVP RHF | 0.06109 s | 0.06648 s | 1.09x |
| H2O/def2-TZVP RHF | 0.41088 s | 0.46255 s | 1.13x |
| OH/def2-SVP UHF | 0.03438 s | 0.03721 s | 1.08x |

A finite Slurm allocation on node3 reproduced the larger endpoint result for the
WATER27 tetramer/def2-SVP RHF case (96 real spherical AOs), two interleaved paired
repeats:

- fused median: 5.9298 s;
- materialized median: 8.4803 s;
- speedup: 1.43x;
- SCF iterations: 13 in both paths;
- energy difference: 1.01e-11 Eh;
- maximum force difference: 4.08e-11.

Fresh-process `/usr/bin/time -v` measurements for the same tetramer endpoint reported:

- fused maximum RSS: 336,932 KiB (~329 MiB);
- materialized maximum RSS: 802,528 KiB (~784 MiB);
- reduction: 465,596 KiB (~455 MiB), about 58%.

The remaining numerical differences are ordinary FP64 operation-order differences and
do not change the converged iteration count in the measured cases.

## Consequences

The normal CPU force lifetime no longer scales with
`ncoord * nbf^2 * naux` for generated s/p/d/f DF derivatives. The weighted reverse pass
still owns `O(nbf^2 * naux)` value/adjoint workspaces, so this decision does not claim
that the complete DF value tensor or every packing temporary has been eliminated.

The materialized implementation remains intentionally reachable as an independent A/B
oracle and as the higher-l fallback. Removing that oracle requires separate validation.

## Revisit when

- generated CPU DF support is promoted beyond f shells;
- reverse-weight workspaces can reuse persistent Q-major value storage without worsening
  complete endpoints;
- #471 replaces fixed provider thresholds with workload-aware tuning;
- MKL/BLIS or additional dense-LA operations are qualified;
- final-head CI exposes a platform-specific behavior not covered by the node3 validation.

## References

- #674
- #730
- #681
- #471

---
Agent: ChatGPT
Model: GPT-5.6 Sol


## Independent review qualification (2026-09-21)

The scalar-provider review independently built the exact source and ran all 46
native CTests. The prepared-provider follow-up was read and included in the
rebuild. No OpenBLAS or GPU result is inferred from this scalar build.

Additional review regressions repair two gaps in the initial test coverage:

- A one-center spherical fixture has a zero nuclear derivative. The new two-center
  spherical fixture uses signed, nonsymmetric metric/three-center weights, requires
  a nonzero reference derivative, and checks every coordinate against materialized
  public derivative tensors.
- The new weighted reverse-chain test executes 48 independently rebuilt energy
  finite differences: RHF/UHF, full and genuinely truncated constant metric ranks,
  four Coulomb/exchange coefficient choices, and three displacement sizes. It also
  checks the force sign, translational sum and materialized derivative parity.

`tests/python/test_df_cpu_weighted_endpoint.py` adds four complete public CPU
endpoints: H2/sto-3g RHF, H2O/sto-3g RHF, OH/sto-3g UHF and H2O/def2-SVP RHF.
Both ownership modes agree, including SCF iteration counts. An independent PySCF
2.14.0 calculation consumes the exact same primitive exponents and coefficients,
not just an equal basis name; independently maintained catalogs can differ in
rounding. Auxiliary-basis response is enabled in the oracle. Each endpoint also
checks two independently reconverged energy finite differences and zero net force.
All four regression cases passed locally without skips.

Across these four endpoint comparisons, the largest fused/materialized force
change was 2.64e-12 Eh/bohr and the largest matched-basis independent force error
was 1.32e-10 Eh/bohr. This is numerical qualification, not a universal performance
claim: the very small systems do not establish a speed advantage. The larger
OpenBLAS measurements above belong to the implementation campaign, not this
independent scalar review. Final published-head CI remains required.

Agent: ChatGPT
Model: GPT-6 Astra Pro
