# Decision: compose CUDA global-hybrid forces from the existing stationary plan

Status: implemented
Date: 2026-09-26

## Problem

Native CUDA SCF already executes canonical global-hybrid graphs, but the force
consumer had fixed seven-source storage, only semilocal snapshots and a
family-to-LDA/PBE/r2SCAN point selector. A public capability flag alone would
omit exact exchange and could substitute PBE for another GGA composition.

## Decision

Derive the native arena layout, source slots, weight functions and final
reduction from StationaryGradientPlan. Full-range exchange contracts the same
ordered ERI derivative provider with D[a,c] D[b,d], separately in each spin.
TensorIR retains the coefficient and RKS/UKS normalization; its bounded inline
lowerer now admits small reducing einsums with an explicit 64-term cap.

Use the actual snapshot MethodIR, including its semilocal graph and exchange
fraction. Public eligibility checks primitive coverage and execution conditions;
it does not duplicate a method-name list. Native SCF composition/domain
admission remains authoritative. Range separation, nonlocal correlation, DF,
ECP hybrids and mixed precision retain their existing rejection boundaries.
In particular, the derivative compiler can lower a different PBE exchange
fraction, but CUDA SCF still rejects unqualified PBE50. Generic derivative
coverage must not be mistaken for permission to broaden that SCF contract.

The CUDA geometry path consumes the same generated point program and work
policy as energy evaluation. Libxc C metadata extraction and split-hybrid
program/emission modules move into the scientific compiler; tools modules
forward to those canonical modules. This makes source generation work in an
installed compiler without importing repository scripts or reference software.
The existing source-derived split-MGGA work policy, spin conditioning and
returned derivatives at work inputs are preserved.

Semilocal public endpoints retain packaged AOT. Composed global-hybrid wrappers
use the existing bounded NVCC/cache path with independently cached integral
objects. They require a discoverable toolkit on first use. There is no CPU
scientific fallback. A future registry-driven AOT inventory can package these
same wrappers without changing their mathematics or source ownership.

## Work and storage

The first implementation traverses ERI derivatives separately for J and K.
Primitive work is therefore 2*K^4 + (A+2)*K^2 + A*(A-1)/2 for global hybrids,
where K counts normalized Cartesian primitive expansions. The work admission
and measured counters both include this second traversal. Eight source panels
replace seven; device storage, host staging and replay budgets include it.
No endpoint speedup is claimed.

## Rejected alternatives

- A PBE0/B3LYP capability whitelist would repeat the gap for other hybrids.
- Treating every GGA as PBE loses semilocal composition and production tails.
- Copying hand-coded exchange factors or split-MGGA point formulas into the
  runtime creates a second owner for scientific semantics.
- Claiming derivative support from CUDA compilation or point agreement alone
  does not establish the complete converged public endpoint.

## Revisit when

- A registry-driven hybrid AOT inventory can preserve the same mathematical
  owner and bounded fallback while removing the first-use toolkit requirement.
- A common J/K derivative traversal is independently qualified with complete
  endpoint timings and updated semantic work admission/counters.
- Native CUDA SCF independently qualifies further semilocal compositions or
  exchange fractions. The generic force consumer should then reuse their point
  programs without adding a parallel method-name admission list.

## Evidence

Qualification used RTX 5090 (sm_120), driver 580.95.05, NVCC 12.9.86,
PySCF 2.14.0 and Libxc 7.0.0. The tested source is base
`6b973f3781407026f612d3bc70d4c785d1460ab4` plus the working-tree changes recorded
by file hash and patch in ignored `.artifacts/hybrid-force/`. This directory
retains the native library hash, toolchain/device identity, per-case numerical
results, loaded artifact hashes, raw logs and reproduction scripts.

- The complete GPU force gate passed all 16 cases under Compute Sanitizer
  memcheck with zero device errors (Slurm job 11828). It covers PBE0, B3LYP,
  M06-2X and MN15 in RKS/UKS, an admitted hybrid under a semilocal method label,
  a p-shell water molecule, and rejected execution/composition boundaries.
- Independent PySCF moving-grid gradients use an energy gate of 2e-8 Eh and a
  force gate of 2e-7 Eh/bohr. Two reconverged directional differences at 3e-4
  and 1e-4 bohr use a 1e-6 gate. Translation invariance, prepared replay,
  changed-geometry reuse and measured primitive work all pass.
- The existing LDA/PBE CUDA diagnostic passed four independent analytic-gradient
  cases, including every signed source. Native CPU PBE0/B3LYP RKS/UKS regression
  passed four cases. The stationary host suite passed 296 cases with 13 explicit
  environment/GPU skips; the real-GPU force gate itself had no skips.
- Eight host-compiled generated exchange weights agree with independent random
  density contractions, covering same-spin normalization and AO permutations.
  Native admission also rejects 64-bit invalid source IDs without narrowing.
- Compiler ownership checks cover 366 modules with zero dependency errors.
  Installed-wheel projection emits the same split-hybrid source without tools,
  runtime or reference imports. Moving the four generator/extractor modules
  preserves M06-2X and MN15 CUDA source bytes against the pre-move tools.

## References

- [Public CUDA hybrid force gate](../../../../tests/python/test_global_hybrid_cuda_forces.py)
- [Generated exchange-weight oracle](../../../../tests/python/test_global_hybrid_cuda_force_lowering.py)
- [User contract](../../../../docs/user/methods.md#cuda-global-hybrid-forces)
- [Execution and resource contract](../../../../docs/developer/stationary_cuda_diagnostic.md)
