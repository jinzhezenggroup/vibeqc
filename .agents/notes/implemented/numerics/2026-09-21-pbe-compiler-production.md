# Compiler-owned tail-stable PBE CPU production point

Status: implemented
Date: 2026-09-21
Refs: #768, #762
Depends on: #765

## Problem

The canonical generated polarized PBE expression is numerically correct in its
ordinary domain but can form non-finite inverse-density/reduced-gradient
intermediates in the native SCF domain. The qualified
`semilocal-scaled-v1/pbe-spin-c2-1e-18` contract explicitly includes total
densities down to 1e-300, empty spin channels, huge finite gradients and
cancellation between opposite spin gradients. Returning zero or clipping a
density is invalid because the energy may underflow while finite first
derivatives remain representable.

## Compiler ownership

The production lowering now separates the stable PBE point model into three
compiler-owned scalar graphs:

1. scaled PBE correlation in normalized spin-density and bounded Cartesian
   total-gradient coordinates;
2. direct reduced-gradient PBE exchange;
3. reciprocal reduced-gradient PBE exchange.

The generated host wrapper owns only numerical scaling and branch selection:
- fixed total-density and gradient scales;
- finite-summand normalization when a total Cartesian gradient overflows;
- exact-zero-gradient selection;
- direct versus reciprocal exchange selection.

The scientific energy and first-derivative formulas in every branch come from
the shared scalar Graph and are hashed independently.

The correlation graph includes the existing C2 spin endpoint extension,
stable PW92 sixth-root/log1p(u)/u algebra, direct v=d/(d+A|h|^2) evaluation,
and the cancellation-safe large-gradient PW+H expression. The exchange graphs
retain independent rho^(1/3) potential evaluation when rho^(4/3) energy terms
underflow.

## Production integration

Native CPU PBE RKS and UKS now consume the generated production point function.
The C++ adapter retains only input admission, finite-result checks and AO/grid
traversal. The handwritten point evaluator remains for response/CUDA/snapshot
owners and as an independent migration oracle; this slice does not alter those
capabilities.

Independent exchange/correlation scaling is part of the generated ABI and is
checked against the retained point oracle, covering PBE0-style semilocal
composition.

## Evidence

- all 61 independent PBE rows in tests/data/xc/scf_domain.tsv pass the existing
  component-wise relative gate;
- combined generated LDA/PBE native gate covers all 97 SCF-domain points;
- rho_total=1e-300, empty/near-empty spins, C2 joins, huge finite gradients,
  opposite-gradient cancellation and underflowing gradient-square fixtures pass;
- scaled X/C point comparisons pass against the retained independent point
  implementation;
- vibeqc_dft_tests and vibeqc_uks_tests pass;
- focused XC/compiler Python suite: 131 passed;
- compiler structure: 259 modules, 0 dependency errors;
- ty 0.0.82 exits 0 with repository-baseline warnings only;
- Ruff, clang-format and git diff checks pass.

No tolerance, SCF convergence policy, CUDA capability, BLAS/provider behavior or
public method identity was changed.

Agent: ChatGPT
Model: GPT-5.6 Sol
