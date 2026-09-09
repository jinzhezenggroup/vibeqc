# Collective provider-budget correction

This follow-up preserves C commit `5f31c4289db59853e4f64a942a51b4cccc68a3cd`.
A subsequent independent engineering reviewer (`c_engineering_review`) found
one P2: interpreter-budget preflight alone did not prevent partial AO work
when all seven individually feasible blocks exceeded the provider's cumulative
pin budget. The original scientific equations and successful endpoint values
were unaffected; failure cost and the promised early-rejection behavior were.

The CC adapter now dry-runs the complete pin/LRU cache transition under the
existing provider RLock, then performs real gets under the same lock. It
accounts for retained blocks, hits, recency and output costs, preserving warm
cache feasibility. No shared provider, HF, registry or #193 files changed.
This internal adapter depends on #147's private cache accounting; future
provider cache changes must retain or update the accompanying regression.

The new test uses the actual ConventionalProvider planner/get path and a
source that fails on any AO read. Each block fits individually, but the
complete cold pinned set fails with zero reads and zero cache mutation. A
fully warm cache succeeds under the same budget. The uninvolved reviewer
rechecked the patch and found the P2 fixed, without new blockers.

Validation after the correction:

- Local solver tests: 11 passed, 5 native-library skips.
- qz `python -m pytest tests/python/test_cc_solver.py -q`: **16 passed in
  56.17 s**, including five fresh native HF→CCSD endpoints and same-C roots.
- Ruff check/format passed; 2 files already formatted. `git diff --check`
  passed. Full 231-test and ten external-endpoint evidence remains the
  separately recorded C baseline; this is the focused correction regression.
- The three-file upload was read back with SHA-256
  `35011e1ae4a9c277c41034d88aa7a6ab8499db010e67be9ffd3c18be2b1dccec`.
  Local/remote hashes matched:
  - solver.py: `9e43fb778c2576098a1c862dfc335ad74428b841abef49a96892c7cd7af47583`
  - test_cc_solver.py: `f11d3597fedfcf022ef3b1e93e7efa66d82352787e23978bc6ca2b9660d1c45c`
  - docs/rccsd_bc.md: `018a8ae50ca86d9009fd8fa06a2ec85ed74df7c3171121044bca2b51c06d0960`

Existing qz Python 3.11/CPU library and single BLAS/OpenMP thread were reused.
No GPU or new instance was used, and no previous computation was restarted.
