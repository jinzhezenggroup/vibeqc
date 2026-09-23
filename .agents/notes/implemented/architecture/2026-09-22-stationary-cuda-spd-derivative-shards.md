# Decision: lower multicomponent s/p/d stationary derivatives through bounded CUDA shards

Status: implemented
Date: 2026-09-22

## Problem

The public stationary CUDA force owner previously admitted only public AOs that
mapped one-to-one to an s or p Cartesian component. Canonical r2SCAN-3c binds the
spherical def2-mTZVPP basis, whose second-row atoms contain real-spherical d AOs
expanded over multiple normalized Cartesian components. The CPU component
executor already preserved those expansions, but the CUDA task format could not
name the canonical center/axis permutation or link the full derivative inventory
within one practical translation unit.

## Decision

Keep the existing first-derivative graphs as the sole mathematical lowering.
Expand each public AO task over its normalized Cartesian terms on the host,
multiply the public transformation coefficients into the existing task weight,
and encode the canonical derivative request plus center/axis permutation in the
task kind. A small device adapter restores the caller's center and Cartesian-axis
ordering around the canonical primitive call.

Generate the finite s/p/d derivative inventory as independently cached CUDA
translation units of at most 16 requests. The method compiler accepts the shard
width explicitly, compiles each unit as relocatable device code, and links the
ordered objects with the stationary wrapper. The s/p one-component route keeps
using its packaged AOT artifact. Multicomponent all-electron s/p/d uses JIT;
ECP d-shell gradients remain rejected.

The public primitive-work default is 16,000,000 records. This admits the first
canonical r2SCAN-3c water gate while retaining the existing explicit byte, AO,
atom, grid, pair-visit and per-tile limits.

## Rejected alternatives

- A separate d-shell recurrence implementation would duplicate the accepted
  derivative algebra and risk different normalization or sign conventions.
- Materializing a Cartesian density or derivative tensor would turn a bounded
  task stream into a large retained intermediate and obscure semantic work.
- One monolithic full-domain CUDA source would exceed the practical compiler
  process boundary and prevent per-shard cache reuse.
- Reusing the s/p AOT artifact for d shells would silently omit required
  components; the public force path instead selects explicit JIT.

## Invariants

- Public AO indices, normalization coefficients, ordered density weights,
  derivative signs and force = -gradient remain unchanged.
- The binding must be a complete center permutation and one of the six Cartesian
  axis permutations; malformed encoded tasks fail closed.
- Generated-source, header, target, toolchain and binary hashes remain part of
  each cached object and final linked-artifact identity.
- ECP component expansion is not promoted by this decision.
- Prepared execution may rebind coordinates only for the exact retained basis
  topology and artifact identity.

## Evidence

- The full ten-label s/p/d domain contains 362 canonical requests split into 23
  CUDA units, 22 with 16 requests and one with 10. Generated source totals
  22,993,773 bytes; the largest unit is 2,689,639 bytes, below the existing
  4 MiB per-unit and 64 MiB whole-program caps.
- The exact spherical def2-mTZVPP work bound is 39,201 records for H2,
  7,322,433 for HF and 12,134,769 for water. HF and water exercise the full
  ten-label domain; H2 alone does not cover d-shell lowering.
- Device-free layout, dispatch, shard-name, source-budget, AOT-selection and
  compiler-ownership gates live in `tests/python/test_stationary_cuda_d_shell.py`
  and the existing stationary merge-boundary suite.
- The opt-in r2SCAN-3c CUDA suite defines water total-force directional finite
  differences, bounded semantic-work reporting, and HF/water ragged
  changed-geometry replay against fresh execution. These numerical gates still
  require a source-matched real-device run before closure is claimed.
- An early RTX 4090 source snapshot based on `dcd9648a` completed its water
  resource gate in 313.18 seconds: 12,134,769 primitive records, 2,568,000
  task descriptors, and a 110,044,688-byte additional-device bound below the
  536,870,912-byte budget. This is resource and execution evidence for that
  snapshot, not numerical acceptance for a later master revision.
- The LF source archive at tree `0e6350ac` was built on an H100 (`sm_90`,
  CUDA 12.9) into native library SHA-256 `9d8931fc...2bfefca`.
  Its first real-device run passed seven of eight opt-in cases, including water
  total-force finite differences (2.01e-9 Eh/Bohr error), exact HF/water
  changed-geometry replay, charged H3+ ragged failure isolation, and a
  production-grid independent water RKS total (8.53e-14 Eh energy,
  1.25e-11 Eh/Bohr force error). The eighth case reached the PySCF OH UKS
  reference, where default, atomic-density and second-order SCF starts failed
  strict convergence; none was accepted as an oracle result.
- For OH, using the native final density only as PySCF's initial guess let
  PySCF independently reconverge and evaluate its own analytic gradient on
  the identical grid (7.11e-14 Eh energy, 5.68e-14 Eh/Bohr force error).
  This qualifies agreement at that stationary basin, not independence of
  basin selection. A separate charged, bent H2O+ UKS d-shell case converged
  from PySCF's default guess and passed the same total gates (8.53e-14 Eh,
  1.06e-11 Eh/Bohr). Both retain the original 2e-9 Eh and 1e-7 Eh/Bohr
  acceptance limits.
- The subsequent H100 run `issue-0172-final-h100-0e6350ac-20260923` reused
  that hash-verified production library, with separately hashed final test
  overrides, and terminated successfully: 60 device-free tests and ten
  real-device cases passed. It additionally qualified a noncovalent H2 dimer
  from PySCF's own default guess (1.07e-14 Eh energy and 1.57e-11 Eh/Bohr
  force error). Water consumed 12,134,769 primitive records in 2,568,000
  descriptors, with a 110,044,688-byte additional-device bound below
  536,870,912 bytes. These are bounded small-system gates, not universal
  H-Ar support or a complete-endpoint performance claim.
- A matched complete E+F matrix then exposed an OH UKS changed-versus-fresh
  mismatch: 2.61e-10 Eh energy and 4.57e-7 Eh/Bohr force, despite stable
  cold/warm replay. At the same displaced geometry both native solves reported
  physical residuals near 4e-12, but their alpha/beta density matrices differed
  by up to 0.0225/0.2148; fresh-to-fresh-warm density differences were about
  1e-11. This confirms distinct converged spin-density branches rather than
  an identical retained state with a changed force. It does not establish
  which branch is the physical target or qualify general OH changed/fresh
  equivalence. Preserve the failed matrix receipt; do not relax the global
  force gate to hide this case.
- A separate five-system complete E+F matrix, explicitly excluding that OH
  branch-sensitive case, succeeded on the same H100 library for H2, H3+,
  noncovalent H2 dimer, water RKS and H2O+ UKS. Water cold/three-warm median/
  changed times were 120.322/43.110/49.445 seconds; H2O+ UKS times were
  55.193/44.241/51.881 seconds. All five retained their raw samples and
  passed 2e-9 Eh / 1e-7 Eh/Bohr changed-fresh gates. These are complete
  endpoint costs, not an isolated-kernel or speedup claim. Raw receipts and
  the failed OH run are preserved under
  `benchmarks/results/issue172-r2scan3c-20260923/`.

## Consequences

The first d-shell force for a target pays finite source-generation and JIT
compilation cost; subsequent calls reuse hash-verified objects and the linked
artifact. Component expansion increases task descriptors and semantic primitive
work, both reported by `result.work`. Packaged s/p execution and its no-compiler
contract are unchanged.

## Revisit when

- A qualified packaged s/p/d AOT inventory can replace first-use JIT without
  expanding wheel size beyond its accepted budget.
- A resident Cartesian transformation can prove lower complete-endpoint work
  while preserving ordered weights and the same independent numerical gates.
- Heavy-element/ECP d-shell forces acquire their own reference and resource
  qualification.

## References

- GitHub issue #172
- `docs/developer/stationary_cuda_diagnostic.md`
- `.agents/notes/implemented/architecture/2026-09-19-stationary-cuda-diagnostic.md`
