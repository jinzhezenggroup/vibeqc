# Decision: freeze DFT-MP-v1 before precision promotion

Status: implemented
Date: 2026-09-24

## Problem

An AUTO selector, small fixture or kernel speedup can appear successful while
the public energy and force call runs strict FP64, uses a different SCF root,
or moves preparation outside timing. Prior water-only tuning could also bias a
performance claim if losing non-water cases were selected afterward.

## Decision

The #1185 manifest fixes molecule coordinates, changed geometries, basis/grid
identity, method variants, mandatory row inventory, resource budget and
statistics before experiments. It separates strict capability, executed mixed
work and promoted net benefit. Product PASS needs all required rows on one
upstream-merged source revision with #1190 review. The row validator consumes
the native precision provenance and requires a reconciled event/operator
ledger, separate independent-oracle results and source-matched strict timing.
The common explicit tight PBE-derived GridSpec is a frozen comparison model,
not a statement of each method's default grid. Optional B3LYP/O2 stress rows
remain visible and cannot replace mandatory RKS/UKS coverage.
Versioned text input and basis hashes use LF-normalized Git-blob bytes so a
Windows CRLF checkout and a Linux checkout retain the same contract identity.
RDKit-derived geometry reconstruction is qualified by the recorded OS, Python
ABI and exact wheel metadata hash. Default regeneration is a read-only audit;
new candidates use a separate output tree and cannot overwrite v1 inputs.
Final merged-source acceptance resolves the official repository's advertised
master OID directly; remote aliases such as `origin` are not trusted identity.
The runner records an in-flight row before starting external work; every
attempt retains hashed stdout, stderr and a per-sample progress ledger. A
crash leaves the campaign stopped until the old process tree is independently
confirmed ended, rather than risking overlap or mixing attempts. Unconfirmed
timeout cleanup also stops the campaign. #1190 owns the installed-production
adapter and final campaign plan.

## Rejected alternatives

- Treating `requested_mode=auto` as proof of mixed arithmetic: fallback can
  execute zero FP32 operators.
- Pooling cold with changed geometry or dropping a losing holdout: either can
  create a gain without a complete endpoint improvement.
- Borrowing H100 results for RTX 5090/sm_120 or DF energy for direct E+force:
  neither is the same product domain.
- Treating a hash-checked receipt as cryptographic proof of execution: a
  dishonest adapter can fabricate JSON, so raw review remains a separate gate.

## Invariants

Total-energy and maximum force-component errors retain the `1e-7 Eh` and
`1e-6 Eh/bohr` limits or tighter existing gates. Independent oracle, finer-grid
and multi-step reconverged finite-difference checks remain separate. Missing,
timed-out and unsupported required cases cannot aggregate as passes. Optional
non-pass cases remain reported, while invalid optional passes are rejected.
Any amended contract gets a new version/hash and an explicit affected-gate
migration map.

## Evidence

`tests/python/test_dft_mp_v1_contract.py` exercises missing rows, timeout,
stale raw hashes, wrong state/oracle, absent real mixed work, numerical errors,
finite-difference/grid negatives, and retained runner journals. No production
GPU result is claimed by these tests.

## Revisit when

Real allocated production experiments show a necessary equivalent input or
method amendment. Preserve the core accuracy, performance and single-revision
acceptance goals, and document every affected row before rerunning.

## References

#1185, #1190, #1191; `tools/dft_mp_v1/manifest.json`.
