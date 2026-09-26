# Decision: retire handwritten ssss force mathematics behind the qualified native scheduler

Status: implemented
Date: 2026-09-20

## Problem

The standalone generated sm_120 ssss force consumer was numerically qualified,
but routing production ssss through a separately materialized generated task
queue caused small-endpoint overhead. Complete endpoint A/B also showed a
workload crossover rather than a universal generated win.

The first native-scheduler adapter removed that second queue and selected
generated mathematics inside the already qualified bounded/direct scheduling
path. RHF endpoints then became essentially tied, while the tested UHF endpoint
moved from a regression to a small generated win.

## Decision

Keep scheduling/runtime ownership native and make the ssss scientific derivative
algebra unconditionally compiler-owned. Remove the VIBEQC_SSSS_FORCE selector,
prepared-plan route bit, A/B branch, handwritten contracted ssss derivative body,
and its handwritten result type. The standalone generated AOT consumer may remain
compiled as generated validation/resource evidence, but production ssss force work
uses the qualified native scheduler with generated mathematics directly.

For the native adapter, lower only the outputs the caller consumes. The ssss
force helper has no integral-value root and emits only independent centers
0/1/2. Center 3 is recovered by the caller through translational invariance.
The adapter therefore does not initialize center-4 product-scale/decay state or
zero-initialize the generic geometry record.

## Invariants

- RHF/UHF density symmetry factors, screening, atom accumulation and recovered
  center semantics remain unchanged.
- The force-only helper is derived from the same weighted-ERI graph; no second
  handwritten ssss derivative formula is introduced.
- Generated selection must not create a second materialized task queue for ssss.
- There is no handwritten production ssss force-math fallback; unsupported or
  failed execution remains an explicit failure rather than silent formula switching.

## Evidence before force-only root pruning

On RTX 5090 at source bf8f2688 with the native-scheduler adapter, generated
versus native warm energy+force medians were approximately: RHF 96x1 -0.09%,
96x4 -0.20%, 192x1 -0.19%, 192x4 -0.19%, 384x1 -0.25%, 768x1 -0.16%;
the tested UHF 19-AO batch-4 endpoint was +0.70%. Numerical deltas remained
near machine precision for the compared routes.

The remaining adapter still emitted an unused integral value and recovered
center-4 derivative. The force-only lowering removes both roots at graph
materialization time rather than relying on downstream CUDA dead-code
elimination. Focused codegen tests pass; GPU endpoint requalification is
required before interpreting this as a performance win.

## Retirement result

The complete A/B evidence showed numerical parity and an RHF tie within roughly
0.1--0.25%, while the tested UHF endpoint improved. The force-only root pruning
remained neutral, so no further special-case micro-optimization was required.
The user authorized promotion and deletion of the handwritten path.

---
Agent: ChatGPT
Model: GPT-5.6 Sol

## Force-only root-pruning follow-up

The specialized low-order header now emits `ssss_force` from exactly nine DAG
roots: centers 0/1/2 times xyz. It emits no integral value and no recovered
center-4 root. The native adapter also stops initializing center-4 product
scale/decay fields and uses an uninitialized Geometry record with every reachable
field assigned explicitly before the force-only helper.

To avoid unrelated artifact-identity churn, IndependentGradient exists only in
the low-order header. The ordinary psss weighted header is byte-identical to
the bf8f2688 baseline (6976 bytes in both outputs).

Post-change checks: full test_codegen.py 296 passed / 48 skipped; weighted-ERI,
range-separated and independent native first-derivative tests passed when run
with the built library. A 192-AO batch-4 endpoint run experienced a machine-load
step between ABBA halves; within each adjacent native/generated pair generated
was -0.09% and -0.13%, with dE 6.82e-13 Eh and max dF 1.53e-13 Eh/bohr.
This is a tie/noise result, not evidence of an endpoint speedup. It suggests
the CUDA compiler was already eliminating much of the unused full-result work.

## Post-retirement validation

Release/sm_120 CUDA rebuilt successfully after removing every selector/route-bit
call site, including bounded fallback and angular/packed dispatch. Full codegen
regression: 296 passed / 48 skipped. Weighted-ERI, range-separated and independent
native first-derivative regressions: 71 passed / 44 skipped.

Default-path RTX 5090 smoke covered RHF 96 AO, RHF 192 AO batch 4, UHF 19 AO
batch 4, and RHF 768 AO. All reported one warm SCF iteration. The 768-AO warm
energy+force median was 5.616 s under the historical-style control; this is a
functional retirement smoke, not a new global performance claim because other
node workloads were not controlled as a clean timing campaign. Setting the old
VIBEQC_SSSS_FORCE variable no longer changes resource identity or code selection;
96-AO and UHF comparisons differed only at floating-point replay noise.

The reproducible CUDA ownership report changes all handwritten scientific lines
from 13,491 at bf8f2688 to 13,471 after retirement, a net reduction of 20 ledger-
classified scientific lines. Physical source deletion is larger because mixed
files include runtime/policy lines classified separately.
