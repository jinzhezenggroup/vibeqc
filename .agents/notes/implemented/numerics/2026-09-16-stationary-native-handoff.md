# Decision: require the live native KS handoff for stationary authorization

Status: implemented
Date: 2026-09-16

Update (2026-09-18): the prohibition on binding this native SCF domain to any
generated XC geometry is superseded by
[the SCF-point/generated-pullback decision](2026-09-18-scf-point-generated-pullback.md).
The live-native authorization, replay, lifetime, and relabeling protections
recorded here remain current.

## Problem

PR #384's array records could be relabeled with arbitrary owner/epoch/basis
identities. Algebraic consistency cannot prove that a solve is current, that
F is the physical KS Fock, or that the overlap belongs to the XC AO basis.
The original H2 diagnostic even used a synthetic diagonal overlap. This note
supersedes the authorization and completion claims in
[the original boundary note](2026-09-15-stationary-dft-gradient-boundary.md).

## Decision

An internal ctypes bridge consumes the actual #162 token/read on a prepared
CUDA KS batch. It copies the provider overlap and the owner's normalized packed
AO basis, atom/model metadata and quadrature alongside D/F/C/epsilon/f/W. The
Python factory checks these actual sources before deriving identity labels.
The native snapshot retains no borrowed batch pointer: validation receives the
live Python-owned batch and queries its current token under the native context
lock. Copied arrays alone never authorize stationary consumption.
The Python handle binding is an immutable integer in a slotted owner: ordinary
assignment, deletion and mutation through a ctypes pointer alias cannot attach
a fresh native token to stale cached arrays. Only internal creation/closure may
set or clear the handle.

Array algebra and generated fixed-density directional diagnostics remain
separate from stationary authorization. Native SCF's scaled tail/spin domain
differs from generated `interior-v1`; preserving this identity deliberately
makes stationary generated XC binding fail until a matching derivative model
exists. This is groundwork, not completion of #163-A or public gradients.

## Rejected alternatives

- Comparing only copied epochs/hashes cannot detect a later solve.
- A Python-issued token or a caller callback would still authorize arbitrary
  physical-state claims without the actual method producer.
- Replacing only basis/overlap labels cannot identify the metric of D/F/C/W.
- Relabeling the native regularization as `interior-v1` would silently
  differentiate a different energy model, especially near empty spin/tail cases.

## Invariants

- Every stationary validation checks current eligibility and original content.
- Native replay, failed execution, owner replacement and closure revoke proof.
- Source checks precede XC contraction, and partial consumption rechecks proof.
- CPU/synthetic records are diagnostic data and cannot bypass native eligibility.
- The bridge is internal and does not alter installed public result layouts.

## Evidence

`test_dft_stationary_native.py` exercises real LDA/PBE RKS/UKS solves, replay,
failed requests, closure, wrong same-size bases, a self-consistent metric
transformation, regularization relabeling and attempted fresh-handle substitution
after replay. `test_ks_cuda.cpp` covers native
bridge invalidation and proves stale reads leave caller arrays untouched.
The fixed-density multistep oracle uses the true analytically integrated H2
overlap and checks separate/combined source motions and sign/omission controls.

## Consequences and revisit conditions

Snapshot export explicitly allocates detached host arrays and transfers native
final-state data; ordinary energy execution remains unchanged. Revisit generated
stationary binding only after an audited derivative shares the native SCF
energy domain, or another real method producer solves the exact generated model.
