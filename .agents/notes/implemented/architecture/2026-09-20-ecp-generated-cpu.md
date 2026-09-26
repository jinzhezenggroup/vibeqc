# Decision: share generated ECP mathematics with the production CPU provider

Status: implemented
Date: 2026-09-20

## Problem

After the CUDA ownership migration, CPU production still executed the independent
quadrature/projector implementation. That made independence useful for tests but
left two scientific production bodies under #171. The previous ownership note
required explicit CPU replacement and independent numerical acceptance before
retiring this production dependency.

## Decision

The existing CPU entry points now call the shared generated grid, AO, projector,
radial, pair, third-center, hcore-addition and two-grid policy helpers. Keep the
old mathematical implementation in tests/native/ecp_reference.cpp, with its own
Legendre construction, AO polynomial derivatives, harmonics, normalization,
radial powers, pair contractions and finite acceptance predicate. Only the new
native test executable links it. No public backend selector or C ABI changes.

Split the existing generated pair consumer into radial preparation and pair
contraction. CUDA retains its original wrapper and schedule; CPU hoists radial
preparation before the AO-pair loop, as before. There is one scientific formula
family in production, and a deliberately independent test-only oracle.

## Invariants and work

CPU still visits ECP centers, radial nodes and triangular AO pairs in ascending
order; each projection reduces angular nodes in the original order. Component
and primitive ordering, FP64 storage, complete A/B/C center scatter (including
coincident owners), local/nonlocal separation and derivative signs are unchanged.
Generated DAG reassociation can change last-bit rounding, so equality is numerical,
not bitwise. Raw exports keep caller-selected grids. Checked execution uses
160/32 and 224/44 and rejects nonfinite values or discrepancies above 2e-9
(values) / 2e-8 (derivatives) before publishing the refined result.

For C centers, R radial nodes, Q angular nodes, N AOs and M=16 projectors, the
upper bounds remain C*R*N*Q AO evaluations, C*R*N*M*Q projection samples and
C*R*N*(N+1)/2*(Q+M) pair terms. Radial terms are evaluated once per center/layer,
never once per pair. Zero-potential layers still skip all AO/pair work. Native
staging remains N*(Q+M)*4 doubles, independent of R; output storage remains
2*N*N*(1+3*A) doubles for derivatives. AO descriptors change from shell pointer
plus expansion vectors to fixed records (three components maximum), bounded by
256 AOs; grid vectors retain the same radial and angular shape. No full-grid
AO tensor, extra provider pass, primitive upload or production oracle call.

## Rejected alternatives

Calling the unchanged CUDA pair wrapper on CPU would repeat every radial term
for every pair. Deleting the oracle would leave generated-vs-generated as the
only native check. A runtime backend switch would enlarge the supported
production policy surface without a consumer requirement.

## Evidence and limits

The native suite compares all matrix and derivative entries at 2e-11*(1+|ref|),
checks translation and nonsymmetric-weight contractions, and compares all nine
center coordinates to independent energy differences at steps 2e-4 and 1e-4
with 2e-7*(1+|ref|) acceptance. It covers both representations and s/p/d/f,
independent ECP-center motion, both checked grids, nonfinite rejection/recovery,
changed geometry, replay, and an 80-AO case. Existing Libcint/complete HF/DFT and
stationary-gradient acceptance tests remain the full-method gates. Validation
results must be read from the PR's exact tested head; no GPU runtime or timing
improvement is inferred from CPU compilation or raw-provider timings.

This supersedes the production CPU fallback classification in
[the prior CUDA ownership decision](2026-09-19-ecp-generated-policy.md), while
preserving its requirement for an independent oracle. Moving scientific source
to a test oracle is production retirement, not deletion of all those source
lines. The CPU adapter remains runtime orchestration/scatter; the CUDA ledger
classification is unchanged. Higher-angular DFT forces and complete-endpoint
performance/schedule promotion remain separate #171 work.
