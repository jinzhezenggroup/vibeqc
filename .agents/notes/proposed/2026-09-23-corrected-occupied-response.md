# Decision: reconstruct corrected streamed RHF response factors explicitly

Status: proposed (independent GPU acceptance pending)
Date: 2026-09-23

## Problem

Strict final-state correction changes the determinant after SCF convergence.
Its density has no matching canonical SCF factor generation; borrowing the old
factor would produce incorrect analytic forces. On the 96-atom HF-DF endpoint,
the independently qualified changed-geometry reference requires this correction,
and the bounded dense force response consumed 617.334 seconds out of the
1,207.797-second endpoint (Slurm job 11499). The 83 raw-tensor-equivalent
count includes 25 force-response passes; this is not an isolated SCF bottleneck.

## Candidate decision

Keep `auto` on its unchanged bounded fallback. An explicit `occupied` request
may reconstruct the corrected full RHF density with the existing device
eigensolver and charged value-plan factor scratch, but only for a singleton,
streamed, full-rank generated source with an admitted occupied value schedule.
Match owner, solve epoch, model and occupations to the current SCF state;
require a bounded, equal density/orbital generation advance and finite input.
Verify the complete reconstructed density before using the new algebraic factor
with unit scale. Clear the old SCF factor generation before overwriting its
scratch; do not publish a canonical factor or symmetric-C final projection.
Rejection retains the bounded dense force response.

## Rejected alternatives

- Lending the preceding SCF factor would give a different density and violate
  exact force-response provenance.
- Enabling corrected factorization under `auto` would change default endpoint
  work before complete numerical and performance qualification.
- Materializing a persistent full raw/whitened tensor exceeds the streamed
  value-plan ownership and obscures the actual response work count.

## Evidence and acceptance gates

Candidate implementation: `src/scf/cuda/df_force_response.cpp`. Require
independent PySCF/GPU4PySCF energy and analytic-force acceptance for cold,
warm and changed geometry, plus accepted-factor trace, occupied projection
products, dense fallback when reconstruction/provenance fails and safe replay
after SCF factor invalidation. Measure the composed 96-atom cold/warm/changed
endpoint against the same 21,421,977,600-byte allowance and reference as
PRs #1139 and #1147; trace raw source work separately from final Fock and
force response. A shorter isolated response does not establish an endpoint
speedup. The baseline trace is `/tmp/qc-1117-composed-96-diagnostic.log`.

The fast-build singleton 96-AO water-tetramer gate passes two Slurm GPU tests
(job 11500, `/tmp/qc-1117-response-fast-gpu-gate.log`) with independent PySCF
energy and analytic forces at 1e-9 Eh and 1e-8 Eh/Bohr acceptance thresholds,
across cold/warm/changed prepared replay and a forced final correction.
Strengthened assertions repeat the two passing tests in job 11501
(`/tmp/qc-1117-response-fast-gpu-gate-final.log`). `auto` emits zero corrected
factor admissions and 192 dense AO products
per phase and zero occupied projection products. Explicit `occupied` emits one
accepted corrected factor per phase, 192 occupied projection products, zero
dense AO products and no reused final projection. These are fast-build
correctness/work diagnostics, not Release endpoint latency evidence.

Composed Release sm_120 validation applies this source patch to PR #1139 and
the final-state source patch from PR #1147. Its library SHA-256 is
`a115a7f4990220454575c5bc35a6da4a179e74bd13b8c94f2c6910b0c259da3d`;
the exact tested artifact is retained at
`/tmp/qc-1117-corrected-composed-release.so`, because the #1139 build
worktree was restored to its original sources and binary after validation.
Slurm 11503 passes all ten independent PySCF cold/warm/changed GPU gates for
the composed value/final/response combinations. Slurm 11504 completes the
original 96-atom, 768-AO / 3712-auxiliary, 21,421,977,600-byte DF allowance
and independent GPU4PySCF energy-plus-analytic-force reference in
`/tmp/qc-1117-corrected-96-release-endpoint.log`. Cold, warm and changed
endpoints take 625.661, 290.117 and 700.329 seconds at 23/7/13 SCF
iterations. Maximum absolute energy and force errors are 5.23e-11 Eh and
1.50e-10 Eh/Bohr. The previous composed Release traced job 11499 takes
626.047/290.256/1207.797 seconds with matching 23/7/13 iterations and
the same reference and allowance: changed endpoint work drops 507.468 s
(42.02%); the three endpoints together drop 23.92%. Both jobs enable the
same detailed DF trace; these are *traced* complete endpoints, not a clean
untraced A/B timing claim.

Changed-geometry response in `/tmp/qc-1117-96-11504-changed.jsonl` admits
one rank-160 corrected factor after three tagged generations, verifies its
full reconstructed density, and executes 7424 occupied projection products
without using a stale final projection. It generates one full raw tensor
(17,515,413,504 bytes) and takes 110.258 s; the job-11499 dense response
generates 25 full tensors (437,885,337,600 raw bytes), 7424 AO products and
takes 617.334 s. The shared J, final J and final K operation counts and raw
bytes are unchanged; the complete changed endpoint falls from 83 to 59
raw-tensor equivalents. The response's separately charged scratch falls
from 7,736,157,200 to 1,980,523,536 bytes, but neither trace is a sampled
whole-endpoint device-memory high-water mark.

Before automatic selection or a general production speed claim, collect
untraced matched Release A/B endpoints, actual whole-endpoint peak memory,
and a structurally different holdout. DFT qualification under #1117 is
separate; this HF-DF result cannot close the tracker.

## Revisit when

The complete endpoint, an independent force oracle or the bounded fallback
fails; or a separate candidate can prove automatic selection profitable across
supported workloads without weakening factor provenance.
