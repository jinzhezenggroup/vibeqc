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
across cold/warm/changed prepared replay and a forced final correction. The
The strengthened assertions repeat the two passing tests in job 11501
(`/tmp/qc-1117-response-fast-gpu-gate-final.log`). `auto` emits zero corrected factor admissions, 192 dense AO products
per phase and zero occupied projection products. Explicit `occupied` emits one
accepted corrected factor per phase, 192 occupied projection products, zero
dense AO products and no reused final projection. These are fast-build
correctness/work diagnostics, not Release endpoint latency evidence.

## Revisit when

The complete endpoint, an independent force oracle or the bounded fallback
fails; or a separate candidate can prove automatic selection profitable across
supported workloads without weakening factor provenance.
