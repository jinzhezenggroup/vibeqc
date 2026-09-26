# Decision: reconstruct corrected streamed RHF response factors explicitly

Status: implemented for explicit opt-in; automatic promotion deferred
Date: 2026-09-23
Qualified through: 2026-09-25

## Decision

Keep `auto` on the bounded dense fallback. An explicit
`VIBEQC_DF_RESPONSE_SPACE=occupied` request may reconstruct a new algebraic
factor from a corrected final RHF density on singleton, full-rank streamed DF
plans when the occupied value schedule is admitted.

The route must match owner/basis/reference, solve epoch, model and occupations;
require bounded equal density/orbital generation advances; reject non-finite
input; reconstruct the complete density under the existing factor tolerances;
revoke the prior SCF factor generation before overwriting its scratch; and never
publish the reconstructed factor as a canonical SCF generation or symmetric-C
final projection. Any failed gate keeps the bounded dense response.

Automatic corrected-factor selection is intentionally outside this change.

## Correctness and provenance

The production implementation is in `src/scf/cuda/df_force_response.cpp`.
It validates the original final-state token, reconstructs the corrected density
with the existing device eigensolver, checks the full matrix rather than one
triangle, enforces rank and response-capacity bounds, and drains the borrowed
host upload on every rejected/error/exceptional exit. The accepted path avoids
an additional synchronization after the reconstruction drains.

The current PR head differs from the independently GPU-qualified production
source `cbfed56e` only by the holdout-test budget change in `fe1f4eb`; no
production source or numerical tolerance changed in that commit.

## GPU evidence

Fast-build RTX 5090 jobs 11500 and 11501 passed independent PySCF
cold/warm/changed energy and analytic-force gates. Under `auto`, corrected
factor admission is absent and dense AO response remains active. Under explicit
`occupied`, the corrected factor is admitted, occupied projection products
replace dense AO products, and no stale final projection is reused.

Composed Release sm_120 validation uses the exact candidate binary SHA-256
`a115a7f4990220454575c5bc35a6da4a179e74bd13b8c94f2c6910b0c259da3d`.
Slurm 11503 passed all ten independent PySCF composition gates. Slurm 11504
completed the original 96-atom / 768-AO / 3712-auxiliary cold, warm and changed
energy-plus-analytic-force endpoint under the 21,421,977,600-byte DF allowance
with 23/7/13 iterations. Candidate timings were 625.661 / 290.117 / 700.329 s;
the matched traced baseline was 626.047 / 290.256 / 1207.797 s. Maximum
independent errors were 5.23e-11 Eh and 1.50e-10 Eh/Bohr.

The changed response trace admits one rank-160 corrected factor, executes 7,424
occupied projection products, and generates one complete raw tensor instead of
25 response tensors. Complete changed-source work falls from 83 to 59
raw-tensor equivalents. Response scratch accounting falls from 7,736,157,200
to 1,980,523,536 bytes.

## Matched endpoint replication

Slurm 11511 is a clean untraced matched Release A/B using the same references,
allowance and 23/7/13 iterations. Baseline cold/warm/changed timings are
625.217 / 290.107 / 1206.340 s; explicit corrected `occupied` timings are
625.777 / 289.868 / 700.038 s. Independent errors remain below 5.23e-11 Eh and
1.51e-10 Eh/Bohr.

Slurm 11513 repeats the complete A/B with VibeQC tracing disabled while an
external 100-ms `nvidia-smi` sampler is active. Baseline timings are
626.657 / 290.480 / 1209.810 s; candidate timings are
626.714 / 290.388 / 700.979 s, again with 23/7/13 iterations and the same
independent numerical gates. Because this run includes external sampling, it is
replication of the endpoint direction and correctness, not a second clean
latency benchmark.

These measurements support a case-specific explicit-route benefit only. They
are not a cross-workload or default-selection speed claim.

## Memory qualification

The Slurm 11513 full-endpoint device-wide samples reached 19,403 MiB for the
baseline and 13,915 MiB for the corrected candidate. Sampling includes driver
and library usage and can miss transient peaks, so these values are not exact
allocator high-water measurements. They complement, rather than replace, the
exact response scratch accounting above.

The explicit route does not materialize a persistent complete raw or whitened
tensor; it borrows already reserved value-plan factor storage and remains
bounded by the response capacity checks. More allocator-level and cross-workload
measurements are required before automatic promotion or a general memory claim,
but they are not a blocker for this explicit opt-in implementation.

## Non-water streamed holdout

The original 16-MiB ammonia-trimer test was below the usable corrected occupied
projection schedule window: it remained streamed but correctly fell back to
dense response. A budget sweep on RTX 5090 established:

- 18 MiB: streamed and corrected occupied response admitted; pass.
- 20 MiB: streamed and corrected occupied response admitted; pass.
- 22/23 MiB: the value plan becomes resident and therefore no longer qualifies
  the streamed holdout.

Commit `fe1f4eb` changes only the ammonia holdout from 16 to 18 MiB. The full
water/ammonia x auto/occupied matrix then passes 4/4 on RTX 5090 in 35.13 s.
The `auto` arms continue to assert no corrected-factor admission, while the
explicit `occupied` arms require accepted corrected-factor traces and
occupied projection products.

## Default-selection boundary

The current qualification explicitly proves the default is unchanged on both
water and ammonia streamed workloads. Promotion of corrected reconstruction
into `auto` requires a separate policy change with broader cross-workload
performance/resource evidence. It must not be inferred from this PR.

DFT qualification tracked by #1117 remains separate and is not implied by this
RHF-DF result.

## Revisit when

Revisit the explicit route if the independent energy/force gates, full-density
reconstruction, token/generation provenance, bounded fallback or scratch
ownership fail. Revisit automatic promotion only with broader clean A/B
replication and resource evidence across representative workloads.
