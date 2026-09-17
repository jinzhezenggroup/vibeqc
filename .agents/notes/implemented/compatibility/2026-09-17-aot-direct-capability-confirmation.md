# Decision: distinguish compiled direct capability from numerical acceptance

Status: implemented
Date: 2026-09-17

## Problem and evidence

The source identity of the qualified DF library did not imply availability of
generated direct Fock kernels. The original AOT-disabled release reported
`generic_cuda` and failed 192-AO direct Fock construction with CUDA 801.
The earlier [binary-provenance decision](2026-09-17-benchmark-binary-provenance.md)
recorded this diagnosis before an AOT-enabled confirmation existed.

Slurm job 9872 confirms that an AOT-enabled Release / `sm_120` build at the same
native source identity completes both 192-AO batch sizes and passes every
original energy/full-force comparison across seven repeats. Both 96-AO strict
force checks still fail. Thus compiled capability explains the 192-AO execution
failure, while numerical acceptance remains a separate requirement.

## Decision and invariants

Keep loaded-library hashes and profile/build probes with both failures and
successes. Preserve the old failed matrix and the new matrix independently;
source equality must never relabel one binary's result as another's. No default
kernel policy, scientific precision, convergence control or error gate changed.
The complete direct matrix remains failed and #206 remains open.

Separate stock profiling accounts for wrapper-added fences and keeps iteration
branch mismatches visible. Profile memory includes intrusive overhead and cannot
stand in for clean memory acceptance. Neither operator counts nor thread-local
inclusive/exclusive scopes alone establish a fixed-work speedup.

## References

[Retained direct matrix, build/test qualification and stock profiles](../../../../benchmarks/results/issue206-aot-and-stock/README.md),
[#421](https://github.com/jinzhezenggroup/vibeqc/pull/421),
[#422](https://github.com/jinzhezenggroup/vibeqc/pull/422), and #206.
