# Decision: Capture primary streaming selection as a device kernel argument

Status: implemented
Date: 2026-09-26

## Problem

PR #1365 removed selected generated classes from bounded Fock pages and
uploaded their streaming flags from an automatic host array inside graph
capture. This introduced two independent CUDA failures: a device-launch graph
rejects that pageable H2D node, and a host-launch graph retains the address after
the producing call returns. A missing streaming flag then omits a class already
excluded from the pages. Selecting another mask also failed to invalidate the
captured paged/streaming partition.

## Decision

Initialize the entire fixed-size flag array on device from a scalar mask passed
by value to a small queue-metadata kernel. Graph capture owns the kernel argument;
it borrows no host buffer. Keep the existing memset for the default zero mask.
The parsed requested mask participates in both cached-plan invalidation and the
driver's defensive identity check, so changing it recaptures both routes.

The mask still intersects generated streaming capability and remains disabled
under mixed Fock precision. It changes scheduling, not equations or screening.
Invalid numeric selectors fail closed: signs, whitespace, overflow, trailing
characters and zero cannot select classes; unsigned decimal/octal/hex and `all`
retain their existing nonzero meanings.

## Rejected alternatives

- Extending only the lifetime of the ordinary host array would leave the
  device-launch graph incompatibility unresolved.
- Updating only streaming flags would leave the captured page exclusion set
  stale when a prepared plan changes its selector.
- Silently accepting `strtoull` overflow or a whitespace-prefixed negative
  would unexpectedly select all or most supported classes.

## Evidence and acceptance gates

The pre-fix CUDA 12.9.86/RTX 5090 reproduction (Slurm job 11749) had a passing
device-only control, `invalid argument` on device-launch graph instantiation
with the H2D node, and a warm host replay changing class flag 1 to 0 after legal
stack reuse. These are API/lifetime results, not molecular performance claims.

`test_direct_streaming_graph_cuda.cpp` exercises the actual production launcher
after the capture call returns, resets poisoned selected/unselected flags on
every replay, and checks both host and NVIDIA device-launch instantiation.
`test_direct_streaming_route_cuda.py` checks RHF/UHF Cartesian/spherical
libcint energy/force parity, a two-system bucket, changing selectors on one
prepared plan, warm replay, geometry changes and final-density work counts.
Native precision-policy tests cover parser bounds and stale errno.

Large-AO per-class performance promotion remains a separate gate: no class is
enabled by default and these fixes make no 384/768-AO speedup claim.

## Revisit when

If routing becomes mutable without recapture, update the page partition and
streaming selection atomically under one graph/plan identity and preserve the
device-launch and replay-lifetime tests.

## References

- PR #1365, stacked on #1358 and #1354.
- `src/scf/cuda/direct_queue_scan.cu`
- `src/scf/cuda/rhf_bucket.cpp`
- `src/scf/cuda_rhf.cpp`
