# Decision: retain loaded native capabilities before benchmark execution

Status: implemented
Date: 2026-09-17

## Problem

The final direct acceptance attempt in #421 failed at 192 AO before any usable
sample. Its native source identity matched the qualified DF source, but the
library was built with `VIBEQC_ENABLE_AOT_SHELLS=OFF`. A source hash does not
encode the selected generated-kernel profile. Returning only a generic CUDA
error made this distinction hard to diagnose.

A Slurm/GDB observation localizes CUDA 801 to the first direct Fock construction.
The bounded consumer explicitly refuses shell classes without a compiled exact
consumer. The failed library reports `generic_cuda`. An AOT-enabled release
reproduction is separate pending work; these observations do not certify it.

## Decision

The batch comparator records the actual loaded library path and SHA-256,
native source identity, selected profile and release/fast-compile/device metadata
using the existing native device probe. Capture this before preparation/SCF and
outside endpoint timing. The four-point direct and separate DF runners always
request a fresh per-point progress journal, linked from their summary even when
the result JSON cannot be produced.

Use the calculator's loaded library rather than the environment's requested
path: a selected local profile may replace that path. Device ordinals remain
process-local and preserve Slurm visibility. A profile name is provenance, not
an assertion that every molecule or consumer is supported.

## Invariants

All original cases, numerical gates, SCF settings and failure verdicts remain.
Never substitute a later build's identity for a retained sample. A successful
metadata regression does not turn the deliberately failing molecule into a
passing calculation. Fresh journals cannot append to a previous attempt.

## Evidence and remaining work

The loaded-profile regression verifies that an environment pointing at another
binary cannot change the recorded loaded-library identity or device ordinal.
Four focused CPU tests pass. A separate Slurm regression retains both failed
192-AO direct points and their original binary/profile metadata.

[Retained diagnosis](../../../../benchmarks/results/issue206-direct-diagnosis/README.md)
also compares saved forces to tighter CPU oracles and preserves separate
convergence-control probes. Native 96-AO direct and stock changed-geometry
results exhibit different convergence sensitivity. These accuracy-only probes
do not replace original timing, change the acceptance settings, or qualify a
performance claim. #206 remains open.
