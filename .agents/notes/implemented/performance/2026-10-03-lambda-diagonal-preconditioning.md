# Decision: right diagonal preconditioning for native Lambda

Status: implemented
Date: 2026-10-03

## Problem

After native CUDA triples response, the 28-AO complete force endpoint still
spent about 2.6 seconds in corrected Lambda and parameter response. The shared
Lambda transpose accounted for 840 actions / 325,920 generated kernels across
four complete calls. Shared GMRES already supported a right diagonal
preconditioner, but neither native Lambda owner supplied one.

## Decision

Supply the physical approximate Jacobian diagonal `-d1, -d2` to shared GMRES
on CPU and CUDA. A diagonal commutes with the packed doubles coordinate
weights: multiplying by `sqrt(2)` again would be incorrect. Use the existing
pair representative and partner maps to admit the doubles diagonal.

A denominator at or below `max(1e-10, gmres.breakdown_tolerance)` in magnitude,
or a pair orbit inconsistent within `1e-10 * (1 + max(abs(d2)))`, disables the
preconditioner for that solve. Nonfinite problem data retain the existing
validation failure. Internal `diagonal_preconditioning=false` explicitly
selects the old solver.

Allocate the already-admitted independent-audit vector before GMRES, borrow
it as the diagonal, then overwrite it for the physical audit. This adds no
numeric owner, device allocation, device transfer or capacity requirement.
Completed-force diagnostics expose the selected path and actual Krylov work.

## Rejected alternatives and invariants

Do not clip small denominators, change the physical operator, or relax a
residual gate to obtain convergence. The shared true-residual check and the
separately expanded physical Lambda audit remain mandatory. Preconditioning
changes only Krylov coordinates; it does not establish same-space orbital
invariance or expand the supported force domain.

A separate preconditioner vector would consume additional simultaneous memory
without improving ownership. Reusing an amplitude or residual buffer would
risk corrupting a live scientific input; the later audit vector is safe because
its first scientific write happens after GMRES returns.

## Evidence

Retained [publication](../../../../benchmarks/results/cc-lambda-preconditioner-20261003/publication.json)
contains all 112 complete cold/warm/changed calls, independent oracle errors,
paired timing samples and semantic work counters. Six alternating process
pairs on the n5 RTX 5090 give:

| Endpoint | Baseline warm | Preconditioned warm | Paired speedup (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| 14 AO | 0.834637 s | 0.448712 s | 1.860 (1.857–1.863) |
| 28 AO | 6.731390 s | 4.774051 s | 1.410 (1.407–1.412) |

The interval resamples whole adjacent pairs, averaging the two warm calls in
each process. These observations do not predict other hardware/basis sets.
The 28-AO Lambda solve decreases from 105 iterations / 210 actions to
24 iterations / 48 actions. On n2, Nsight independently counts 74,496 shared
transpose kernels across four calls. Its measured owner-stream transfers
match the complete Lambda transfer diagnostics. Canonical quartets, derivative
consumers, primitive records, triples pages and public capacity fields remain
unchanged. All GPU work uses finite Slurm allocations and assigned visibility.

The native independent noninteracting oracle includes nonzero external singles
and doubles sources with distinct orbital gaps. It exercises off-diagonal pair
orbits, CPU/CUDA, exact admitted capacity, disabled preconditioning, zero/tiny
singles denominators and asymmetric doubles denominators. The preconditioned
case takes one iteration; original/fallback cases take 24. Both match the
analytic source/denominator solution within 1e-11. CUDA native tests and four
complete 14-AO force calls pass compute-sanitizer memcheck.

Complete molecular comparisons require energy error <=3e-9 Hartree, triples
error <=2e-9 Hartree, maximum force error <=1e-6 Hartree/Bohr, and response
residual <1e-9. Independent reference values come from pinned PySCF analytic
gradients; PySCF is never a production dependency. Host allocation tests protect
the complete budget, and changed translation units also compile without CUDA.

The frozen parent library is the #1747 measured library. Later parent changes
only defer a NumPy import (byte-identical generated CC sources) and repair a
standalone test harness; the frozen candidate similarly predates that harness
repair. Library, source archive, source patch and probe hashes are retained.

## Evidence storage and environment

The 64-MiB benchmark aggregate cap had only 744 bytes available. Whitespace-only
JSON compaction of six preceding owned CC families reclaimed 233,067 bytes.
All parsed scientific values are preserved; canonical parsed equality was
asserted, attachment hashes/publication byte counts were refreshed, and exact
large-file review entries were updated. Source patches and retention caps are
unchanged. The summary retains before/after hashes and byte counts.

n1 had no free filesystem space, so qualification moved to n5. Its default
CUDA runtime discovery selected an older cuSolver lacking the symbol needed by
the frozen library. Task-local Python vendor-library symlinks point to CUDA
12.9.1; exact resolved library hashes are retained. The system and repository
loader were not changed. Missing test-only PySCF dependencies were installed
privately for the full public oracle suite on n2.

## Consequences and revisit conditions

The host diagonal preparation is linear in packed response dimension, and the
host GMRES arithmetic remains. The improvement removes repeated scientific
operator work without changing generated kernels or their numerical order.
This is numerical qualification plus paired performance evidence, not a full
automated compiler-promotion envelope with measured process memory/compile cost.

The public force domain remains 28 AOs. At 56 AOs, same-space degeneracy needs
an independently validated gauge-invariant triples/orbital response; a better
Lambda solver does not resolve that frontier. Revisit the preconditioner if
strongly coupled systems worsen convergence, retaining the explicit original
path and both independent acceptance gates.
