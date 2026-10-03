# Decision: reuse private VV10 force panels across ordered phases

Status: implemented
Date: 2026-10-03

## Problem

The resident molecular nonlocal force owner keeps raw density/gradient,
effective density/gradient and point derivatives in distinct full-grid panels,
although their lifetimes do not overlap. At the README's 96-atom water cluster
(2,359,296 grid points), the owner consumes 566,378,516 bytes. Under the
unchanged 1 GiB device / 2 GiB host composite force budgets, this contributes
to selecting 128-point force tiles and 18,432 geometry batches.

## Decision and ordering

Allocate 15 rather than 22 full-grid I/O doubles per point, plus the unchanged
VV10 pair workspace and three sticky error flags. Both the dry capacity query
and the allocating owner use this same inventory.

1. Copy or collect density and Cartesian density gradient into private owner
   storage. Producer-stream D2D copies and the existing event handoff preserve
   the source lifetime. The source arrays are never modified.
2. Prepare the MolecularV1 domain in place. Each point's thread loads all its
   inputs before writing any effective output. Original quadrature weights
   remain separate: a later reset may unscreen a previously inactive point.
3. Build local scales on the owner's stream. This is the last reader of the
   density-gradient panel.
4. The ordered later pair kernels overwrite that dead three-channel panel with
   Cartesian point derivatives. They still read the separate effective density
   and pair-local scales; pair arithmetic and partner order do not change.
5. Pack point derivatives from AoS into the separate six-channel SoA seed
   panel. Consumers retain the existing borrowed-generation contract: finish
   reading seeds before resetting or releasing the owner.

The owner recovers exactly `7 * N * sizeof(double)` bytes: 126 MiB at 96 atoms.
No new stream, event, synchronization, oracle, host copy, or resident-state
identity is introduced. Tiled feature collection remains available when final
resident KS features cannot be borrowed. Public primitive allocations, caps,
precision, screening, SCF convergence and scientific parameter domains are
unchanged.

## Alternatives and invariants

- Aliasing original and effective weights would permanently lose screened
  weights, so unscreening after reset would be incorrect.
- Aliasing point derivatives with the SoA seed destinations would create
  inter-thread hazards during the AoS-to-SoA pack. They remain separate.
- Raising the public memory allowance hides the planner cliff without removing
  unnecessary storage. The existing bounded resource policy is retained.
- Cross-owner borrowing would require additional identity/lifetime proof. This
  change reuses only panels already private to this owner.

Future kernels must preserve the last-reader ordering above. In particular,
adding a density-gradient read after pair execution invalidates the alias.
Reset must repopulate every point before execution; invalid input must poison
every output channel, and a valid later generation must recover.

## Qualification and provenance

Base: master `9e9b938239d31564d6f92e9b99b922aee0dbb1cc`.
Baseline native SHA-256:
`35e12d25ad1d14bed1ea55ea1eb5fd63325c0b592bd27512ba3117bf08e2d0a3`.
Candidate native SHA-256:
`159c4e2d98ef936999e5108d9b0e17d855aaa8a66fa9310cbdaf2fdd24bfe450`.
Both Release CUDA 12.9/sm_120 builds use verified C++/CUDA ccache launchers.
Exact source snapshots, build/cache receipts and raw qualification results are
retained locally in `.artifacts/wb97m-force-storage/`, and copied explicitly to
n1 at `/data/jzzeng/wb97m-force-storage-20261003`.

n1 Slurm job 5484 qualifies the two binaries on one RTX 5090:

- Both binaries pass all 22 new phase-storage tests. An independent NumPy
  fixed-grid oracle checks all six feature/geometry channels at `rtol=2e-11`,
  `atol=1e-12` for VV10/rVV10, 1/129/257 points, signed/zero weights, threshold
  boundaries, reset/unscreening, tiled tails and distinct producer streams.
- Eighteen complete seed generations are bitwise identical between binaries.
- The candidate passes 16 capacity tests, including real allocations at the
  exact queried cap and rejection one byte below it. Full 48/96-atom grids
  require 217,128,980 / 434,257,940 bytes, respectively.
- Memcheck, initcheck and synccheck each execute all 22 phase tests with zero
  errors. Invalid density/gradient poisons all channels; reset recovers.
- Seven complete independent WB97M-V force, rebuild-failure and stale-snapshot
  tests pass, including displaced-geometry reference checks.

Focused host checks pass 92 tests with 29 explicit GPU skips. Sixty-eight
compiled host probes were repeated with a verified ccache alias after the
first alias pointed to an absent executable; that first probe run is not
claimed as cached. Native CMake builds always used verified cache launchers.

The real-basis capacity planner selects 256-point tiles and 9,216 geometry
batches for the candidate at 96 atoms, versus 128 / 18,432 for the baseline.
The respective device bounds are 956,819,828 / 1,041,987,956 bytes and host
bounds 2,005,033,492 / 2,107,925,012 bytes. These are resource/work plans, not
endpoint timings. At 24 and 48 atoms, both builds retain 1,024-point tiles.
Complete timing campaigns are separate qualification; memory savings alone
do not establish a speedup over the baseline or GPU4PySCF.

## Revisit when

Revisit the alias if later force kernels need raw/effective gradients again,
if domain preparation gains cross-point reads, or if consumer execution may
overlap reset on another stream. Any such change needs an explicit ordering
or separate bounded storage path and renewed independent/sanitizer checks.
