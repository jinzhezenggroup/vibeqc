# Decision: bounded, reusable reference grids for README GPU comparisons

Status: implemented
Date: 2026-09-22

## Problem

The README comparison extends DFT to 96 atoms while requiring both engines to
integrate the same discrete grid. `MolecularGrid.explicit()` targets small
references and exports in 256-point tiles. On the 96-atom PBE grid there are
2,654,208 points and 4,560 atom pairs: that tile size repeats the Python pair
loop 10,368 times, independently of the GPU solver being measured.

## Decision

The benchmark consumes the existing public `MolecularGrid.tiles(32768)` iterator,
reducing pair-loop invocations to 81 tiles. An ignored, atom/specification-keyed
NPZ cache shares this export between direct and DF and between functionals using
the same grid. Atomic cache publication permits host precomputation alongside a
queued GPU job. Grid export and cache I/O precede solver endpoint timers and are
reported separately. Native production grid execution remains unchanged.

## Invariants and evidence

The exporter must preserve point order, coordinates, weights and the discrete
energy. `test_large_export_tiles_preserve_reference_grid_and_cache` checks exact
array equality against the original small exporter both before and after cache
reload. Every supported DFT endpoint must still pass the independent GPU4PySCF
energy gate at `1e-8 Eh`, including cold and priming calls. Per-point records
retain molecular-grid identity and a hash of the actual point/weight arrays.

Do not replace the common grid with different engine defaults to obtain better
timings. These measurements qualify a shared discrete energy; they do not prove
independent grid refinement or change production grid defaults.

## Revisit when

A production export API supports larger grids efficiently, or the comparison
explicitly expands to include grid construction in its timing contract. See the
[measurement protocol](../../../../benchmarks/results/readme-20260922/README.md).
