# D4 numerical qualification baseline

`src/dft/dispersion/d4_reference.hpp` provides a bounded molecular D4
energy/derivative evaluator callable on CPU and from a CUDA device kernel.
It is an internal migration baseline, not a public DFT-D4 method or a
performance-promoted runtime. It has no runtime dependency on xTBloom,
DFT-D4, PySCF, Python, or Fortran.

## Supported contract

The caller supplies atomic numbers (H–Rn), Cartesian positions in bohr,
and independent partial charges. `gfn2_d4_parameters()` explicitly selects
the xTBloom/GFN2 compatibility profile; there is no generic DFT default.
Method parameters and the three sharp cutoffs are explicit. The input
reference model must be GFN2; EEQ is rejected rather than silently using
GFN2 reference polarizabilities.

`evaluate_d4_fixed_charge` returns the two-body energy, zero-charge-reference
ATM energy, the total Cartesian gradient at fixed charges, and the
charge derivative. Energies are Hartree, gradients are Hartree/bohr, and
charge derivatives are Hartree/electron. **Gradients are not forces.**
The Cartesian gradient includes all coordination-number response. Complete
DFT-D4 forces additionally need the selected charge model and its response:

```text
dE/dR = (partial E/partial R)_q + (dq/dR)^T (partial E/partial q)
force = -dE/dR
```

The data tables are GFN2-specific, not merely the charge input. Standard D4
requires independently qualified EEQ reference charges/polarizabilities as
well as an EEQ charge solver. Canonical r2SCAN-3c requires its exact parameter
manifest and the remaining electronic/basis/gCP components. None of those
public capabilities are granted by this baseline.

## Memory and execution

Each call owns one molecule of 0–256 atoms. Scratch requires exactly `27*N`
doubles; no pair matrix or triple tensor is materialized and the evaluator
does not allocate. All numerical buffers and immutable tables must be
disjoint. Outputs are committed only on success; disposable scratch may
change on a failed call. The caller must provide the documented array extents.

CPU and CUDA share migrated scalar mathematics, with an explicit table view.
Device consumers upload the immutable tables once and supply device-resident
geometry, charges, scratch and outputs. The CUDA tests run unequal packed
members concurrently and exercise an empty member and peer-local failure.
There is no production batch scheduler or public SCF integration yet.

Work is quadratic for coordination/two-body terms and cubic for ATM, with
reference interpolation work inside the loops. One CUDA lane per molecule
is a correctness baseline, **not evidence of competitive GPU performance**.
The cutoffs preserve the historical sharp GFN2 convention, which is not
differentiable at a cutoff crossing. There is no PBC, Hessian, or full
DFT-D4-force capability.

## Reproduction

The compact table stores only the symmetric lower triangle, including the
diagonal. Regenerate it without importing the runtime:

```sh
python tools/parameters/generate_d4.py \
  --source-git-dir /path/to/dftd4/.git \
  --revision 6e1f59c3f39d919a2dbef0601d2576727c8b30e8 \
  --output-dir src/dft/dispersion
```

CTest targets are `vibeqc_d4_reference_tests` and (CUDA builds)
`vibeqc_d4_reference_cuda_tests`. They do not link or build the main QC
library. Tests cover coordinate/charge finite differences, ATM decomposition,
signed s8, symmetry, table domains, bad inputs and transactional failure.
Independent energy and derivative fixtures from DFT-D4 4.2.0 are checked by
both targets. Their adapter, library hashes and package builds are retained in
`tools/oracle/` and `tests/data/d4/oracle_manifest.json`. Regenerate these
test-only fixtures with:

```sh
python tools/oracle/generate_d4_reference.py --prefix /path/to/reference-prefix
```

The CUDA target exercises real device arithmetic rather than calling a CPU
reference from a CUDA-labelled wrapper.

See the [migration decision](../.agents/notes/implemented/architecture/2026-09-19-d4-migration-baseline.md)
for ownership boundaries and retirement conditions.
