# Current-density CPU feature sources (#235 A)

`vibeqc_compiler.dft.DensitySource` provides a fixed-input CPU contract for
choosing between the original density matrix D and compatible occupied or
fractionally occupied orbitals C/f. This is the A slice of #235, tracked in
#295. It builds on the [grid feature conventions](dft_grid.md) and supplies
the same feature dictionary consumed by XC contractions.

## Mathematics and spin conventions

For each real spin block, define `B = C sqrt(f)` with nonnegative occupations:

```text
D = B B.T
Y = Phi D                  Psi = Phi B
rho = sum(Phi * Y)          rho = sum(Psi**2)
grad_k rho = 2 sum(Phi_k * Y)
                           grad_k rho = 2 sum(Psi_k * Psi)
tau = 1/2 sum_k sum((Phi_k D) * Phi_k)
                           tau = 1/2 sum_k sum(Psi_k**2)
```

Summations reduce the complete AO or orbital dimension before forming
`sigma = (grad_a², grad_a dot grad_b, grad_b²)`. Cross-spin sigma has no
additional factor two. A total matrix `[AO,AO]` splits equally between alpha
and beta. Orbital inputs always carry **per-spin** occupations; an RHF total
occupation of two becomes one in each spin channel. Occupations are never
renormalized or inferred from electron count. B is formed before collocation
so a zero occupation does not multiply an overflowing unweighted square.

Both rectangular `[2,AO,norb]` inputs and pairs of `[AO,norb_spin]` arrays work.
An empty spin channel has shape `[AO,0]` and occupation shape `[0]`. Orbital
columns are not truncated. An explicit `ao_ids` map restricts D to `D[I,I]`
and C to `C[I,:]` in the same order, retaining all cross terms and occupied
columns. Unsorted unique maps and empty local supports are supported.

## Identity, validation and replay

```python
from vibeqc_compiler.dft import DensitySource

# basis.identity includes coordinates, normalized basis, AO representation,
# atom order, charge and spin policy. D is this producer state's density.
current = DensitySource(
    D, basis_identity=basis.identity,
    basis_generation=3, density_generation=17,
)
candidate_stamp = current.stamp  # capture with C/f at their production time
checked = current.with_orbitals(C_spin, f_spin, stamp=candidate_stamp)
features = checked.features(
    jets, stamp=current.stamp, route="orbitals",
    ingredients=("rho", "gradient", "sigma"),
)
```

`DensityStamp` identifies the normalized D content and total/separate layout,
basis content, basis generation, density generation, and state/response role.
The existing `spin_densities` validator preserves signed D and symmetrizes only
accepted roundoff asymmetry. Every array has its own immutable backing bytes;
new sources with attached factors share the immutable original D. An accepted
`factor_identity` hashes both C/f spin blocks, their shapes and their stamp.

The caller must carry the stamp with the orbital snapshot and supply the
**current consumer's** stamp on replay. An old stamp cannot be relabeled merely
because dimensions match. Same-density producer updates still advance the
density generation; changed geometry advances the basis generation. Creating
a new source computes a content hash as a further check on changed matrices.
There is no trusted-producer bypass in this external-input CPU contract.

Attachment compares every entry of `B B.T` to the actual D, using
`atol=1e-12, rtol=1e-10`. This is numerical compatibility at the existing
density tolerance, not exact-arithmetic equality or a universal bound on
subsequent XC errors. Compatibility uses row panels of at most
`validation_rows` (default 64); it costs O(NAO²*norb) once per attachment.
`validation_max_abs_error` reports the largest accepted entry error. Replay
checks the stamp without reconstructing D. This CPU reference deliberately
does not infer an eigendecomposition or repair eigenvalues to manufacture C.

`with_orbitals` returns a new source. Rejected candidates clear any previously
accepted factors; the original source remains usable. Diagnostics distinguish
`missing_orbitals`, `stale_orbitals`, `invalid_orbitals: ...`,
`incompatible_orbitals` and `response_density`. Invalid coefficients include
complex/nonfinite data, shape mismatches and negative occupations. A source
declared `role="response"` always keeps D, including when a particular response
happens to be positive. Missing or rejected C never changes the original D.

`route="auto"` selects by availability in this CPU reference only: validated
C, otherwise D. It is not a cost model or production performance promotion.
Explicit `density_matrix` and `orbitals` routes support parity tests; forcing
an unavailable orbital route raises with its fallback reason. Replaying a
stale **source** raises instead of falling back to its equally stale D.

## Requested outputs, storage and integration boundary

Both `density_features` and `orbital_features` accept `ingredients` drawn from
rho, gradient, sigma and tau. Rho alone accepts value-only AO jets; requesting
sigma computes the needed gradient internally. Tau requires first derivatives
and is omitted entirely when unused. Ordinary derivative orders 0–3 remain
supported within the requested domain. Higher supplied jets are not contracted.

Retained storage includes both spin D matrices, C and occupations. Attachment
additionally holds one weighted factor O(NAO*norb) and validation temporaries
O(validation_rows*NAO), including matrix/error/comparison panels. Tile execution
uses local D or C gathers, input-validation copies, weighted factors, and up to
four point-by-orbital panels per spin. The caller supplies a bounded point tile;
this helper never assembles a complete molecular grid or stores tile history.
It is not a composed memory-budget guard, and does not remove D storage.

Native `src/scf/density_factor.hpp` already owns the SCF/RI-K factor contract:
its integer occupations, exact density witness, native reference/orbital IDs
and restricted-spin convention remain authoritative there. This CPU external
reference adds fractional-spin mathematical acceptance without changing native
SCF semantics. A future #235 B/C adapter must reuse that producer provenance
and map restricted occupations explicitly, then compose resources under #203.
Native GPU collocation, prepared/local-task integration, geometric derivatives,
and candidate selection under #168 remain future slices. This PR supplies no
new molecular method, solver, force capability or speedup claim.

## Validation

`tests/python/test_density_source.py` consumes the existing hash-checked PySCF
fixtures for H2, water, Cartesian/spherical f shells, diffuse and tight bases.
It checks every rho/gradient/sigma/tau element on identical saved AO jets at
`atol=1e-11, rtol=1e-10`; no fixture regeneration or PySCF import is required.
Further tests cover fractional/empty spins, signs and equal-occupation rotations,
three-step density directional differences, local cross terms, AO nodes,
zero/tiny occupations, invalid and stale factors, immutable state and replay.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/python/test_density_source.py -q
```
