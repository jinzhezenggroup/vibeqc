# RCCSD(T) energy and internal analytic-gradient validation

`tools.vibeqc_cc/triples.py` provides the auditable standard closed-shell
non-iterative (T) energy definition used by RCCSD(T). Bounded CUDA triples, the
generated native CPU energy evaluator, and generated response paths share that
definition. `VIBEQC_METHOD_RCCSD_T` now has a native/public CPU energy owner and
homogeneous prepared-batch support; public analytic forces and the native CUDA
owner remain fail-closed. PySCF is used only by pinned validation tooling and is
never a runtime dependency.

## Mathematical contract

Closed-shell RHF reference, all electrons active, real canonical MOs in
PySCF numbering (occupied then virtual). Indices i,j,k (occupied), a,b,c
(virtual), f,m (summed). All quantities are full-MO and Hartree:

```text
ovvv[i,a,b,c] = eris.get_ovvv()[i,a,b,c]     (i occupied; a,b,c virtual)
ovoo[i,a,j,m] = eris.ovoo[i,a,j,m]
ovov[i,a,j,b] = eris.ovov[i,a,j,b]           (= (ia|jb))
fov[i,a]      = Fock[o,v];  fvo = fov.T      (= F[v,o])
t1[i,a], t2[i,j,a,b] (pair-symmetric), eps_o[i], eps_v[a]
```

All three energy entry points reject non-finite values in every input array,
nonnegative denominators, and denominators whose magnitude is at or below
`denominator_threshold` (default `1e-10` Hartree). Canonical RHF orbitals are a
caller precondition: occupied/virtual energy ordering alone cannot establish
that the full Fock matrix is diagonal. The threshold accepts Python and NumPy
real numeric scalars; booleans are not valid tolerances.

With `t2T = t2.transpose(2,3,0,1)`, `eris_vvov = ovvv.transpose(1,3,0,2)`,
`eris_vooo = ovoo.transpose(1,0,2,3)`, `eris_vvoo = ovov.transpose(1,3,0,2)`
and `fvo = fov.T`, the label seeds for one virtual triple (a,b,c) are:

```text
W(a,b,c)[i,j,k] = sum_f vvov[a,b,i,f] t2T[c,f,k,j]
                - sum_m vooo[a,i,j,m] t2T[b,c,m,k]
V(a,b,c)[i,j,k] = vvoo[a,b,i,j] t1T[c,k] + t2T[a,b,i,j] fvo[c,k]
r3(w) = 4 w + w.transpose(1,2,0) + w.transpose(2,0,1)
      - 2 w.transpose(2,1,0) - 2 w.transpose(0,2,1) - 2 w.transpose(1,0,2)
```

The (T) energy is a triangular virtual sum (a >= b >= c) with an exact
degeneracy prefactor `d = 6` when `a==b==c`, `2` when `a==b` or `b==c`, else
`1`, applied to the distributed denominator

```text
d3[i,j,k] = d * (eps_o[i]+eps_o[j]+eps_o[k] - eps_v[a]-eps_v[b]-eps_v[c])
E_T = 2 * sum_{a>=b>=c} [36-term contraction]
```

The 36-term contraction is the literal table of `ccsd_t_slow.kernel`: six
rows (one per virtual permutation z of `r3(W + 0.5 V)/d3`) and six columns
per row, each pairing a virtual permutation w of the left factor with an
occupied permutation (`ijk`/`ikj`/`jik`/`jki`/`kij`/`kji`) of w. This is the
perturbative triples of Raghavachari et al., JCP 94, 442 (1991),
DOI:10.1063/1.460359 (the upstream code comments the same reference).

## Engines

* `triples_energy(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)` is
  the auditable triangular reference (transcription of `ccsd_t_slow.kernel`,
  including the 6/2 degeneracy and the literal 36-entry table).
* `triples_fullsum(...)` is an independent full-sum oracle: no triangular
  shortcut and no 6/2 pre-factor; it loops over all ordered (a,b,c) and all
  36 (z-perm, w-perm) pairs, deriving the occupied pairing from the S3
  relation `compose(inv(w_perm), z_perm)` (proved equal to SLOW_TABLE's 36
  entries in `tests/python/test_cc_triples.py`), then divides by 6. The
  reference and the oracle agree to ~1e-13 on random inputs; the test asserts
  `atol=1e-11, rtol=1e-10`.
* `build_triples_program(nocc, nvir)` / `triples_energy_tensorir(...)` lower
  the same inventory to unshared tiny-TensorIR (`einsum`/`transpose`/`gather`
  /`reduce_sum`/`divide`/`broadcast`/`add`). Inputs are declared general
  (symmetry-free) `restricted_spatial` parameters so dense tangent/cotangent
  spaces are legal under the current AD rules; the program passes a
  `dot_test` adjoint check and a one-entry finite-difference JVP check
  (issue #150 steps 3 and 11).

## Pinned upstream and provenance

`tools/vibeqc_cc/source_manifest.json` records PySCF 2.14.0 raw-file SHA-256:

| path | sha256 |
| --- | --- |
| pyscf/cc/ccsd_t_slow.py | 4fd3638e0176639781ecb138575a226912c99301a1923b60d20351eec36ac960 |
| pyscf/cc/ccsd_t.py | a498d415289f46350f301a9405b2ff1061870dd22619b7cb2200dd9bf9fc8f34 |

Both are Apache-2.0, copyright 2014-2020 The PySCF Developers (ccsd_t_slow
author Qiming Sun). `NOTICE` carries the (T) attribution alongside the
RCCSD attribution; `LICENSE.pyscf` is the shared license text.

## Ground truth

`tests/reference_data/cc/rccsd-t.json` (generated from committed endpoints) records, per
molecule, both this repository's `triples_energy` and pinned PySCF
`ccsd_t.kernel`, plus an inputs hash and the ground-truth value below:

| molecule | (o,v) | E_T (Hartree) |
| --- | --- | --- |
| h2 | (1,1) | 8.392021714075268e-49 (~0) |
| he | (1,1) | 0.0 |
| h2o | (5,2) | -6.731393342463869e-05 |
| nh3 | (5,3) | -1.122922812723691e-04 |
| ch4 | (5,4) | -1.555665872715297e-04 |

The nonzero-triples molecules are h2o/nh3/ch4. h2/he are the two-electron
sanity cases: for two electrons CCSD already equals FCI, so (T) must vanish
to machine rounding. The table is hard-coded in `tests/python/test_cc_triples.py`
as a PySCF-free regression.

## Verification

From the repository root (NumPy-only, local; Windows uses `;` as the list
separator, Linux `:`):

```bash
python -m pytest tests/python/test_cc_triples.py -q
```

On qz (pinned PySCF 2.14.0, Python 3.11 per docs/qz-development.md):

```bash
source .venv/bin/activate
export PYTHONPATH=.:python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m tools.generate_cc_triples_references \
    --output tests/reference_data/cc/rccsd-t.json
python -m tools.generate_cc_triples_references \
    --output /tmp/rccsd-t-second.json \
    --compare tests/reference_data/cc/rccsd-t.json
```

The generator verifies the installed upstream bytes against the manifest
hashes before producing the committed JSON and re-running with `--compare`
to assert two-generation stability. Ordinary tests check the committed JSON's
source identity, molecule hash and endpoint-input hashes without importing
PySCF. Each audited numerator term is also checked against explicit
source-index loops and the executed NumPy seed on unequal occupied/virtual
dimensions, so a wrong subscript cannot silently change the inventory hash.

## Bounded CUDA tiles (slice B)

`tools/vibeqc_cc/triples_tiles.py` decomposes the triangular `a>=b>=c`
virtual sum into a-chunked tiles (`TriplesTileEnumerator`); occupied space
is never chunked. `tile_triples_energy` / `tile_triples_energy_masked` are
the per-tile CPU reference and the masked-domain oracle;
`build_tile_triples_program` lowers one tile to the same unshared
tiny-TensorIR inventory as slice A. It uses distinct virtual spaces for
prefix-bounded label axes (`a,b,c`) and the full W1 summation axis `f`,
so partial-tile resident feeds have exact TensorSpec shapes without truncating
the canonical contraction.

`tools/vibeqc_cc/triples_cuda.py` `CudaTriplesTiles` evaluates each tile on
CUDA through the #420 resident TensorIR owner: one exact-shape plan and one
`PreparedResident` per tile (compile-cached to disk). Label axes are sliced
to `a_end`, while `ovvv` axis 2 and `t2` axis 3 retain full `nvir` for
the W1 sum; a full `nocc³ × nvir³` T3 or denominator tensor is never
allocated. The shared finite/canonical/denominator
guards run before any GPU work. CPU oracle comparison is opt-in
(`oracle=True`); the default path performs no CPU reference work.
`CudaTriplesResult` records per-tile scalars, per-tile plan peak bytes,
artifact keys, and the probed `runtime_device`.

GPU validation (numerical gates ≤1e-9 total / ≤1e-10 per tile, two-run
bitwise determinism, two-budget peak-memory evidence) runs on CUDA via
`tools/validate_cc_triples_tiles.py`. Retained manifests bind the run to the
exact git head and SHA-256 identities of the triples execution/validation
sources. The `--molecules` option selects a declared endpoint subset; every
requested shape/budget must pass, and an infeasible plan makes qualification
exit unsuccessfully while retaining its diagnostic record. Rationale for the
per-tile resident design and revisit conditions:
[`.agents/notes/implemented/performance/2026-09-18-bounded-cuda-triples-tiles.md`](../.agents/notes/implemented/performance/2026-09-18-bounded-cuda-triples-tiles.md).

```bash
python -m pytest tests/python/test_cc_triples_tiles.py -q
```

## Complete analytic-gradient validation boundary

`tools.vibeqc_cc/triples_complete_gradient.py` provides the internal conventional
RCCSD(T) complete-gradient endpoint for #155 B. It composes the existing pieces
in one state-bound chain:

```text
RHF -> RCCSD -> baseline Lambda -> corrected RCCSD(T) Lambda
    -> direct (T) + denominator + canonical-gauge response
    -> one total RHF Z solve -> AO/nuclear derivative contraction
```

The response mathematics is owned by `BoundCCSDTOrbitalResponse`; the final
`BoundCCSDTGradient` layer does not solve another Z-vector or introduce another
set of coupled-cluster derivative equations. On CPU, its generated Lambda,
parameter-response, triples VJP, raw-Hamiltonian, canonicalization and AO
back-transform programs execute through the common native TensorIR CPU backend
from #772 rather than the NumPy TensorIR interpreter. It reuses the already-
qualified RCCSD CPU dense derivative oracle and generated bounded CUDA one-
electron / weighted-ERI consumers for the final h/g/S cotangents. CPU and CUDA
final contractions therefore consume the same total response weights.

The current qualification is restricted to real closed-shell canonical RHF,
conventional unscreened all-electron Hamiltonians, no frozen core, no ECP or
auxiliary basis, and the existing <=12-AO small-system validation boundary.
`derivative_backend="cuda"` moves only the final AO/nuclear derivative consumers
to CUDA; it does not by itself qualify a fully resident GPU response chain.
Failure of any RHF, CCSD, corrected-Lambda, canonical-gauge, Z-vector, identity,
or derivative-consumer gate prevents a gradient result. No force/torque
projection is applied after assembly.

Pinned PySCF 2.14.0 analytic gradients for H2O and NH3 are independent acceptance
oracles in `tests/python/test_ccsd_t_complete_gradient.py`; complete-energy
finite differences and omission controls remain in
`tests/python/test_ccsd_t_gradient_validation.py`. The native/public CPU energy
owner and homogeneous prepared batch are covered by
`tests/python/test_rccsdt_public.py`; native/public force publication remains
#155 C even though the qualified CPU response/gradient TensorIR execution is now
native. The ownership rationale is recorded in
[the complete-gradient Agent Note](../.agents/notes/implemented/numerics/2026-09-21-ccsdt-complete-gradient-assembly.md).
