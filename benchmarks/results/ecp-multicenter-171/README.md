# Mixed-family two-center scalar ECP qualification

This slice checks simultaneous, distinct physical ECP centers on the existing
direct HF path. It starts at master `b78215ef0bf2cb43c91234c9c2dc852151605468`
and is independent of the pending projector-f, Stuttgart and DFT-energy PRs.
Production mathematics, compiler lowering, native schedules and resource
formulas are unchanged. Refs #171.

## Scope and reference

Na uses the installed PySCF 2.14.0 LANL2DZ orbital/ECP record; K uses Stuttgart
RLC (`stuttgart-dz`). Both atoms carry orbitals and distinct scalar potentials.
Na removes 10 electrons and K removes 18; each has effective ionic charge 1.
The combined spherical basis has 16 AOs. Neutral NaK uses RHF with two explicit
electrons; NaK+ uses doublet UHF with one. These are bounded numerical fixtures,
not a claim about experimental accuracy or all mixed-family molecules.

The canonical combined orbital/ECP fingerprint is
`e7860773555887760528a11fcbc37958a2923df0fd5d4e6f2874e17b5acf29c0`.
Tests reject a changed fingerprint. Parameters are read from the independent
reference installation and are not redistributed or consulted by production.
Na starts at `(0.13,-0.21,0.17)` bohr and K at `(0.43,0.29,5.8)` bohr.
Raw tests also displace K by +0.37 bohr along z.

Libcint supplies local/nonlocal blocks independently. Operator-center selection
retains all orbital centers and verifies that both distinct nonlocal potentials
contribute, while only Na has a nonzero local residual. All six physical nuclear
coordinates are differentiated independently at steps `2e-4` and `7e-5` bohr.
Random nonsymmetric AO weights test the full contraction convention.

Reversing atom order also reverses ECP core mapping, AO blocks and derivative
atom axes. Explicit permutation checks cover matrices, derivatives, complete
energies and forces. The PySCF complete-HF oracle converges from both `1e` and
`minao` guesses before comparison. Budgeted prepared executions move both atoms,
check complete-energy directional differences at both step sizes and restore
the original geometry.

## Acceptance gates

- Separate raw local/nonlocal matrices: absolute `2e-9` Eh.
- Coarse/refined matrices and derivatives: `2e-9` and `2e-8` respectively,
  using the existing 160/32/64 and 224/44/88 grids.
- Independent raw derivative differences: `3e-7` absolute, `2e-6` relative;
  arbitrary-weight contraction: `2e-6` absolute.
- Atom/AO permutation for raw values and derivatives: `2e-10` absolute.
- Independent complete energy/force: `2e-8` Eh / `2e-6` Eh/bohr;
  complete-energy directional differences: `2e-6` Eh/bohr at both steps.
- Permuted complete energy/force: `2e-10` Eh / `2e-8` Eh/bohr.
- CUDA allocation ledger peak must fit the declared budget with no rejections.

These gates do not extend angular momentum, radial powers, method capabilities,
or the 16-AO budget inventory. No general speedup or full #171 completion is
claimed. Handwritten/compiler scientific and runtime ownership deltas are zero;
the independent native CPU oracle/fallback remains intact.

## Measured results and provenance

CPU and CUDA each pass all seven new numerical tests. Native CTest passes
31/31 on CPU and 3/3 targeted ECP tests on CUDA. Input/basis/IR preflight passes
75 tests; existing CPU ECP regression passes 33 with 20 expected CUDA skips.
Compute Sanitizer passes the two-center raw matrix/derivative case with zero
errors. Local structure/input/ownership tests pass 161 cases.

Across eight reported complete endpoints (two atom orders, two methods and
two backends), maximum energy error is `3.886e-15` Eh and maximum force error
is `1.166e-11` Eh/bohr. Separate local/nonlocal matrix errors are `8.813e-16`
and `5.524e-15` Eh. CPU/CUDA raw values and derivatives differ by at most
`3.331e-16`. Exact values are retained in the two endpoint reports.

| CUDA method (both atom orders) | Planned host bytes | Planned device bytes | Observed owned-buffer peak |
|---|---:|---:|---:|
| RHF | 6,119,392 | 4,281,136 | 3,038,968 |
| UHF | 6,250,464 | 4,392,000 | 3,094,400 |

All four CUDA ledger records have zero rejected allocations.

The measured runtime is RTX 4090, CUDA 12.9, architecture 89, AOT shells off,
with Compute Sanitizer 12.8. Existing Release libraries from base `072f6de`
were reused after checking all 604 relevant native/compiler/build source
inputs and both previously recorded library hashes. This was not a rebuild.
The candidate source snapshot has 3,893 unchanged baseline files; both added
test/reporter files match the measured hashes. The reused test helper is
identified separately in each endpoint report.

`source-identity.json` contains the compact source/library receipt;
`raw-evidence-manifest.json` identifies the original archive and 33 raw records.
The raw archive is retained outside Git at the existing qb-ilm workspace's
`evidence-171/ISSUE-171-multicenter-results.tar.gz`; it is not a test dependency.
The public base plus committed tests/reporter reproduce the qualification.

Two launch-script diagnostics are retained outside Git: an initial CPU
preflight named a nonexistent test file, and editing that shared script while
CUDA was running interrupted it after the successful test/sanitizer stages.
CPU validation was rerun with the corrected path; the CUDA final report ran
through a separate immutable script. Final source, numerical gates and error
tolerances were unchanged. These script failures are excluded from acceptance.

## Reproduction

Use Release CPU/CUDA libraries and the pinned reference extra described in
[the ECP contract](../../../docs/user/ecp.md). Set `PYTHONPATH=python`, the explicit
`VIBEQC_LIBRARY`, and one OpenMP/OpenBLAS/MKL thread. The reporter imports the
committed multicenter fixture and the existing heavy-element reference helper.

```sh
python -m pytest tests/python/test_ecp_multicenter.py -q -k cpu
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_multicenter.py -q -k cuda
python tools/qualify_ecp_multicenter.py --device cpu --output cpu-endpoints.json
python tools/qualify_ecp_multicenter.py --device cuda --output cuda-endpoints.json
```

Reported singlepoint times cover synchronous setup, SCF and complete forces
after raw/reference setup, with no warmup or comparative performance claim.
Planned memory and owned-buffer allocation-ledger peaks are distinct from
whole-process GPU memory, driver and library-internal allocations.
