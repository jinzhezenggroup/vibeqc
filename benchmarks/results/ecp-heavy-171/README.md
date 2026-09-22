# LANL2DZ Rb/Cs scalar ECP qualification

This slice qualifies two real heavy-element parameter records on the existing
scalar ECP implementation. It starts at master
`e16f526c162512432657f5a320588716f269bbda` (after #442) and does not depend on
the nonlocal-f extension in #446. All production/compiler/native source files
are unchanged. Refs #171.

## Scope and independent reference

Unmodified PySCF 2.14.0 LANL2DZ orbital/ECP data for Rb and Cs, paired with STO-3G
H, give 13 public AOs. Rb removes 28 electrons and Cs removes 46; both have
effective ionic charge 9. Neutral singlet hydrides have 10 explicit electrons
and use RHF; singly charged doublet cations have 9 and use UHF. The orbital
records use s/p shells, local ECP label f and nonlocal s/p/d projectors.

The canonical combined orbital/ECP parameter fingerprints are:

| Element | SHA-256 |
|---|---|
| Rb | `9d8f07743d6859efb8fa6155fe1ac83e7e9de18bb453f51dcdb2846c202277aa` |
| Cs | `f5d99d7ab2ca5fae6d454127584aa3bc241de2ea8e3d4870d5889f8c23b0aa9e` |

The test loads these parameters from the reference installation and rejects a
changed fingerprint. Numerical tables are not redistributed. PySCF is used
only by tests and the evidence driver; production receives owned scalar input
records and never imports the reference package. Consult the installed
LANL2DZ basis-file notices and original references before redistributing data.

The heavy atom is at `(0.13,-0.21,0.17)` bohr, H at `(0.43,0.19,4.4)` for Rb
or `(0.43,0.19,4.8)` for Cs. Raw gates additionally move H by +0.37 bohr along z.
Complete-energy differences use both atoms' directional displacements at
steps `2e-4` and `7e-5` bohr. The exact geometries, charge/spin, identities and
errors are retained in the endpoint reports and acceptance test.

Libcint provides separately selected local/nonlocal matrices, all-center
finite differences and arbitrary nonsymmetric AO-weight references. PySCF
provides complete RHF/UHF energies and forces. The existing independent CPU
implementation is also compared directly with CUDA, without sharing generated
projector arithmetic.

## Acceptance gates

- Local/nonlocal raw matrices separately: absolute `2e-9` Eh.
- Existing coarse/refined matrix/derivative differences: `2e-9` / `2e-8`.
- Raw derivative finite differences: `3e-7` absolute, `2e-6` relative;
  arbitrary fixed-weight contractions: `2e-6` absolute.
- Complete RHF/UHF energies/forces: `2e-8` Eh / `2e-6` Eh/bohr.
- Complete-energy directional differences: `2e-6` Eh/bohr at both steps.
- Budgeted geometry replay and restoration pass for both methods and elements;
  CUDA allocations must stay within the declared limit with zero rejections.

These gates retain the existing production grid policy and error boundaries.
They do not claim arbitrary Rb/Cs molecular environments, other LANL2DZ
elements, other families, spin-orbit, ECP DFT/DF/MP2, or higher angular limits.

## Measured results

- CPU native CTest: **31/31**; CUDA ECP native CTest: **3/3**.
- Final heavy-element suite: **10 CPU passed** (10 expected GPU skips) and
  **10 CUDA passed** (10 deselected).
- Existing CPU ECP/IR/input regression: **53 passed**, 10 expected GPU skips.
- Compute Sanitizer on both heavy-element raw matrix/derivative cases:
  **2 passed, 0 errors**.
- Local IR/input checks: **30 passed**; compiler/SCF dependency and unchanged
  183-file CUDA ownership checks pass.

Across the four CPU and four CUDA complete endpoints, maximum local/nonlocal
matrix errors are **2.62e-14 / 9.46e-13 Eh**, energy error **3.45e-13 Eh**,
and force error **2.32e-10 Eh/bohr**. CPU/CUDA raw blocks agree within
**1.89e-15 Eh** for matrices and **6.94e-17 Eh/bohr** for derivatives.
The exact values are retained in `cpu-endpoints.json` and `cuda-endpoints.json`.

| CUDA case | Planned host bytes | Planned device bytes | Observed ledger peak bytes |
|---|---:|---:|---:|
| RbH RHF | 4,611,424 | 3,257,056 | 2,330,688 |
| RbH+ UHF | 4,697,952 | 3,330,288 | 2,367,304 |
| CsH RHF | 4,611,168 | 3,256,800 | 2,330,656 |
| CsH+ UHF | 4,697,696 | 3,330,032 | 2,367,272 |

All four recorded CUDA ledgers have zero rejected allocations. The ledger
covers owned buffer capacities, excluding driver/graph/pool and library-internal
allocations; planned bounds are not observed whole-process memory peaks.
Budgeted displaced-geometry and restoration assertions also pass in the tests.

## Reproduction and provenance

Build Release CPU/CUDA libraries as described in [the ECP contract](../../../docs/user/ecp.md).
The measured GPU is an RTX 4090 with CUDA 12.9, architecture 89, and AOT shells
disabled; Compute Sanitizer is 12.8. Use the pinned reference-test extra,
`PYTHONPATH=python`, the exact `VIBEQC_LIBRARY`, and one OpenMP/OpenBLAS/MKL
thread. The test/driver are self-contained additions to the public base.

`source-identity.json` verifies all 3,880 regular baseline archive files are
unchanged and records the two added qualification source hashes. Those hashes
match the local Git blobs and both endpoint reports; each report also identifies
its exact shared library. `raw-evidence-manifest.json` verifies 35 raw evidence
files, including build/test logs, full baseline manifest and execution scripts.
The archive additionally retains the exact test and driver copies (37 members).
The 235,779-byte archive is retained outside Git at the original qb-ilm
workspace's `evidence-171/heavy-results.tar.gz` and downloaded locally.
The public base plus the two committed test/driver files suffice to reproduce
the qualification; the remote archive is not a test dependency.

```sh
python -m pytest tests/python/test_ecp_heavy.py -q
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --output cpu-endpoints.json
python tools/qualify_ecp_heavy.py --device cuda --output cuda-endpoints.json
```

The initial CPU test draft incorrectly expected the public backend label
`cpu` instead of the API's `cpu_reference`. Its four assertion failures are
retained as diagnostics and excluded from acceptance. The assertion was
corrected and the final suite rerun; production code and tolerances did not
change. The final suite also includes UHF budgeted energy differences.

There is no production arithmetic or schedule change: generated capability
delta none, handwritten scientific CUDA LOC +0/-0, runtime CUDA LOC +0/-0,
no production path retirement, and the independent CPU oracle/fallback
remains. Resource observations establish feasibility for these 13-AO cases,
not a larger inventory. Single-call endpoint timings include synchronous
Calculator setup, SCF and forces after raw/reference setup; they are not cold
process-startup measurements or comparative speed evidence.
