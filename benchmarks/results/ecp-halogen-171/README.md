# Real Br/I scalar-ECP qualification

This slice extends the existing real LANL2DZ fixture suite to Br and I, using
the unmodified orbital/ECP records from the pinned PySCF 2.14.0 test library
and STO-3G hydrogen. The initial measurements below use master `97adc1a`,
independently of the then-open PR #458. They are historical evidence; results
for the rebased implementation are recorded separately under `refresh/`.
Production native/runtime/compiler code is unchanged. There is no native LOC
or generated-science retirement, new method registration, or performance claim.

| Element | Atomic number | Removed core | Ionic charge | Molecular AOs |
| --- | ---: | ---: | ---: | ---: |
| Br | 35 | 28 | 7 | 9 |
| I | 53 | 46 | 7 | 9 |

Neutral singlets contain eight explicit electrons (RHF); their +1 doublet
cations contain seven (UHF). The s/p orbital records use scalar local and
s/p/d nonlocal terms. Parameter tables are loaded only by tests; they are
not redistributed in this bundle. See `docs/ecp.md` for parameter provenance
and the scalar-potential boundary.

## Gates and reproduction

The shared suite checks two off-axis bond geometries, separate local/nonlocal
Libcint matrices (`2e-9` absolute), 160/32/64 versus 224/44/88 quadrature,
all-center two-step derivative differences, nonsymmetric fixed-weight
contractions, effective-charge bookkeeping, complete energy (`2e-8` Eh) and
force (`2e-6` Eh/bohr) agreement, force translation balance, and budgeted
changed-geometry complete-energy differences. CPU/CUDA both retain the
original Rb/Cs regression cases. CUDA memcheck covers complete HF endpoints.

Build Release CPU/CUDA with AOT shells disabled; select architecture 89 for
the measured RTX 4090. Set `PYTHONPATH=python`, explicit `VIBEQC_LIBRARY`,
`VIBEQC_PROFILE=off`, and one OMP/OpenBLAS/MKL thread. Use the reference-test
extra pinned in `pyproject.toml`.

```sh
python -m pytest tests/python/test_ecp_heavy.py -q -k cpu
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --elements Br I --output cpu-endpoints.json
python tools/qualify_ecp_heavy.py --device cuda --elements Br I --output cuda-endpoints.json
```

The report retains parameter, fixture, driver and binary hashes; geometries;
matrix/refinement/backend errors; complete energy/force values and errors;
planned resource bounds and owned-device ledgers. Ledger peaks exclude driver,
library-internal, graph and pool allocations. Single-call times are context
for reproduction and do not establish performance leadership.

## Initial measured acceptance (2026-09-18)

Both fresh Release builds completed. CPU and CUDA each pass 20 tests (20
other-backend cases deselected), retaining the original Rb/Cs cases. Native
ECP CTest passes 2/2 on CPU and 3/3 on CUDA. Compute Sanitizer passes all eight
complete RHF/UHF cases across Rb/Cs/Br/I with zero errors. Local ECP IR/input
checks pass 40 tests; Ruff and formatting pass for both changed Python files.

| Maximum absolute error | CPU | CUDA |
| --- | ---: | ---: |
| Local ECP matrix | 4.864e-12 | 4.864e-12 |
| Nonlocal ECP matrix | 1.492e-13 | 1.368e-13 |
| Complete energy (Eh) | 6.928e-14 | 5.151e-14 |
| Complete force (Eh/bohr) | 1.208e-09 | 1.099e-09 |
| Net force (Eh/bohr) | 4.441e-16 | 2.557e-15 |

Maxima above cover the four newly qualified Br/I states per backend. All
errors meet the unchanged gates; no quadrature, tolerance or production
formula was changed to pass these cases.

| CUDA endpoint | Planned device bytes | Observed owned-device peak bytes |
| --- | ---: | ---: |
| Br RHF | 2,345,904 | 1,835,784 |
| Br UHF | 2,381,056 | 1,853,360 |
| I RHF | 2,346,416 | 1,835,848 |
| I UHF | 2,381,568 | 1,853,424 |

Every budgeted endpoint records zero rejected allocations. The complete
reports preserve each state's charge, multiplicity, parameter checksum,
reference/result energy and forces, refinement/backend differences and ledger.

## Initial provenance and retained evidence

`source-identity.json` binds base commit, 613 verified native/compiler/build
inputs, the two measured test/driver files, fresh library hashes and matching
CPU/CUDA generated ECP header hashes. All corresponding local LF-normalized
files were checked, and staged Git content is verified before committing.
The measured checkout has no production or compiler changes from the base.
`toolchain.json` records compiler/CUDA/GPU identity.

The external `evidence-171/ISSUE-171-halogen-results.tar.gz` archive contains
26 individually verified raw files (build, test and sanitizer logs, source
receipts, reports and reproducible run/collection scripts), totaling 86,557
compressed bytes. Its SHA256 is
`d41d254ee72fdf3ff1580048d331f7e9aadf31e085d6b43cb00b6debfe66efdc`.
`raw-evidence-manifest.json` retains each member's size/hash. Raw logs and
binaries remain outside Git. The allocated Notebook was stopped after
retrieval and verification.
