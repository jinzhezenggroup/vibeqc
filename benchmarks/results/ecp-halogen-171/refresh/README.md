# Br/I qualification after integration with current master

Measured 2026-09-19 at commit `66f083a146b3f49291cc3d2625cb83500e61e9a7` on base
`45af86a3be530aaef4fa77630099ac8cd0c16241`. This incorporates merged Au qualification and
bounded radial batching. Final publication adds documentation and evidence
only; all 1094 recorded source inputs match the measured commit.
Production/native/compiler sources are unchanged from the base. Existing
fixture/test function bodies are unchanged; the case table adds Br/I.

## Scope and acceptance

Pinned PySCF 2.14.0 LANL2DZ Br/I with STO-3G H, neutral singlet RHF and +1
doublet UHF, nine spherical s/p AOs, scalar local plus s/p/d projectors.
The shared suite retains Rb/Cs/Au, including Au's negative-charge doublet,
independent real-f channel and unsupported CUDA budget-domain checks.

- CPU: 26 passed, 26 deselected, 2 warnings in 883.40s (0:14:43).
- CUDA: 26 passed, 26 deselected, 2 warnings in 91.94s (0:01:31).
- Native ECP CTest: CPU 2/2; CUDA 3/3.
- Compute Sanitizer: 10 passed, 42 deselected, 2 warnings in 287.16s (0:04:47); zero errors.
- The two warnings in each run are PySCF `remove_linear_dep_` deprecations.
- Local input/IR tests: 40 passed; Ruff check/format, compiler and SCF dependency
  checks, CUDA ownership inventory and evidence retention checks passed.

All original gates remain: separate Libcint operator components, all-center
two-step finite differences, fixed nonsymmetric weights, complete HF forces,
planned-budget displaced-energy/recovery replay and CUDA allocation limits.
No tolerance or quadrature was changed. Maximum errors below cover only the
four Br/I states in the retained endpoint reports, not arbitrary chemistry.

| Maximum absolute error | CPU | CUDA |
| --- | ---: | ---: |
| Local ECP matrix | 4.864e-12 | 4.864e-12 |
| Nonlocal ECP matrix | 1.492e-13 | 1.368e-13 |
| Complete energy (Eh) | 6.928e-14 | 5.151e-14 |
| Complete force (Eh/bohr) | 1.208e-09 | 1.099e-09 |
| Net force (Eh/bohr) | 4.441e-16 | 2.175e-15 |

| CUDA endpoint | Device budget bytes | Observed owned-device peak bytes |
| --- | ---: | ---: |
| Br RHF | 5,705,136 | 5,195,016 |
| Br UHF | 5,740,288 | 5,212,592 |
| I RHF | 5,705,648 | 5,195,080 |
| I UHF | 5,740,800 | 5,212,656 |

All four CUDA reports record zero rejected allocations. Device-ledger peaks
exclude library-internal, driver, graph and pool allocations. CPU replay also
uses its planned budget. Timing samples are reproduction context, not speedups.

## Reproduction

Build the measured commit in Release with AOT shells disabled, CUDA 12.9 and
architecture 89 on the recorded RTX 4090. Use the pinned reference-test extra,
`VIBEQC_PROFILE=off`, one OMP/OpenBLAS/MKL thread, `PYTHONPATH=python`, and set
`VIBEQC_LIBRARY` to each corresponding freshly built `libvibeqc.so`.

```sh
python -m pytest tests/python/test_ecp_heavy.py -q -k cpu
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --elements Br I --output cpu-endpoints.json
python tools/qualify_ecp_heavy.py --device cuda --elements Br I --output cuda-endpoints.json
VIBEQC_ECP_CUDA_TEST=1 compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_heavy.py -q -k 'cuda and complete_hf'
```

## Provenance

`source-identity.json` records the exact source/base commits, 1094 input
hashes, fresh CPU/CUDA library hashes and equal generated ECP header hashes.
The source was exported directly from the fetched Git commit, without overlays.
Tracked symlink hashes refer to target content; the recorded mapping explains
the tools shell-class manifest link. Git object checks resolve that target
instead of comparing Windows link-placeholder text to Linux file contents.
Endpoint reports bind the fixture, driver and library hashes to every result.

Raw archive: `ISSUE-171-halogen-refresh-results.tar.gz`, 115,927 bytes.
SHA256: `9b1caec69fa59930e948a8ba4c5f5b5183e54773992435f906e5170787f3fa88`.
All 26 manifest members and the downloaded archive were verified.
Build/test/sanitizer logs and reproduction/collection scripts remain in the
external workspace evidence archive; binaries and raw logs are not added to Git.
The parent directory retains the separately labeled historical 2026-09-18 run.

This qualifies the declared Br/I states and geometries only. It does not enable
DFT forces, DF/ECP, spin-orbit or general heavy-element chemistry, repair AuH+
SCF state selection, or extend the 16-AO CUDA HF budget inventory. Refs #171.
