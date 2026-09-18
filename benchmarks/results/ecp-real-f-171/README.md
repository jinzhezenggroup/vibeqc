# Real LANL2DZ Au f-projector qualification

This bundle qualifies the unmodified PySCF 2.14.0 LANL2DZ Au orbital/ECP
records with STO-3G hydrogen. It reuses the heavy-element suite and retains
Rb/Cs regression coverage. The pinned parameter checksum is
`618e1d76ee6ec1af1f846befb9ca9da96b54c8bbac109033cf5b34414b615642`.
Numerical parameter tables are test-only external data and are not redistributed.

Au has nuclear identity 79, removes 60 core electrons and has effective ionic
charge +19. Its real nonlocal f difference accompanies s/p/d channels and a
local g label. The s/p/d orbital records give 23 spherical molecular AOs.
The accepted states are neutral singlet AuH (20 explicit electrons, RHF) and
the -1 doublet (21 electrons, UHF), at the off-axis geometry in the fixture.

## Gates

The shared tests compare separate local/nonlocal matrices against Libcint at
two geometries, both 160/32/64 and 224/44/88 grids, value-only exports,
all-center derivatives with two finite-difference steps and nonsymmetric
fixed-weight contractions. The additional Au test independently isolates
Libcint's nonzero f block and compares it and its all-center derivatives
against full-minus-zero-f native results. The local matrix must not change.

Complete energy and force gates remain `2e-8` Eh and `2e-6` Eh/bohr. Prepared
replay checks two-step complete-energy directional differences and restores
the original geometry. Au SCF stopping criteria are `1e-12` in energy and
`1e-10` in density, at most 200 iterations. Independent reference calculations
must converge to the same energy from minao, one-electron and atomic guesses.
No integral grid, scientific implementation or acceptance tolerance changed.

## Measured acceptance

CPU and CUDA each pass all 16 tests, including the original Rb/Cs regressions.
The full CPU suite takes 1,041.53 seconds and CUDA takes 78.96 seconds on the
recorded host; the 23-AO CPU replay is intentionally included in this gate.
Native ECP CTest passes 2/2 CPU and 3/3 CUDA. Both complete Au CUDA endpoints
pass Compute Sanitizer with zero errors. Local ECP IR/input tests pass 40
checks, and Ruff/format checks pass for both changed Python files.

| Maximum absolute Au endpoint error | CPU | CUDA |
| --- | ---: | ---: |
| Local matrix | 8.616e-14 | 8.039e-14 |
| Nonlocal matrix | 1.342e-13 | 1.217e-13 |
| Complete energy (Eh) | 6.537e-13 | 7.106e-13 |
| Complete force (Eh/bohr) | 1.042e-10 | 1.971e-10 |
| Net force (Eh/bohr) | 8.882e-16 | 8.882e-16 |

The compact reports retain state-specific values, all three reference initial
guess energies, parameter/library/fixture/driver identities and resource status.
`unqualified-cation.json` retains the cation/anion state-selection diagnostic.

The raw archive `evidence-171/ISSUE-171-real-f-results.tar.gz` contains 36
files and is 147,074 compressed bytes. SHA256:
`c37d0a18cfada325d7350e711d568d070802ee927dc04508a921fb53a347ed28`.
The downloaded archive, member manifest and all 649 local input hashes were
verified. Raw logs and binaries remain outside Git. The Notebook was stopped
and its STOPPED state verified after retrieval.

## Resource boundary

The existing CUDA budget inventory supports at most 16 public AOs. AuH's
23-AO CUDA numerical endpoints and replay therefore execute without a budget.
Resource estimation must report unsupported, and explicit-budget preparation
must raise before execution. CUDA reports retain that diagnostic, with null
planned bounds and no claimed ledger. CPU Au and CUDA Rb/Cs replay retain
exact-budget checks. This work does not extend a resource inventory.

## Excluded cation and retained diagnosis

AuH+ is not qualified. Its first default native UHF calculation differed from
the three mutually consistent independent references by about 0.06968 Eh.
Tighter native stopping criteria failed to converge within 200 iterations.
The original failed CUDA tests, source identity and state diagnostic are
retained in the external raw archive. The superseded CPU replay was stopped
after the cation was excluded; it is not counted as a passing run.

The accepted anion is a distinct declared fixture: all three independent
initial guesses and the native solution agree. No cation result is relabeled
as passing, and these tests do not prove a global HF minimum or arbitrary
gold chemistry. See the [decision note](../../../.agents/notes/implemented/numerics/2026-09-18-ecp-real-f-gold.md)
for the state-selection and resource rationale.

## Reproduction and source identity

Build Release CPU/CUDA from base master `97adc1a` with AOT off and the target
architecture selected (89 on the measured RTX 4090). Set `PYTHONPATH=python`,
explicit `VIBEQC_LIBRARY`, `VIBEQC_PROFILE=off`, and one OMP/OpenBLAS/MKL thread.
Install the pinned `reference-test` extra.

```sh
python -m pytest tests/python/test_ecp_heavy.py -q -k cpu
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --elements Au --output cpu-endpoints.json
python tools/qualify_ecp_heavy.py --device cuda --elements Au --output cuda-endpoints.json
VIBEQC_ECP_CUDA_TEST=1 compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_heavy.py -q -k 'cuda and Au and complete_hf'
```

Production native, runtime and compiler code is unchanged. The measured
CPU/CUDA Release binaries from `build-171/halogen-20260918` are reused after
647 native/runtime/build inputs, both binaries and generated ECP header hashes
match. `source-identity.json` binds those inputs, the original build receipt
and final test/report source hashes. A report-only follow-up handles absent
resource diagnostics for unbudgeted execution; numerical test source is
unchanged between its successful run and that report correction.
The original build logs remain in
`evidence-171/ISSUE-171-halogen-results.tar.gz` (SHA256
`d41d254ee72fdf3ff1580048d331f7e9aadf31e085d6b43cb00b6debfe66efdc`).

Native scientific LOC delta is zero. Single-call timings are reproducibility
context, not a performance claim. Spin-orbit methods, other ECP families,
DF/DFT forces, g orbitals/projectors and larger CUDA budget inventories remain
outside this qualification.
