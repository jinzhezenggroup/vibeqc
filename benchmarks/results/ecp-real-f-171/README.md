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
The full CPU suite takes 997.69 seconds and CUDA takes 66.72 seconds on the
recorded host; the 23-AO CPU replay is intentionally included in this gate.
Native ECP CTest passes 2/2 CPU and 3/3 CUDA. Both complete Au CUDA endpoints
pass Compute Sanitizer with zero errors. Local ECP IR/input tests pass 40
checks, and Ruff/format checks pass for both changed Python files.

| Maximum absolute Au endpoint error | CPU | CUDA |
| --- | ---: | ---: |
| Local matrix | 8.616e-14 | 8.038e-14 |
| Nonlocal matrix | 1.359e-13 | 1.235e-13 |
| Complete energy (Eh) | 6.537e-13 | 7.390e-13 |
| Complete force (Eh/bohr) | 1.042e-10 | 1.970e-10 |
| Net force (Eh/bohr) | 8.882e-16 | 2.220e-15 |

The compact reports retain state-specific values, all three reference initial
guess energies, parameter/library/fixture/driver identities and resource status.
`unqualified-cation.json` retains the cation/anion state-selection diagnostic.

`raw-evidence-manifest.json` binds the 28 fresh qualification logs, timing/exit
records, compact endpoint reports, Release libraries and generated ECP headers
by byte size and SHA256. Raw logs and binaries remain outside Git on the
qualification host. `source-identity.json` independently binds all 680 tracked
native/runtime/build inputs plus the final test and report-driver sources.

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

Build Release CPU/CUDA from qualification base `0ed06a3` with AOT off and the
target architecture selected (120 on the measured RTX 5090). Set `PYTHONPATH=python`,
explicit `VIBEQC_LIBRARY`, `VIBEQC_PROFILE=off`, and one OMP/OpenBLAS/MKL thread.
Install the pinned `reference-test` extra.

```sh
python -m pytest tests/python/test_ecp_heavy.py -q -k cpu
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --elements Au --output cpu-endpoints.json
python tools/qualify_ecp_heavy.py --device cuda --elements Au --output cuda-endpoints.json
VIBEQC_ECP_CUDA_TEST=1 compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_heavy.py -q -k 'cuda and Au and complete_hf'
```

Production native, runtime and compiler code is unchanged by this PR. Fresh
CPU/CUDA Release binaries were built from the rebased qualification tree; no
prior binary is reused. `source-identity.json` binds all 680 native/runtime/build
inputs, both final libraries, both generated ECP headers and the final
test/report sources. During finalization master advanced from `0ed06a3` to
`e215b30` via #448; that commit changes no file in the 680-file qualification
identity, so the final merge candidate remains byte-identical for every
qualified build/runtime input. The exact comparison is retained in the source
identity.

Native scientific LOC delta is zero. Single-call timings are reproducibility
context, not a performance claim. Spin-orbit methods, other ECP families,
DF/DFT forces, g orbitals/projectors and larger CUDA budget inventories remain
outside this qualification.
