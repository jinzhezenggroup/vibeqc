# Issue #162 CPU UKS matched-grid endpoints

Date: 2026-09-13

The latest source is `9d43a84a2780c4df9fc63050ea513812b95e00b3`, which also
fixes extreme-spin exchange underflow, large-gradient correlation overflow
and stationary OH occupation cycles. Its clean-source endpoint rerun passes
all six cases and is recorded in
[`pr305-corrections-20260913.md`](pr305-corrections-20260913.md), including the
raw result path and checksum. The remote `d0b5862` run below is retained as
the preceding correction's reproducible evidence.

## Purpose

This records the independent endpoint gate for the CPU energy-only LDA/PBE UKS
slice. It is evidence for two small open-shell STO-3G systems on the exact
native `GridSpec v1` quadrature, not acceptance of prepared CUDA, batching,
gradients, density fitting, performance or quadrature convergence.

## Previous source and environment

- Branch: `codex/issue-0162-b`
- Clean source commit: `d0b586231a83350ad1fffb0591acdc24860d0043`
- Validator: `tools/validate_uks_endpoints.py`
- Validator SHA-256: `b0152b23c128a86515c85b1b22414183851ed4ecf9e267ba412a3445491648d6`
- Remote provider: qz CPU Notebook `general`, `CPU资源空间`
- Python `3.11.16`, NumPy `2.2.6`, PySCF `2.14.0`, Libxc `7.0.0`
- Linux `5.15.0-119`, glibc `2.35`; one OMP/OpenBLAS/MKL thread
- Grid: `GridSpec v1`, 49,152 explicit points for each H2 case
- Native backend: `cpu_reference`, conventional total-density J, energy only

The PySCF consumer independently builds AO integrals, Coulomb matrices, XC
energy/potentials, SCF iterations and final commutator residuals. VibeQC and
PySCF receive identical atoms, basis shells, charge, multiplicity, points and
weights.

## Gates and results

- Absolute total-energy error: `<= 1e-8 Eh`
- Native and independent physical residual RMS: `<= 1e-9`

| Case | Method | N alpha/beta | Native energy (Eh) | PySCF energy (Eh) | Absolute error (Eh) | Native residual | PySCF residual | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H2- doublet | LDA UKS | 2 / 1 | -0.44581926551511630 | -0.44581926551501294 | 1.034e-13 | 1.334e-11 | 1.074e-16 | pass |
| H2- doublet | PBE UKS | 2 / 1 | -0.49908288473813744 | -0.49908288473800233 | 1.351e-13 | 8.776e-12 | 1.383e-16 | pass |
| H2+ fully polarized | LDA UKS | 1 / 0 | -0.51584723067823930 | -0.51584723067821550 | 2.387e-14 | 2.776e-16 | 6.799e-17 | pass |
| H2+ fully polarized | PBE UKS | 1 / 0 | -0.54087410700709120 | -0.54087410594905530 | 1.058e-09 | 0.000e+00 | 6.799e-17 | pass |

The current implementation uses the documented
`semilocal-scaled-v1/pbe-spin-c2-1e-18` extension so the PBE energy and
potential remain continuous at an empty spin. It also returns the density that
was actually evaluated by the energy, residual and convergence gates. The
fully polarized PBE error passes the fixed gate but is the largest result and
must not be reported as machine-precision agreement.

## Evidence identity and recovery

- Persistent raw result:
  `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0162-b/20260913T1544Z-pr305-current-head/uks-endpoints.json`
- Raw result SHA-256:
  `a9a38e9527ddaee26b4ddda75a30484ad3d9fe53e802224e4350905be1067441`

The earlier `c02ac225` record and its
`20260913T1312Z-cpu-uks/uks-endpoints.json` raw result are retained only as
superseded lineage. They do not support acceptance of the current PBE spin
extension or returned-state semantics.

Reproduce in a clean Linux checkout with the pinned dependencies and built CPU
library:

```bash
export VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python tools/validate_uks_endpoints.py --output /tmp/uks-endpoints.json
```

The persistent JSON is the full record. This tracked file retains the gates,
critical values, source identity, checksum and recovery location without
duplicating the complete iteration histories into Git.
