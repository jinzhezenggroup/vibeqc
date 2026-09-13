# Issue #162 CPU UKS matched-grid endpoints

Date: 2026-09-13

## Purpose

This records the independent endpoint gate for the CPU energy-only LDA/PBE UKS
slice. It is evidence for two small open-shell STO-3G systems on the exact
native `GridSpec v1` quadrature, not acceptance of prepared CUDA, batching,
gradients, density fitting, performance or quadrature convergence.

## Source and environment

- Branch: `codex/issue-0162-b`
- Clean source commit: `c02ac225ab0161810c1836c9a0257c10885f4e02`
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
| H2- doublet | LDA UKS | 2 / 1 | -0.44581926551511586 | -0.44581926551501294 | 1.029e-13 | 4.459e-13 | 1.074e-16 | pass |
| H2- doublet | PBE UKS | 2 / 1 | -0.49908288473811100 | -0.49908288473800233 | 1.087e-13 | 1.630e-13 | 1.383e-16 | pass |
| H2+ fully polarized | LDA UKS | 1 / 0 | -0.51584723067823890 | -0.51584723067821550 | 2.343e-14 | 9.615e-17 | 6.799e-17 | pass |
| H2+ fully polarized | PBE UKS | 1 / 0 | -0.54087410700709100 | -0.54087410594905530 | 1.058e-09 | 8.777e-17 | 6.799e-17 | pass |

The complete-polarization PBE case initially failed by `2.503e-2 Eh` because
the production tail replaced PBE with LDA whenever one spin density was zero.
The accepted commit keeps exact active-spin PBE energy and derivatives at this
boundary and uses the documented bounded inactive-spin potential. The final
error passes the fixed gate but is the largest result and must not be reported
as machine-precision agreement.

## Evidence identity and recovery

- Persistent raw result:
  `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0162-b/20260913T1312Z-cpu-uks/uks-endpoints.json`
- Raw result SHA-256:
  `ec68ea059414dea836ede98fbcfa4b57af1983435f4ddf2e4510e1a06946313a`

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
