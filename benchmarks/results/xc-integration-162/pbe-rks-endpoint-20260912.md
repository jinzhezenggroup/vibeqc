# Issue #162 PBE RKS endpoint

Date: 2026-09-12

## Purpose

This is an independent, matched-grid endpoint for the limited PBE RKS slice
implemented by Issue #162. It is evidence for the closed-shell CPU energy path,
not acceptance of complete DFT support.

## Identities

- Local worktree: `codex/issue-162-a`
- Local source commit: `7530b9c185793f1cac0711642a39bf7151091dd3`
- Local source state: dirty worktree containing the uncommitted Issue #162 changes
- Remote source commit used by the endpoint: `1e93c3cfa10afdf99ee26c70312bcfe533114331`
- Remote provider: qz CPU Notebook `general`
- Workspace: `CPU资源空间`
- Remote project path: `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-issue-162`

The remote endpoint used the remote checkout's `tools/vibeqc_dft` layout. The
remote source identity is recorded separately because that checkout predates
the current worktree layout; the numerical comparison is between the
independent PySCF/Libxc consumer and the local PBE SCF result below.

## Environment

- Python `3.11.16`
- PySCF `2.14.0`
- Libxc `7.0.0`
- NumPy `2.2.6`
- GridSpec v1 materialized points: `49,152`
- Method: closed-shell RKS PBE
- Coulomb term: conventional J
- Density and grid: matched between the independent endpoint and local case

## Results

| Quantity | Independent PySCF/Libxc | Local PBE RKS | Absolute difference |
| --- | ---: | ---: | ---: |
| Total energy (Eh) | `-1.15206437533967110` | `-1.15206437533967154` | `4.44089209850062616e-16` |

- Independent endpoint status: `converged=True`
- Endpoint gate: `1e-8 Eh`
- Result: **pass**

## Scope and limits

This endpoint supports the claim that the tested H2 matched-grid closed-shell
PBE energy agrees with the independent PySCF/Libxc consumer. It does not cover
UKS or spin-polarized semantics, gradients, batching, quadrature convergence,
GPU execution, other molecules or basis sets, or the finite-grid fallback tail.
The native DFT and DFT API regression tests remain required alongside this
independent endpoint.
