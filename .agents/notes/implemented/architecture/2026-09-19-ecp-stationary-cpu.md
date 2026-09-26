# Decision: bind ECP to the shared stationary CPU gradient diagnostic

Status: implemented
Date: 2026-09-19

## Problem

DFT ECP energies use core-adjusted charges and residual local/projector
potentials. An all-electron stationary gradient cannot be applied to that
Hamiltonian. AO arrays alone do not identify ECP parameters, and occupation
counts alone cannot distinguish two different ECPs with the same core count.

## Decision

Extend the existing mean-field envelope with scalar-semilocal ECP semantics.
The compiler declares effective-charge attraction/nuclear primitives and two
additional local/projector sources with both AO- and ECP-center derivatives.
The existing TensorIR AD and contraction/reduction rules supply their weights.

CPU ECP snapshot version 4 appends exact native core counts and ECP terms.
All-electron CPU version 2 and CUDA version 3 remain compatible. The new private
CPU derivative read validates the current token before reading the actual owner,
uses the existing checked independent CPU ECP provider, and publishes only after
success. It accepts no caller-supplied substitute Hamiltonian. The snapshot's
mathematical model identity includes the native ECP parameters.

## Rejected alternatives

- Inferring an all-electron Hamiltonian from AO shape or method name: misses the
  effective attraction, nuclear repulsion and residual projector response.
- Adding a complete HF force: double-counts common terms and adds HF exchange.
- Inventing a second ECP derivative formula: the existing independent CPU
  provider already supplies the complete derivative semantics.
- Promoting CUDA/public forces from this diagnostic: they require distinct
  backend, complete-endpoint, resource and failure qualification.

## Invariants and evidence

Density weights are occupation-weighted, summing alpha and beta once. The common
plan rejects missing local/nonlocal sources. Nuclear gradients use effective
ionic charges and force publication is not performed here. Tests use independent
PySCF analytic full-grid-response gradients, reconverged three-step energy
differences, nonzero ECP omission checks, translation and stale-owner rejection.

## Consequences and revisit

The retained CPU oracle is an explicit diagnostic provider, not retirement of
that formula family or generated CPU ECP production. It materializes two dense
`3*natom*nao**2` derivative arrays before bounded TensorIR contraction. Revisit
when qualifying generated CPU/native CUDA ECP derivative consumers and public
resource-bounded force endpoints. Refs #171 and #163; neither umbrella is closed.
