# Decision: publish VV10/rVV10 through the existing bounded CPU stationary-force endpoint

Status: implemented
Date: 2026-09-22

## Context

PR #903 merged the self-consistent CPU VV10/rVV10 contribution through the
generic KS execution plan. The analytic nonlocal geometry implementation was
already qualified independently and the stationary-gradient plan already owned
three source names: nonlocal_ao, nonlocal_grid, and nonlocal_weight. The
remaining primitive-level force gap was therefore composition, not a new force
derivation.

The separate draft PR #935 owns final WB97M-V/RSH/B97M composition. This change
does not modify its range-separated exchange consumer, named-method admission,
or final WB97M-V capability.

## Decision

Populate the existing stationary sources from the live successful native
snapshot:

- nonlocal_ao receives the audited AO-center feature pullback;
- nonlocal_grid scatters exact-grid point-motion partials by the snapshot's
  native grid owners;
- nonlocal_weight contracts dE/dw against the same Becke partition adjoint used
  by the semilocal stationary path.

The native bounded pair provider is reused with the exact MethodIR variant,
parameters, coefficient, grid, density, source library and configured nonlocal
memory budget. No alternate VV10 formula or method-name branch is introduced.

The public capability is promoted only on the already-qualified CPU stationary
force domain. CUDA nonlocal self-consistent forces remain fail-closed.

## Bounded-work contract

Admission now charges the complete Ngrid^2 nonlocal pair traversal plus the
extra partition-response traversal before derivative compilation/provider
execution. The additional host inventory includes a conservative nonlocal force
block for the native pair plan and rho/gradient/feature/geometry arrays. The
existing global CPU-force host cap remains authoritative.

## Evidence

On the qz validation host, a dependency-free synthetic Na-H scalar-ECP fixture
exercised the public Calculator energy-plus-force endpoint. VV10 and rVV10,
through both RKS and UKS owners, matched independently recomputed central energy
differences at steps 3e-4 and 1e-4 bohr. The finest-step directional errors were
8.7e-13 to 1.2e-11 Eh/bohr and translational force sums were about 1e-16.

The three nonlocal stationary sources are all materially nonzero on the same
fixture. A representative VV10-RKS run observed approximately 1.58e-3,
1.50e-3, and 8.27e-4 Eh/bohr maximum absolute contributions for AO, grid, and
weight sources respectively. The work ledger records the exact Ngrid^2 pair
count.

## Remaining boundary

This closes the public CPU analytic-force composition for the VV10/rVV10
primitive. Self-consistent CUDA KS composition remains a separate backend gate.
Full WB97M-V composition/qualification is intentionally left to #935 and still
depends on the open #167 range-separated exchange work.

Refs #491, #903, #935, #167

Agent: ChatGPT
Model: GPT-5.6 Sol
