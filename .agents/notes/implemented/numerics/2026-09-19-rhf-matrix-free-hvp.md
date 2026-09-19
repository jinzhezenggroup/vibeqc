# Decision: compose the RHF HVP from directional generated consumers

Status: implemented
Date: 2026-09-19

## Problem

Issue #180 needed a complete conventional RHF Hessian-vector product without
turning the tiny dense Hessian integration into the production algorithm. The
available pieces already included #178 shell-local second-integral
`weighted_hvp` consumers and #526/#530 directional H1/S1 plus one shared RHF
response solve, but the molecular relaxation and center-to-atom composition
were missing.

## Decision

`rhf_hvp` composes five contributions directly: nuclear repulsion, core
second integrals, overlap/Pulay second integrals, two-electron second
integrals, and electronic relaxation. The skeleton terms call #178
`weighted_hvp` programs with the physical direction expanded to every
mathematical integral center and scatter each recovered center response back to
physical atoms.

Exactly one directional CPHF solve supplies `D1(v)` and `W1(v)`. The
relaxation is then evaluated shell-locally from generated first derivatives as

`Tr[H1_R D1(v)] - Tr[S1_R W1(v)]`

for every output coordinate. The ERI first-derivative weight is the directional
derivative of the frozen RHF two-electron energy weight; no molecular ERI
Jacobian or all-coordinate H1/S1 tensor is formed.

The current B3 qualification keeps second-integral HVP execution and the final
relaxation contraction on CPU. Qualified CUDA directional H1/S1 and direct J/K
may be selected independently, producing an explicitly mixed host/device path.
Moving AO/MO transforms and Krylov to a resident device path is separate B2
work and is not required for the correctness slice.

## Rejected alternatives

- Form the tiny dense analytic Hessian and multiply by `v`: numerically useful
  as an oracle but violates the matrix-free B3 requirement and scales with the
  full coordinate matrix.
- Build all-coordinate H1/S1 and contract them after the solve: correct in the
  tiny domain but defeats the directional memory boundary.
- Wait for fully device-resident CPHF before publishing any HVP: unnecessarily
  couples B3 scientific closure to B2 execution residency.
- Differentiate a first-order custom VJP a second time: first-order implicit
  differentiation does not establish the required second-order response rules.

## Invariants

- One physical direction enters; no molecular Hessian or coordinate-indexed ERI
  derivative tensor is allocated.
- The response uses the same reference/operator identity and shared #179 solver
  as the existing directional response.
- Center-to-atom incidence is applied on both the HVP input and output, including
  repeated mathematical centers on one atom.
- Raw HVP symmetry is validated through the bilinear identity; no post-hoc
  symmetrization or projection may hide a missing term.
- CUDA-qualified stages and host stages are reported separately.

## Evidence

The B3 tests compare:
1. the shell-local relaxation contraction with the independently assembled dense
   CPHF relaxation block times `v`;
2. every HVP component and the total with the #449 tiny dense analytic
   `H @ v`;
3. `u.T @ H(v)` with `v.T @ H(u)`; and
4. the complete HVP with three step sizes of independently reconverged native
   analytic-gradient directional differences.

Existing directional-response regression tests continue to protect response
metric, convergence, reference identity, failure/replay, and no-dense-input
contracts.

## Consequences

The implementation is scientifically complete for the declared conventional
RHF tools domain but is not yet a production-size or all-device endpoint. Cold
code generation and repeated shell/component work are not performance claims.
B4 can assemble bounded block/full Hessians from repeated directional calls
without changing the scientific HVP contract.

## Revisit when

Revisit execution ownership when B2 provides validated resident AO/MO
transforms and Krylov, or when a shared generated first-integral scalar
contraction provider can replace the current CPU orchestration without changing
the coefficient graph.

## References

- #180
- #178
- #179
- #449
- #526
- #530
- `docs/hessian.md`
- `tools/vibeqc_hessian/hvp.py`

Agent: ChatGPT
Model: GPT-5.6 Sol
