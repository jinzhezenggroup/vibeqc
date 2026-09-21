# Decision: compile KS execution intent from MethodIR

Status: first execution-plan slice implemented
Date: 2026-09-21

## Problem

MethodIR already represents semilocal XC, full/range-separated exact exchange,
VV10/rVV10 and geometry corrections, but the Python/native KS boundary still
contains a second partial interpretation of those primitives. In particular,
the current native v2 descriptor can lower semilocal PBE-family scaling plus
one full-range K term, while RSH and nonlocal correlation remain separate
qualified providers. Adding another named-method branch at this boundary would
make omegaB97M-V and future compositions repeat scientific dispatch that
MethodIR already owns.

## Decision

Compile every self-consistent KS-capable MethodIR through one
`KsExecutionPlan`. The plan records:

- the canonical semilocal primitive;
- all full-, short- and long-range exchange contributions with exact
  coefficients, omega and spin-convention Fock coefficients;
- an optional nonlocal-correlation primitive;
- geometry-only post-SCF corrections separately;
- the primitive lowerers required to execute the plan.

The execution-plan identity is semantic and therefore shared by method aliases.
The descriptive method/manifest identity remains available in the full payload.

The current native v2 adapter consumes this plan and then applies an explicit
backend projection. Full-range exchange remains executable where already
qualified. Missing SR/LR exchange or nonlocal-correlation lowerers fail by
primitive name rather than by rejecting a named method. Public capabilities are
unchanged by this slice.

## Consequences

omegaB97M-V now compiles automatically to one scientific contribution plan:
its B97M semilocal primitive, 3/20 short-range exchange, unit long-range
exchange at omega=3/10, and VV10(b=6,C=1/100). This is not yet a public
omegaB97M-V SCF claim: the production native VV10 lowerer and the remaining
native RSH/public qualification still have to satisfy their own gates.

The next runtime slice should extend the native KS descriptor/owner to consume
the plan's exchange/nonlocal contribution records directly. That extension
should add primitive lowerers, not a WB97M-V method branch.

Refs #396, #167, #491, #720, #723.

Agent: ChatGPT
Model: GPT-5.6 Sol
