# Decision: retained stationary owners keep per-call admission

Status: implemented
Date: 2026-09-20

## Ownership

Prepared stationary execution retains source, grid and TensorIR owners under a
batch item. Its topology key excludes solve epochs and coordinates so a fresh
validated D/W snapshot and a geometry-only rebind can reuse the same allocations.
The live stationary contract still guards every calculation and publication.
Changed topology requires reconstruction; failed work requires reset/rebind.

## Budget rule

Budget caps are per-call admission requests, not mathematical topology. A replay
must check both retained host and device bounds before changing coordinates or
touching a retained owner. Keeping caps out of the topology key avoids treating
a larger acceptable budget as a new scientific execution plan. This does not
expand the documented additional-memory scope to total SCF/process memory.

## Rejected alternatives and evidence

Checking only initial allocation permits a later smaller cap to be bypassed.
Checking only after rebind mutates state for an inadmissible request. Four new
host-only tests use explicitly mocked device owners and reproduce both lowered
caps with unchanged/changed coordinates. They fail before repair, pass afterward,
and verify successful replay at the exact bound. They are not GPU numerics.
Full device numerical/performance qualification remains separately attributed.

Agent: ChatGPT
Model: GPT-6 Astra Pro
