# Decision: Bind stationary task kinds to the loaded component inventory

Status: implemented
Date: 2026-09-26

## Problem

A basis using only some SPD components selected a subset-shaped AOT request,
although the packaged artifact contains the full SPD inventory. Substituting the
full tuple only at load time would still dispatch the wrong derivative kinds:
compact JIT inventories and the full package assign different request indices.

## Decision

Select the qualified full SPD artifact for component-expanded bases. Construct
task-kind indices from its validated component-domain metadata, checking that
every basis request exists in that inventory. JIT artifacts retain their compact
basis-specific inventory. No additional component kernels are compiled at runtime.

## Evidence

The d-shell regression loads an actual package manifest and verifies every subset
request against the packaged inventory, including a request whose compact and
packaged indices differ. Opaque fixture binary bytes are integrity-checked but
never executed. Existing task tests cover public AO indices and expansion weights.
