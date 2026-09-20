# Decision: generate GFN2 short-range geometry from shared IR

Status: implemented
Date: 2026-09-20

## Boundary and equations

GeometryIR/PairIR fixes the molecular pair topology; TensorIR owns both primal
and coordinate VJPs. For pair distance r and scaled covalent-radius sum R,
the CN contribution is logistic(10*(R/r-1))*logistic(20*((R+2)/r-1)).
Each pair contributes to both atoms. Repulsion is Zeff_i*Zeff_j/r times
exp(-sqrt(arep_i*arep_j)*r^k), with k=1 only when both elements are H/He,
and k=1.5 otherwise. Distances are bohr. The exact 25-bohr cutoff includes
equality; topology changes require rebuilding rather than differentiating a
changing neighbor list. Near-coincident pairs below squared distance 1e-12 are
rejected, as are nonreal/nonfinite/malformed coordinates.

## Provenance and alternatives

The Z=1..86 table is bound to xTBloom 2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3,
tblite fa8a4416e8fe093d0075bc10ac875494c2a449a9 and parameter JSON SHA-256
de0f20e90b592b7b92f107eb672bd3dd29c1096f904d7a472b05693f9238ed1a.
The parameter identity includes the actual extracted numbers and conventions.
Copying a separate handwritten gradient or importing xTBloom at runtime was
rejected: both would create a second numerical owner.

## Evidence and limits

Pinned 16-atom CN and 24-atom repulsion references, coordinate finite differences,
translation invariance and stale-topology/coordinate-admission checks pass.
The shared TensorIR primal/VJP CUDA lowering is tested separately from execution.
No SCC, public GFN2 endpoint, ragged production-device promotion or new timing
result is implied. Revisit after heterogeneous/device endpoint qualification.
Refs #504/#501 and tests/python/test_gfn2_geometry_ir.py.

Agent: ChatGPT
Model: GPT-6 Astra Pro
