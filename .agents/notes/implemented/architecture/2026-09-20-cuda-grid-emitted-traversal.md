# Decision: emit AO/feature/local-XC traversal beside the scalar compiler

Status: implemented
Date: 2026-09-20

## Ownership

The AO compiler now emits the seven existing AO, density/orbital feature,
sigma-finalization, XC-point, integration and potential-contraction functions.
Scalar Gaussian jets and feature algebra remain Graph-owned. Native cuda_grid.cu
retains maps, staging, symmetric scatter, arenas, resources, streams and ABI.
This is code ownership relocation, not a new discretization or numerical method.

## Evidence and invariants

All seven moved function definitions match their original C++ token sequences.
Preserve primitive/component order, partial occupied-tile accumulation, completion
of both spins before sigma, masks, deterministic reduction and error propagation.
Structure/source tests require definitions before the runtime include and prohibit
the retired native bodies. Compile the emitted translation unit as a whole;
string presence alone is not a link or device-numerical qualification.

## Alternatives and limits

Keeping duplicate scientific definitions or rewriting formulas during migration
was rejected. Runtime assets and compiler source jointly determine artifact
identity. No new performance result, public method or device residency is claimed.
Future scheduling changes require independent numerical and endpoint validation.
Refs #349/#163; tests/python/test_compiler_structure.py and test_grid_cuda.py.

Agent: ChatGPT
Model: GPT-6 Astra Pro
