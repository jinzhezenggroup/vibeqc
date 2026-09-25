# Decision: share bounded range moments with CUDA exact exchange

Status: implemented
Date: 2026-09-25

## Problem
The range-separated CUDA exchange provider needs the same bounded SR/LR radial moments as the CPU/generated range-ERI path. Importing the full integral header through the public CUDA kernel interface violated the SCF dependency boundary, while the original `std::` math calls were not device-callable under the CuMetal CUDA compatibility compiler.

## Decision
Keep `integrals/range_moments.hpp` as the single scientific owner of the bounded quadrature. Permit only the retained CUDA integral-numerics layer to include that exact header. The host/kernel launch ABI uses a CUDA-local transport enum, so provider host code and kernel interfaces do not acquire integral recurrence ownership. The shared moment evaluator selects device-callable global math functions for device compilation and the standard-library functions for ordinary host compilation.

## Rejected alternatives
Duplicating the SR/LR quadrature under `scf/cuda` was rejected because it would create two scientific owners that could drift numerically. Allowing the entire CUDA provider or kernel-interface layers to depend on `integrals/` was rejected because it would erase the existing architecture boundary.

## Invariants
- CPU/generated and CUDA range exchange consume the same bounded range-moment implementation.
- Only `integrals/range_moments.hpp` is admitted across the CUDA integral-numerics boundary; this is not a general `integrals/` exemption.
- Provider host code and public kernel-launch interfaces remain free of integral recurrence headers.
- Full-range CUDA Coulomb/Boys arithmetic and its rounding path remain unchanged.
- Range derivatives remain fail-closed until independently qualified.

## Evidence
The existing SR/LR CUDA provider tests compare restricted and unrestricted K against the independent CPU `build_range_eri` oracle. The SCF structure check enforces the narrowed dependency exception. CUDA and CuMetal compilation exercise the device-math path.

## Consequences
The scientific primitive is shared without duplicating the quadrature, at the cost of one explicit dependency exception for the retained CUDA integral-numerics layer and a small transport-enum conversion at the launch boundary.

## Revisit when
A repository-wide neutral numerical-primitives layer replaces both integral and CUDA-local numerical ownership, or when a separately qualified operator-specific range recurrence supersedes the bounded quadrature.

## References
- PR #1278
- Issue #167
- `src/integrals/range_moments.hpp`
- `tools/check_scf_structure.py`

Agent: ChatGPT
Model: GPT-5.6 Sol
