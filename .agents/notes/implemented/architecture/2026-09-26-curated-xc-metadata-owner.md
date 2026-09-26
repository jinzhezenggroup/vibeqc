# Decision: centralize curated native XC execution metadata

Status: implemented
Date: 2026-09-26

## Problem

After the bulk Libxc production/admission stack landed, the native curated KS path still repeated the same execution identity in several places: family code, display name, SCF-domain version, SCF-domain string, and CUDA availability. Python also used bare numeric family codes when selecting the curated work domain. Those copies made adding or changing a native family prone to silent identity drift.

## Decision

Keep the existing curated LDA/PBE/r2SCAN/B3LYP/WB97M-V kernels and public method behavior, but give their native execution metadata one C++ owner in `src/dft/semilocal_family.hpp`. Callers query that metadata instead of re-listing families or SCF-domain strings.

Python keeps the same stable transport codes but names them through `_NativeSemilocalFamily` and a symbolic domain mapping. The MethodIR composition remains authoritative; the refactor does not promote a bulk Libxc registration or infer capability from a numeric code.

## Retained compatibility boundary

`legacy_ks_execution_plan()` remains intentionally present. A C caller may omit `ks_options` and request an already-supported named method, so removing that adapter would change the current public preparation contract. The cleanup targets duplicated identity tables, not that compatibility path.

The existing curated specialized kernels also remain. #1119/#1121 still own generic native runtime and real molecular-SCF qualification; source generation or production-domain evidence alone is not sufficient reason to retire a qualified curated execution path.

## Invariants

- Stable curated codes remain LDA=0, PBE=1, r2SCAN=2, B3LYP=3, WB97M-V=4.
- Existing SCF-domain strings and domain versions are unchanged.
- CUDA admission still requires the metadata-backed `cuda_ks` capability; being present in the registry alone is not sufficient.
- Generated split-hybrid codes remain separately owned by the generated registry.
- No numerical formula, tolerance, backend, force capability, or public method is changed.

## Revisit when

After #1119 and #1121 qualify generic native CPU/CUDA molecular execution for admitted bulk semilocal programs, reassess whether the curated family enum and specialized dispatch can be retired entirely rather than merely centralized.

## References

- #1119
- #1120
- #1121
- merged #1314-#1339 admission/qualification stack

Agent: ChatGPT
Model: GPT-5.6 Sol
