# Decision: reuse xTBloom D3 mathematics as a qualification baseline

Status: implemented
Date: 2026-09-19

## Problem

Issue #492 initially described both D3 backends as new implementation work.
xTBloom already has GFN1 D3(BJ) CPU and CUDA energy/gradient implementations.
Searching only for `add_d3_cpu` missed the real device implementation inside
`gfn1_classical_corrections.cu`. Its generic interaction tag being reserved does
not imply that GFN1's internal D3 is CPU-only.

## Decision

Extract the D3-only device arithmetic, sharing it with a small CPU qualification
harness. Keep xTBloom's independent source/provenance records and simple-dftd3
1.4.0 numerical oracles. Generalize damping/cutoff parameters into immutable
`D3Spec`, and represent it as a distinct MethodIR correction node. Existing
semilocal graph identities and native execution restrictions are preserved.

This is a reference/qualification slice, not permanent method-specific native
production arithmetic. The compiler's existing scalar/tensor primitives must own
subsequent generated production lowering; runtime code must own state, streams,
resource lifetimes and errors. The current harness is intentionally separate
from `libvibeqc` and wheel assets.

## Rejected alternatives

- Reimplementing D3 physics from scratch or claiming GPU D3 is missing.
- Migrating the entire GFN1 runtime, halogen correction or SCC state.
- Calling the migrated CPU and CUDA pair an independent verification.
- Hard-coding GFN1's parameters/cutoffs into generic DFT method definitions.
- Simultaneously changing the equations, data packing and pair-parallel schedule.

## Invariants

Complete CN-response gradients, explicit gradient/force sign, immutable data
identity, nonzero ATM rejection, original data licenses, unsupported native
method rejection, and finite memory gates must remain tested. GFN1's `s9=0`
does not supply an ATM implementation. `[npair,49]` storage and a serial worker
per molecule are a correctness baseline, not a scaling/performance claim.

## Evidence

The nine tiny fixtures are independently generated with simple-dftd3 1.4.0,
with GFN1, PBE and PBE0 damping parameters and ATM explicitly disabled. Tests
also cover multi-step derivatives, atomic permutation/translation/rotation,
cutoff switching and coefficient scaling. Backend provenance is queried from
the compiled library rather than supplied as a caller label. See the PR for
executed CPU/CUDA qualification results; compilation alone is not a GPU run.

## Consequences and revisit conditions

Only the essential D3 tables and 86 radii are imported, not complete GFN1 model
data or large logs. The single-molecule synchronous harness and native logical
budget are explicit limitations. Revisit this baseline after generic provider
binding and compiler lowering have independent parity, then optimize parallel
pair reductions and data reuse separately.

## References

- VibeQC #492, #396 and #163.
- `manifests/xtbloom-d3.json` and `docs/dft_d3.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
