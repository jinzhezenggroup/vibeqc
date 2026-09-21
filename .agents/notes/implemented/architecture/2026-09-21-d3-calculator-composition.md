# Decision: compose D3 at the prepared Calculator execution boundary

Status: implemented
Date: 2026-09-21

## Problem

The production D3(BJ) owner and canonical MethodIR correction node existed, but
public Calculator execution still treated electronic KS and D3 as unrelated
endpoints. Passing a correction-bearing MethodIR directly to native KS was
intentionally rejected so that native execution could not silently omit D3. A
composition layer had to preserve that fail-closed rule, reuse the qualified D3
owner, preserve RKS/UKS spin semantics, and combine gradients with the correct
force sign without adding method-name-specific scientific kernels.

## Decision

Calculator accepts a spin-explicit MethodIR containing one production D3
correction, or the same graph through `KsOptions.composition`. It retains the full
graph as the public scientific identity and derives an electronic-only graph by
removing exactly the D3 correction node. Native KS receives only that electronic
graph. The existing `D3CorrectionBatch` receives the full graph and the same
fixed atom topology.

`PreparedBatch.execute()` is the sole composition boundary. Once the electronic
item has accepted a geometry and succeeded, the matching D3 result is combined as
`E_total = E_electronic + E_D3`. When forces are requested and the electronic
endpoint is independently admitted, D3's `dE/dR` is converted by subtraction:
`F_total = F_electronic - dE_D3/dR`.

The initial automatic MethodIR admission maps PBE-family semilocal primitives and
explicit spin to the existing PBE RKS/UKS runtime family. PBE0 therefore uses the
same PBE-family runtime plus its explicit MethodIR exact-exchange coefficient; D3
is not selected by the descriptive PBE/PBE0 method identifier.

## Rejected alternatives

- **Teach native KS to accept and ignore correction nodes.** This would defeat the
  existing omission guard and make a lower-level call capable of returning a
  mislabeled DFT+D3 result.
- **Add handwritten `pbe-d3` / `pbe0-d3` scientific branches.** D3 is geometry-only
  and already represented by MethodIR; named branches would duplicate arithmetic
  and violate the common composition goal of #396.
- **Compose only in `Calculator.singlepoint()`.** That would leave prepared/ragged
  batches, changed-geometry replay and per-item failure semantics on a different
  path.
- **Claim the existing global ResourcePlan covers D3.** The current planner owns
  electronic KS resources only. D3 keeps its explicit independent byte bound and
  composed global estimates fail closed until a real aggregate planner exists.

## Invariants

- A correction-bearing MethodIR is never passed to native KS and silently omitted.
- Exactly one D3 owner contributes exactly once to each successful composite item.
- D3 publishes `dE/dR`; public force composition always subtracts that gradient.
- D3 cannot grant electronic force capability. If the electronic endpoint does not
  advertise forces, the composite method does not either.
- A malformed geometry rejected by the electronic per-item boundary is not passed
  as a changed geometry to D3.
- D3 table/radii/correction identity remains checked by the production D3 owner.
- ATM, zero damping, D4 and other correction families are not admitted by this
  D3-specific composition slice.

## Evidence

`tests/python/test_d3_calculator_composition.py` checks PBE and PBE0 energy
composition against independent standalone production D3, exact force-sign
composition on the already-qualified CPU ECP force domain, full-versus-electronic
MethodIR identities, ragged changed-geometry failure isolation, and fail-closed
global resource estimation. Existing standalone D3 simple-dftd3 fixtures continue
to qualify the correction arithmetic and complete CN-response gradient.

## Consequences

The electronic SCF remains unaware of a geometry-only correction and can retain
its existing cache/solver/runtime contracts. Composite execution owns one extra
persistent D3 context/batch and one separately bounded resource domain. Result
objects now expose the correction component explicitly.

## Revisit when

Revisit this boundary if MethodIR gains a generic runtime correction scheduler
that can own D3/D4/gCP together while preserving the same per-item failure and
resource contracts, or when the global resource planner can represent multiple
prepared owners as one admitted plan.

## References

- #492
- #396
- #549
- #627
- #718
- `docs/dft_d3.md`
