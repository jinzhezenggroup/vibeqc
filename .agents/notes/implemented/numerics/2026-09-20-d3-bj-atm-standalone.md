# Decision: gate D3(BJ)-ATM as a standalone scientific primitive

Status: implemented
Date: 2026-09-20

## Problem

The production D3(BJ) runtime intentionally rejects nonzero `s9`, while issue
#492 requires a separately validated Axilrod-Teller-Muto (ATM) three-body term
with complete analytic derivatives. Folding ATM directly into Calculator or the
batched runtime would mix scientific validation with orchestration and CUDA
scheduling changes.

## Decision

Add a header-only, non-periodic `evaluate_d3_bj_atm` primitive and keep the
production D3 runtime unchanged. Reuse the existing D3 coordination-number and
C6 interpolation contract. Expose the already-audited packed `vdw_radii` from
`upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1_d3.json` through generated `PairData` rather than
introducing a second radius table.

For rational/BJ D3, preserve the simple-dftd3 ATM convention: `rs9 = 4/3` and
the effective zero-damping exponent is `alp + 2 = 16` when the tabulated BJ
`alp` is 14. The gradient includes direct three-body geometry derivatives,
smooth-cutoff product-rule terms, and the full coordination-number response of
all three interpolated C6 coefficients.

## Rejected alternatives

- Do not enable nonzero `s9` in the existing public D3 runtime in this slice;
  doing so would expose a new production capability before the standalone
  numerical gate exists.
- Do not copy the upstream vdW-radius table into a new source file. The same
  audited H-through-Rn packed radii are already present in the pinned xTBloom
  D3 JSON.
- Do not use the tabulated `alp=14` directly as the ATM exponent. The rational
  simple-dftd3 path passes `alp+2` to the ATM evaluator.

## Invariants

- `evaluate_d3_bj` continues to reject nonzero `s9`.
- The standalone ATM primitive uses the same 16*n workspace layout and C6/CN
  interpolation semantics as the existing D3(BJ) reference.
- CPU and CUDA execute the same host/device scientific function.
- A smooth ATM cutoff must differentiate all three switching factors.

## Evidence

- A static four-atom fixture was generated independently with the `dftd3 1.6.0`
  Python binding as the difference between otherwise identical `s9=1` and
  `s9=0` rational-damping evaluations. Energy and all Cartesian gradient
  components agree with the new primitive to numerical roundoff.
- Central finite differences cover both the unscreened path and a smooth ATM
  cutoff where the product-rule terms are active.
- A CUDA test compares the same smooth-cutoff calculation against the CPU path.

## Consequences

The generated D3 pair record grows by one double (`vdw_radius`) and the existing
CUDA D3 table upload therefore carries this field automatically. No public API,
MethodIR, Calculator, or batched D3 runtime semantics change in this slice.

## Revisit when

Integrate this primitive into the production D3 runtime only after the batching,
resource-planning, and public-parameter contracts for nonzero `s9` are reviewed
and separately validated.

## References

- #492
- `tools/parameters/dispersion_parameter_sources.json`
- `upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1_d3_manifest.json`
- simple-dftd3 rational damping / ATM implementation at the pinned source
  revision
