# Decision: compiler-owned auxiliary-g values with a separate derivative boundary

Status: implemented
Date: 2026-10-03

## Problem

The internal native CUDA molecular DF-CCSD composition in #1777 rejected the
auxiliary bases of both large targets: ethane 230 AO / aug-cc-pVTZ-RI and benzene
264 AO / cc-pVTZ-RI contain g shells. The original f-only molecular jobs on n2
(2164 and 2165) were cancelled before completion; their RHF/source/CCSD endpoints
have no accepted result. Checking source capability before expensive RHF remains
mandatory.

## Decision

Preserve the existing through-f Rys value path. Add a value-only auxiliary-g
lowering from the common Gaussian moment DAG: f/f/g three-center integrals and
g/g metrics use polynomial coefficients and exact Boys moments through F10.
The Boys evaluator, Gaussian product geometry and polynomial integration are
shared with the existing derivative emitter. Extracting those helpers preserves
the complete derivative generated-source SHA-256:
`71430c3d8929868c10156d36578f719bbd56d595b35c4ae56131930395f4ce04`.

The value inventory grows from 80 to 105 signatures, retaining the old records
and identifying g records with the existing subset-Wick mathematical IR. The
compiler emits 85 bounded one-axis cases with an unambiguous stride. All compute,
storage and accumulation remain FP64.

The source has an explicit immutable g/value-only capability. Its generic math
provenance is `generated_rys_auxiliary_g_polynomial`. Alternative experimental
math policies are rejected for g owners. Raw/transformed three-center and metric
derivative requests reject before launch or publication. Legacy exporters,
SCF orbital support and derivative capabilities keep their f restriction.

Reuse the host basis packer with an explicit DF-values mode: Cartesian metadata
through g, no warm density, occupation, shell-pair or quartet task construction.
The public g transform has its own six-term record; the SCF three-term ABI is
unchanged. This removes unused structural work rather than uploading unused
SCF tables and then releasing them. Existing single-stream source ownership,
capacity admission and failure draining remain intact.

## Rejected alternatives

- Increasing the Rys limit to six without qualified tables. f/f/g has total
  degree ten and cannot be represented by the existing five-root rule.
- Relaxing only the source shell guard. The shared packer also had an f limit,
  and the spherical g, m=0 expansion needs six Cartesian terms.
- Increasing every SCF expansion bound or advertising g forces. Neither follows
  from value-only qualification.
- Restarting large molecular RHF jobs before the source passes. This repeats
  the avoidable cancelled-job cost.

## Evidence

See `benchmarks/results/df-auxiliary-g-20261003/qualification.json`.

- 131 combined CPU/GPU tests: every new angular class, full Cartesian and
  spherical blocks, signed contractions, diffuse/tight/far regimes, unsupported
  roles, small native factors/blocks, rank truncation, budget admission and four
  H2 molecular CCSD cases against an independent determinant oracle.
- The same full test command under unfiltered memcheck: zero errors.
- 120 compiler/derivative/admission regression tests, with live PySCF 2.14.0.
- 32 native source raw/metric/J/K comparisons, covering four mappings, full and
  ragged tiles, two-item batches, mixed representations and duplicate auxiliary
  shells. Every g probe exercises derivative rejection with unchanged output.
- Release-enabled host metadata test verifies g is rejected by ordinary SCF
  packing, admitted only in DF value mode, and creates no unused SCF/task state.

## Large-source limitation and next investigation

Supplied-orbital source calls now complete at 230 and 264 AO, but neither passes
all previously declared `3e-10` absolute/relative factor/block gates. These are
not accepted large molecular endpoints. Retain their failed results explicitly.
The 230-AO maximum Bvv difference is about `2.55e-7`, with metric condition number
`3.51e8`; the 264-AO difference is about `3.20e-8`.

For 230 AO, separate native raw and metric reconstruction gives raw error
`2.76e-12`, metric error `1.71e-11`. The maximum raw error restricted to g
auxiliaries is only `1.14e-13`; the larger errors are in the unchanged f-and-lower
Rys path. Applying the independent metric whitening to native raw integrals
still gives Bvv error about `2.60e-7`, so replacing the metric eigensolver alone
cannot fix this gate. Do not relax the gate or ascribe the discrepancy solely
to the new g lowering. Retained ignored raw arrays support a conditioning and
precision investigation before another expensive RHF/CC solve.

## Revisit when

Qualify a more accurate source component where conditioning requires it, with
independent energy/amplitude/residual gates and complete-endpoint measurements.
Only then qualify large molecular composition, native triples and forces. Apply
#1763 fusion, #1764 component-wise precision and #1765 bounded buffering where
complete-consumer work/timing evidence identifies an actual bottleneck; this
change adds neither reduced precision nor asynchronous buffer ownership.

## References

- #1777, #1771: native molecular source and staged AO-to-MO traversal.
- #1763, #1764, #1765: fusion, precision and pipeline requirements.
- `.agents/notes/implemented/architecture/2026-10-03-native-cuda-df-cc-source.md`.
