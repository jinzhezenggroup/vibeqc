# Scalar ECP nonlocal f-projector qualification

Extend the scalar semilocal operator through nonlocal f projectors, separately
from the orbital-f support delivered by #432. Compiler-owned polynomial
harmonics and reductions now cover 16 slots. The native constructor uses the
emitted bound, BSE/NWChem local labels extend through g, and the independent CPU
oracle and resource inventory cover the larger projection storage. Refs #171.

This qualifies a bounded synthetic operator extension with Cartesian/spherical
s/p/d/f orbitals, powers 0..4 and direct RHF/UHF. It does not qualify g orbitals
or projectors, a heavy-element parameter family, DFT, ECP density fitting, MP2,
or a faster schedule. Local label g does not mean orbital/projector g support.

## Source, environment and retained evidence

The baseline is master `dd7f9c9e2c16accbbbc1f23589f71b655b1c6450`, after #432.
The candidate is its Git archive snapshot plus the 16 files reconstructed by
`source.patch`. Full manifests verify 3,878 candidate files; CPU and CUDA source
identities match. `source-match.json` verifies each measured overlay file against
the staged Git blob after LF normalization. Documentation is outside this
numerical overlay. `summary.json` retains exact library/source/archive hashes,
hardware/toolchain, numerical errors, all timing samples, shapes and resources.

RTX 4090, CUDA 12.9, sm_89, Release, AOT shells disabled, PySCF 2.14.0,
one OpenMP/OpenBLAS/MKL thread. Compute Sanitizer is CUDA 12.8. Complete endpoint
timing includes synchronous fresh-system SCF and forces. CPU f qualification
uses one sample without an endpoint warmup; CUDA uses a warmup and three samples.
The raw ECP/reference setup precedes these calls, so the CPU number is not cold
library/process startup. The slower spherical CUDA endpoints are retained.

The 61-file diagnostic archive is retained under the original qb-ilm workspace's
`evidence-171/fprojector-results/raw-evidence.zip` and downloaded locally.
`raw-evidence.manifest.json` verifies its bytes and every entry. Routine build
logs, generated headers and failed runs stay outside Git. Reproduction does not
require this archive: the public base, patch, fixtures/tests and drivers suffice.
The repository copies of Python drivers are formatted and bind immediate loop
callbacks explicitly; the archive retains the exact original executed scripts.

## Merge-source reconciliation

`merge-source-match.json` records the reconciliation with upstream after the
merge-readiness review. The sole conflict was the generated
`docs/cuda_ownership_current.json`; it was regenerated from the combined source
and semantic ledger, retaining both upstream DF ownership and this ECP extension.
All 16 qualified ECP source/test files still match the retained measurement
hashes. Regenerating the ECP header also reproduces the qualified CPU/CUDA bytes.

The resolved tree passes 85 local IR/input/ownership/publication/build-structure
tests. One CMake command-execution test needs the CMake executable absent from
the Windows host and is left to Linux CI. Existing publication fixtures were
restored to their unchanged canonical Git bytes locally to remove CRLF-induced
hash failures. Compiler/SCF dependency and ownership checks also pass.
Incoming upstream CMake refactoring is exercised by the new-head CI build/tests;
the earlier hardware measurements below retain their original source identity
and are not presented as a fresh numerical run of the merged tree.

## Validation

- CPU CTest: **31/31**.
- CPU ECP/IR/input/f-projector Python: **66 passed, 22 expected GPU skips**.
- CUDA ECP CTest: **3/3**; GPU-selected Python: **22 passed, 66 deselected**.
- Compute Sanitizer: **0 errors** for native error/recovery and both
  Cartesian/spherical f-projector/f-orbital raw matrix/derivative cases.
- External-basis suite with canonical LF inputs: **45/45**.
- Local IR/input/ownership: **43/43**; compiler/SCF dependency checks and CUDA
  ownership inventory pass. `local-checks.json` records Windows limitations.

Independent Legendre addition-theorem and 16x16 harmonic Gram checks cover the
generated f mathematics. Libcint checks cover every power, signed multi-exponent
terms, separate local/nonlocal matrices, detached ECP centers, all-center
derivatives at two displacement steps and arbitrary nonsymmetric weights.
Both representations/backends pass complete RHF/UHF energy/force gates with f
orbitals. Budgeted changed-geometry replay passes complete-energy finite
differences and checks the CUDA allocation ledger. Raw value-only exports agree
with derivative exports, and the unchanged two-grid convergence checks pass.

Two retained CPU and five CUDA complete f-projector endpoints have maximum
matrix error **3.47e-12 Eh**, energy error **9.77e-15 Eh**, and force error
**5.15e-10 Eh/bohr** against PySCF/Libcint. This includes a 29-AO Cartesian CUDA
case with f orbitals on both Na and H. The 19/29-AO CUDA cases execute numerically
but their resource estimates remain explicitly unsupported; the existing public
CUDA inventory limit is 16 AOs. No larger-budget capability is claimed.

## Existing-domain regression and resource cost

Matched ordinary NaH/NaH+ complete endpoints, three warm API samples each:

| Endpoint | Baseline median (ms) | Candidate median (ms) | Change |
|---|---:|---:|---:|
| RHF energy + forces | 985.91 | 988.93 | +0.31% |
| UHF energy + forces | 980.74 | 984.71 | +0.40% |

These sequential small-fixture samples are a bounded regression observation,
not statistical speedup evidence. The candidate UHF samples include a slower
1044.07 ms call; it is retained with the other samples. Component-labelled
measurements zero the other coefficients but retain the full engine; they are
not isolated kernel timings. This extension always projects 16 rather than 9
harmonic slots per AO/radial shell, including inputs without f terms.

Projection workspace grows by `7 * 4 * 8 * nAO = 224 * nAO` bytes. Sphere records
grow from 104 to 160 bytes: +216,832 bytes for each refined 3,872-point grid copy.
The planner's existing 256-byte grid-record allowance covers the larger record;
its host/device estimates increase by 2,016 bytes on the 9-AO fixture. Planned
RHF host/device peaks are 2,837,696 / 2,345,392 bytes; UHF peaks are
2,879,168 / 2,380,544 bytes. These are planned bounds, not observed allocation
peaks. Budgeted execution separately verifies no rejected allocation and an
observed device peak within its declared limit.

All five kernels retain register/shared/local counts. The contract kernel stack
increases from 32 to 40 bytes; the other stack and constant-bank sizes are
unchanged. See both resource dumps. Generated ECP source grows from 34,335 to
35,562 bytes (+1,227). The mixed native CUDA adapter remains conservatively
scientific: **+9/-7 physical code lines, net +2 (272 -> 274)**; runtime delta 0.
No native production formula is retired; the independent CPU oracle remains.

## Diagnostic failures and reconstruction

The first incremental CUDA attempt preserved the archive's old timestamps and
did not rebuild the library. Both binary hashes equal the baseline, and the new
native projector/capability tests failed as intended. This attempt is retained
and excluded from qualification. `fprojector-run-cuda-v2.sh` explicitly touches
the overlay and regenerates the ECP header before building; all reported CUDA
results use the rebuilt candidate and matching binary identity.

Windows `git archive` applied CRLF conversion to the baseline snapshot. Measured
manifests describe those actual bytes, and source reconstruction uses canonical
Git line endings. The initial Linux external-input suite passed 44 tests but
failed its fixed-byte fixture hash test. `fprojector-input-lf.py` makes a separate
test copy and normalizes only the five pinned input files and their reference
generator; `input-line-endings.json` in the archive records both hashes. The
scientific Python and native library remain the measured candidate. All 45 tests
then pass. This does not modify the measured baseline/candidate source trees or
relax the pinned hash checks.

## Reproduce

Apply `source.patch` to the baseline checkout. Build Release CPU/CUDA libraries
with the commands in [the ECP contract](../../../docs/ecp.md), substituting the
allocated GPU architecture for 89 when needed. Install the pinned `reference-test`
extra, set `PYTHONPATH=python`, and point `VIBEQC_LIBRARY` to the exact build.

```sh
ctest --test-dir build-ecp-cpu --output-on-failure
python -m pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py \
  tests/python/test_ecp_validation.py tests/python/test_ecp_f_projector.py -q
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp.py \
  tests/python/test_ecp_f_projector.py -q -k cuda
python tools/benchmark_ecp.py --device cuda --repeats 3 --output result.json
```

`reproduction/` retains the original staging, CPU/baseline, corrected CUDA,
endpoint, input-line-ending and collection workflows. Adjust their workspace
paths on another machine; use the corrected `run-cuda-v2` driver. For new Git
archives use `git -c core.autocrlf=false archive` so pinned input bytes stay LF.
