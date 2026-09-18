# Bounded CUDA ECP radial scheduling

The matched performance measurements below are historical qualification of
commit `0ba41df84e06e9d3a459c53afd5bbc8f109e02c0`, before integration with
master `1865c5a` (which adds f projectors and DFT energy support). They are
not measurements of the merged PR head. The merged schedule uses master's
16 projector components throughout allocation, indexing and resource bounds;
its separate integration evidence is described in `merge-validation.md`.

This slice starts at master `b78215ef0bf2cb43c91234c9c2dc852151605468`.
It changes the execution schedule of existing generated ECP arithmetic,
not the radial/projector equations, supported methods or quadrature tolerance.
Refs #171.

The compiler selects a four-layer radial batch for at most 16 public AOs and
44 polar points. Larger shapes retain one layer. AO and projector sources are
evaluated once per radial layer. Each unique AO pair accumulates layers in
ascending order, preserving the former per-layer FP64 atomic additions.
An optional staging OOM retries one layer. Other CUDA errors, host failures
and minimum-workspace OOM retain their existing error categories.

## Acceptance and scope

- Independent CPU/Libcint raw matrices and all-center derivatives, Cartesian
  and spherical f orbitals, real Rb/Cs parameters, detached ECP centers and
  arbitrary fixed weights retain their existing acceptance gates.
- New value/derivative checks cover a 161-node incomplete batch and the
  44/45-polar fallback boundary. Native checks force an allocation budget
  between one-layer and four-layer capacity and require successful cleanup.
- Complete RHF/UHF endpoints must satisfy energy error below `2e-8` Eh and
  force error below `2e-6` Eh/bohr against independent PySCF on every sample.
- Complete first-call, initialized prepared replay, changed-geometry,
  two-item batch and explicitly budgeted endpoints are retained, including
  each sample's iteration count. The first call is not process-startup timing.
- A 29-AO Cartesian double-f case exercises the larger-shape fallback and
  does not claim an expanded CUDA budget inventory. Budget gates remain
  bounded to the existing 16-public-AO inventory.
- CUDA compute-sanitizer checks cover native OOM/error recovery and a tail
  batch. The independent CPU production/oracle implementation is retained.

The timing fixtures are LANL2DZ Na / STO-3G H (9 AO RHF/UHF), the same system
with a spherical f shell on H (16 AO RHF), and Cartesian f shells on both
atoms (29 AO RHF). They are the existing independently tested ECP fixtures.
The reference installation supplies parameters only to the test/benchmark;
production receives owned basis records and does not import PySCF.

## Work, memory and ownership

On the RTX 4090, the measured complete endpoint times were:

| Fixture | Warm baseline / candidate (ms) | Changed geometry baseline / candidate (ms) |
| --- | ---: | ---: |
| 9 AO RHF | 691.156 / 547.489 | 1016.050 / 698.476 |
| 9 AO UHF | 690.446 / 547.321 | 1016.162 / 697.988 |
| 16 AO RHF | 8369.013 / 8069.048 | 9740.514 / 9209.397 |
| 29 AO RHF fallback | 1154.213 / 1137.469 | 1887.114 / 1876.546 |

Warm and changed-geometry entries are medians of three samples. All paired
samples have identical SCF iteration counts. The 9-AO cases reduce warm time
by 20.7–20.8%, changed-geometry time by 31.3%, and two-geometry batch time by
31.2–31.3%. The 16-AO improvements are smaller (3.58% warm, 5.45% changed).
The fallback's single first-call sample is **6.55% slower**, from 2445.625 to
2605.910 ms, while its repeated endpoint medians differ by only 0.4–1.5%.
These measurements do not establish a universal speedup or statistical
significance for small differences. Exact samples, initialization time and
budgeted/batch endpoints remain in the two endpoint JSON files.

Every endpoint sample passed its independent PySCF gates. Across candidate
samples, maximum energy error is `9.326e-15` Eh and maximum force error is
`1.095e-9` Eh/bohr. Validation also passed 51 CPU Python tests (six CUDA
skips), 35 CUDA ECP tests, 33 HF/KS resource tests, the generated CPU native
projector test, and all three CUDA ECP CTests. Both native OOM/error recovery
and incomplete-batch Compute Sanitizer runs report zero errors. The initial
native OOM test incorrectly budgeted less than one sphere-plus-layer; its
failed log is retained. The corrected test derives sphere capacity from
`sizeof(EcpSpherePoint)` and passes, including under the sanitizer.

Nsight Systems records a raw call with 161 radial layers, 32 polar points,
64 azimuthal points, nine AOs and one ECP center. The raw kernel summary
checks actual AO-evaluation, projection and contraction launches independently
of the timing driver. Semantic work remains 2,967,552 AO point evaluations,
13,041 projector components and 7,245 unique pair/radial contributions, with
one source pass. A launch reduction must not be described as less integral work.

The actual raw trace reduces each of the three launch families from 161 to
41, totaling **483 to 123 launches** (74.5% fewer). Projection kernel time
falls from approximately 119 to 31 ms, while contraction rises from 73 to
119 ms. The endpoint measurements, rather than each kernel individually,
support the bounded schedule choice.

Four-layer staging trades extra temporary device memory for fewer launches.
HF and KS CUDA resource bounds share the emitted schedule policy; the CPU
workspace is unchanged. Selection uses public AOs while capacity uses the
conservative Cartesian count. Owned-buffer ledger peaks exclude driver,
library-internal, graph and pool allocations.

| Fixture | Planned device bytes, baseline / candidate | Owned device peak bytes, baseline / candidate |
| --- | ---: | ---: |
| 9 AO RHF | 2,343,376 / 5,696,560 | 1,616,872 / 4,970,056 |
| 9 AO UHF | 2,378,528 / 5,731,712 | 1,634,448 / 4,987,632 |
| 16 AO RHF | 4,675,744 / 11,754,688 | 3,038,560 / 8,999,776 |

These budgeted samples record zero rejected allocations and 29 / 25 owned
allocation events. The larger fixture is outside the budget inventory;
its JSON zero-valued resource placeholders do **not** measure zero usage.
The host estimate for CUDA setup also grows conservatively with the shared
workspace bound; this is not a measured host-memory increase.

The CUDA adapter remains conservatively classified as scientific in the
ownership ledger. No handwritten scientific body is retired in this slice.
Removing the added schedule helper from the generated ECP header must recover
the exact previous generated header, proving the scientific emission unchanged.

The native adapter grows from 278 to 296 physical lines, or 272 to 285
nonblank/noncomment code lines (+13 conservatively scientific ledger lines).
The generated header grows from 34,335 to 34,489 bytes and 1,196 to 1,200 code
lines; the four-line schedule helper accounts for the complete difference.
The original ownership snapshot retained unchanged generated-family measurements
from the base build and updated ECP with the verified candidate header. The
merged snapshot is regenerated as described in `merge-validation.md`. This
slice does not claim handwritten-science retirement.

## Reproduction and retained evidence

Build Release with CUDA 12.9, `CMAKE_CUDA_ARCHITECTURES=89`, AOT shells off,
and the pinned reference dependencies. Set `PYTHONPATH=python`, explicit
`VIBEQC_LIBRARY`, `VIBEQC_PROFILE=off`, and one OpenMP/OpenBLAS/MKL thread.

```sh
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_schedule.py -q
python tools/benchmark_ecp_schedule.py --repeats 3 --output endpoints.json
nsys profile --sample=none --cpuctxsw=none --trace=cuda -o raw-trace \
  python tools/benchmark_ecp_schedule.py --trace-raw --output unused.json
nsys stats --report cuda_gpu_kern_sum --format csv raw-trace.nsys-rep
```

The existing baseline binary is reused only after verifying its recorded
hash and 604 relevant source/build inputs. The candidate is rebuilt separately.
Final source/binary identities, exact endpoint samples and profiler summaries
are retained here; raw build/test/profiler files remain in the external archive.
Results are limited to the reported RTX 4090 fixtures and do not establish
universal speed leadership or close the remaining #171 capabilities.

`source-identity.json` records the final measured nine-file overlay, not the
initial upload. All nine local LF-normalized sources and the locally emitted
header match those identities. `raw-evidence-manifest.json` binds 55 verified
raw files to the external `evidence-171/ISSUE-171-schedule-results.tar.gz`
archive (435,589 bytes, SHA256
`4a7fe2e98a0e44f4566d229d84fb7405c6b1589c6823474d588cefb768089bb9`).
It includes build/test logs, initial failed test attempts, exact measured
sources, collection scripts and Nsight report/SQLite files. These raw files
are kept outside Git; compact summaries and exact endpoint samples are here.
