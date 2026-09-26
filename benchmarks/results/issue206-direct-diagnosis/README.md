# Direct build capabilities and strict-force diagnosis

These records explain failures retained in [#421](https://github.com/jinzhezenggroup/vibeqc/pull/421).
They contain no replacement clean timings and do not qualify the original failed
acceptance. #206 remains open. All GPU probes ran in finite Slurm RTX 5090 jobs;
the independent PySCF/libcint oracle runs use CPU only.

## 192-AO direct execution

The original library has source identity
`ceafaa3df32bf05154e7532fb962c19b35c5d919603a9651b94274dcf17d0356`
and SHA-256 `6100ffe6cf4997935ff440b91e622e37dbb8b96207894291d8ff586071320d9d`.
Its CMake configuration disables AOT shell kernels. Its native device probe
reports `generic_cuda`, release mode and no fast-compile mode.

GDB job 9865 observes CUDA 801 (`cudaErrorNotSupported`) during the first direct
Fock construction, including the subsequent attempt without cuBLAS. The bounded
Fock consumer explicitly rejects uncovered shell classes. The AOT-disabled
registry publishes an empty generated Fock mask. This localizes the failure;
an AOT-enabled release reproduction is still pending and no successful replacement
calculation is claimed here.

The metadata regression reruns both 192-AO direct points with one requested
repeat, retaining both failures. Each point produced a progress journal with the
actual loaded library, hash and native capability probe before SCF starts. Those
two execution streams now restore from the checksum-bound
[`retention-488`](../retention-488/README.md) snapshot; the retained metadata
summary preserves their identities and failure conclusion without treating the
progress stream as another benchmark result.
The metadata check passes; both molecular calculations still fail. These are
failure-retention tests, not one-repeat performance measurements.

## Independent force errors

The CPU oracles use PySCF 2.14.0 / NumPy 2.5.3, FP64, the exact saved spherical
def2-SVP geometries, energy tolerance `1e-14`, orbital-gradient tolerance `1e-12`,
screening `1e-14` and 200 maximum cycles. The changed DF oracle explicitly uses
def2-SVP as its auxiliary basis and includes auxiliary-basis response. All six
CPU systems converge; orbital-gradient norms are at most `4.85e-13`.

Maximum absolute force errors across the original seven repeats, in Eh/Bohr:

| Retained endpoint | Native vs CPU | Stock vs CPU | Original paired-force gate |
| --- | ---: | ---: | ---: |
| Direct 96 / batch 1 | 3.7437501e-11 | 1.6966650e-11 | 3e-11 |
| Direct 96 / batch 4 | 3.9397423e-11 | 2.2087221e-11 | 3e-11 |
| DF changed 192 / batch 1 | 3.8579724e-11 | 7.3015372e-10 | 5e-10 |

Thus the direct failure includes an actual native deviation beyond the original
gate. The changed-geometry discrepancy is predominantly in the original stock
result. Both original paired failures remain unchanged.

## Separate convergence controls

Three accuracy-only native repeats at each setting preserve energy tolerance
`1e-12`, screening `1e-14`, ordinary cold preparation and an immutable post-cold
warm density. Only the native requested density tolerance changes:

| Density tolerance | Direct 96 / batch 1 vs CPU | Direct 96 / batch 4 vs CPU |
| --- | ---: | ---: |
| 1e-10 | 3.7446605e-11 | 3.9419353e-11 |
| 1e-11 | 1.6345160e-11 | 1.6813252e-11 |
| 1e-12 | 4.8209648e-12 | 5.6766813e-12 |

Stock changed-geometry probes keep the original bare density seed, exact target
geometry and DF model. Each tighter orbital-gradient tolerance has one probe:

| Gradient tolerance | Stock force error vs CPU | Final orbital-gradient norm |
| --- | ---: | ---: |
| 1e-9 | 8.8093088e-11 | 9.3405917e-10 |
| 1e-10 | 8.3808516e-12 | 9.7879990e-11 |
| 1e-11 | 1.6431301e-12 | 9.9225203e-12 |

These experiments diagnose convergence sensitivity. They do not modify the
committed direct gate's SCF settings or error tolerances, and have no speed claim.
The first stock attempt failed at geometry-format handling after saving all
native controls. The second failed because its gradient-norm call omitted the
explicit Fock argument. The third completes the stock-only probe. Both failed
attempts, partial results and exact scripts remain visible.

## Reproduction and validation

`manifest.json` remains the 27-record historical inventory; the current checkout
omits only the two progress streams listed in `retention-488`.
`reproduction/` preserves exact scripts as text; workstation paths are provenance.
Reuse the saved geometries, seeds, binary identity and controls for a fresh probe.
Keep accuracy/profiler runs separate from clean endpoint timing. A later
AOT-enabled library has a separate binary hash even if its source identity agrees.

Four focused CPU benchmark tests pass, including loaded-profile selection and
the unchanged direct/DF gate command contracts. The Slurm metadata regression
checks the actual failed molecular path. Remaining acceptance work includes the
AOT-enabled direct matrix, numerical fixes under the committed gates, and the
missing external endpoint/resource coverage listed in #421.
