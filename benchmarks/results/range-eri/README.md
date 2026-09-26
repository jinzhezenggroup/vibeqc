# Experimental range ERI raw and weighted shell endpoints

These bundles accept numerical correctness of the explicit CPU/CUDA range
provider at `3ac4bbe6c9e8263b8537290ebd58d94431d94380`. They do not promote a
production schedule, speedup, RSH functional, range-separated DF route,
Hessian or omega derivative.

Each backend covers 16 shell cases: long- and short-range operators at
omega 0.63 inverse Bohr, psss/dpsp/fsss Cartesian and spherical functions,
dpsp exchange cotangents, and coincident psss centers. Each case retains
three selected raw components, a weighted scalar, all twelve center
derivatives, five synchronous timing samples, three directional finite
difference steps, translation checks and an LR+SR=Coulomb check. Libcint
through PySCF 2.14 independently supplies each range part. Raw/weighted
FP64 gates use absolute tolerance 1e-11 and relative tolerance 1e-10;
the directional finite-difference gate uses absolute tolerance 1e-6.

| Backend | Maximum raw/weighted absolute error | Numerical result |
| --- | ---: | --- |
| CPU | 1.9804e-14 | Pass |
| RTX 5090 CUDA | 1.9846e-14 | Pass |

`evidence.json` contains the shared numerical gates and provenance.
`samples.json` retains actual/reference arrays and timing samples.
`artifacts.json` records source, binary, program and compiler-resource
identities. `resources.json` contains shared resource plans, including
explicit exclusions for caller streams, object metadata, native call stacks
and CUDA context. Numeric native allocation amounts are checked during
preparation; process-wide peak memory was not measured. No matched provider
performance comparison was attempted.

Reproduction commands are in each `publication.json`; install NumPy and
PySCF for this reference driver and put the recorded C++/CUDA compiler on
`PATH`. CUDA commands require the finite Slurm allocation shown in the
manifest and preserve its assigned device visibility. Reproduction uses the
source revision above and `tools/validate_range_eri.py`; no remote archive,
compiled binary or generated source file is required.

Separate automated tests cover all eight ERI permutations, orbit folding,
atomic-force scatter with two slots on one atom, malformed records,
capacity/identity preflight, partial/empty chunks and failed-call replay.
Native CPU ASan/UBSan and CUDA memcheck probes passed for both range families.
