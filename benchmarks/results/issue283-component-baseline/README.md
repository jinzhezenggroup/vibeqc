# DF force component baseline

These are measurements before any mathematical optimization for #282–#284.
The 96/192-AO probe uses the original zero-budget, host-resident compatibility
route. Source-backed generated execution is a separate positive-budget route;
its recomputation cannot explain these particular endpoint samples.

Both binaries use Release, CUDA 12.9.1, architecture 120, production optimization
and stable AOT shards. `unprofiled.json` identifies pristine source `640a81e` and
its library. `components.json` identifies diagnostic source `cf1a284` and its
different library. The trace adds event/synchronization overhead; these two
files do not establish a speedup. Full hashes, native-test results, raw traces,
and build settings are in `manifest.json` and `trace-library.json`.

| AO | unprofiled energy median, 3 samples | energy + force median | iterations |
|---:|---:|---:|---:|
| 96 | 0.455 s | 1.055 s | 34 |
| 192 | 5.556 s | 12.433 s | 39 |

Paired energies agree exactly and each pair has identical iteration counts.
The first 96-AO energy sample includes initial runtime setup (0.627 s versus
0.455 s subsequently); all raw samples are retained. These results agree with
the repository's refreshed version-2 #281 ledger, rather than the obsolete
version-1 numbers still quoted in issue bodies.

For the profiled 192-AO pair, named host intervals and measured synchronization
cover **97.1%** of its 6.913-s force increment. The largest exclusive component
is **host response-weight computation, 5.123 s**. Generated three-center
derivative contraction takes 0.588 s of CUDA-event time; one-electron/nuclear
derivative generation takes 0.873 s. Metric derivative contraction adds 0.00165 s.
The host compatibility route uploads about 56.9 MB of response/metadata and
gathers 169.9 MB from its existing raw host tensor. These are logical bytes,
not measured hardware bandwidth or the complete resource peak.

CUDA-event intervals surrounding host response computation include GPU idle
time; the corresponding 5.128-s interval is **not GPU kernel execution**.
`baseline.nsys-rep` and its kernel/API tables provide the independent device
timeline. The full 192-AO energy-plus-energy/force pair contains approximately
2.35 s of kernel work, making the large host portion visible independently.
The Nsight run has its own endpoint samples and binary identity.

The profiled 96-AO pair has a negative outside-operations residual (-0.137 s)
because the first energy-only solve includes initialization. Its 130% ratio is
retained as a diagnostic of pair variation, not treated as an attribution
success. Further warmed/repeated measurements are required for that coverage
gate.

J/K capture records describe graph construction only. They are kept separate
from executed stream calls and are not multiplied by SCF iteration counts.
The raw Nsight graph-node capture is retained under an exact-content evidence
policy exception so reviewers can inspect actual replay.

Validation: 151 host protocol/structure tests and four Slurm native suites
passed (DF numerical/reference checks, direct HF, CUDA Fock provider and Fock
composition). No priority issue is complete on the strength of these counters.
