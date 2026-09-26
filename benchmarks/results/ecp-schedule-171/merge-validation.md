# Integration with master 1865c5a

PR #458 integrates master `1865c5a`, including f projectors, DFT energies,
Stuttgart parameters and mixed physical-center qualification. The original
matched timing results in `README.md` remain bound to `0ba41df` / `b78215e`.
The integration endpoint report is a candidate-only correctness and resource
check; it does not establish a matched speedup against current master.

## Resolution and invariants

The native adapter retains ordered radial accumulation and the optional
single-layer OOM fallback. It uses the current compiler-owned 16-component
projector count for all offsets, launch sizes and staging capacities. HF/KS
resource planning includes four layers of 16 components. Tail and 44/45-polar
boundary tests cover both ordinary and f projectors, values and derivatives.

Fresh generation from master and the merged sources proves that removing
only the schedule helper exactly recovers master's generated ECP header.
`merge-generation.json` records both hashes. The ownership snapshot is
regenerated from the merged tree, with the materialized ECP header measured
separately; other unchanged generated-family entries are retained from master.
Native adapter size changes from 280 to 298 physical lines,
274 to 287 nonblank/noncomment lines; the conservative scientific role is
unchanged. No handwritten-science retirement is claimed.

## Validation

- Release CUDA build on the same RTX 4090 / CUDA 12.9 environment completed.
- Native ECP CTest: 3/3 passed.
- CUDA schedule, f-projector, DFT, multicenter and Stuttgart suite: 53 passed,
  42 deselected. Raw schedule comparisons also use the independent CPU path.
- HF/KS resources: 33 passed.
- Compute Sanitizer: zero errors for native allocation/error recovery and
  tail checks (three tests passed, ten deselected).
- Local compiler/SCF/input/schedule/ownership checks: 172 passed, 12 CUDA
  skips. Compiler/SCF structure, Ruff/format and ownership checks passed.

The first incremental build used old archive timestamps and retained a stale
generated header, causing a compile failure. Its logs are preserved. The
successful rerun uses `tar -xzmf` to refresh input timestamps and rebuild
dependencies; source content is bound by SHA256 independently of timestamps.

`merge-source-identity.json` binds the library, generated header and all 28
source overlay files. Local LF-normalized source and staged Git blobs are
checked against that identity. `merge-endpoints.json` retains every complete
endpoint sample; `merge-summary.json` records numerical maxima, timings and
allocation observations. `merge-raw-manifest.json` identifies the external
raw archive, including exact sources, scripts, build/test/sanitizer logs and
the failed initial build. No original measurements are overwritten.

## Final integration endpoint observations

Every recorded endpoint sample satisfies the independent gates. Maximum
energy error is `8.882e-15` Eh; maximum force error is `1.095e-9` Eh/bohr.

| Fixture | Warm median (ms) | Changed-geometry median (ms) | Owned device peak (bytes) |
| --- | ---: | ---: | ---: |
| 9ao-rhf | 541.448 | 695.061 | 5,194,952 |
| 9ao-uhf | 541.405 | 694.717 | 5,212,528 |
| 16ao-rhf | 8008.334 | 9186.439 | 9,230,944 |
| 29ao-rhf-fallback | 1146.830 | 1875.405 | outside budget inventory |

All budgeted cases record zero rejected allocations. The raw archive contains
47 verified files, is 96,211 bytes, and has SHA256
`1a8cde14387f4de8322345fcf2d1a15941410d075a413ba923c6087ccba8da7a`.
The GPU Notebook was stopped after retrieval and verification.

## Review correction: distinguish host bookkeeping OOM

Review of `f0ee7ec` found that the shared allocation ledger reports both device
OOM and host registry `std::bad_alloc` as `cudaErrorMemoryAllocation`. The new
radial staging catch therefore could retry a host failure as a smaller device
allocation, contrary to its stated contract.

The allocator now optionally reports host metadata failure separately, while
preserving the CUDA status for existing callers. ECP propagates that failure
before the radial fallback catch. Native regression checks distinguish tracked
and untracked device OOM from host registry failure, verify reservation/storage
cleanup, and retain the real one-layer budget fallback and recovery tests.

The RTX 4090 measurements and source manifests above remain historical evidence
for the pre-correction head. The correction changes allocation error handling;
the generated ECP header still matches the recorded SHA256 exactly. It does not
establish new performance measurements.
