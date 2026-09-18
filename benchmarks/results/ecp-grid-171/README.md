# ECP compiler-owned host grid qualification

This slice generates production host Gauss-Legendre nodes/weights, the radial
map and Jacobian, sphere coordinates/weights, real s/p/d harmonics and Cartesian
component coefficients. It preserves the independent CPU ECP implementation,
the one-radial-shell CUDA schedule, FP64, physical-center scatter and the
160/32 versus 224/44 complete-HF convergence policy. Refs #171; no expanded
angular/element/DFT domain or performance promotion is claimed.

## Source and environment

The baseline is upstream `755bcaa3531612bfb52a1ade216bd43ad2b8e564`. The measured
candidate is that Git archive plus the six exact files in the retained
`source-overlay.tar.gz`, not a claimed clean Git checkout. Complete per-file
manifests in the raw archive verify all source files and record overlay, header,
library and CMake configuration hashes. The CPU and CUDA candidate source
manifests must agree. Documentation and ownership-ledger edits do not change
the measured scientific files.

Hardware/toolchain: RTX 4090, CUDA 12.9, sm_89, Release, AOT shells disabled,
one OpenMP/BLAS thread. Sanitizer uses the retained CUDA 12.8 installation.
`summary.json` contains exact GPU/driver/compiler identities, matched endpoint
measurements and planned resource peaks. `raw-evidence.zip` retains unedited
build/test/benchmark logs, generated headers, kernel resource usage, complete
source identities and reproduction scripts. Its manifest hashes every member.

## Gates

Native host checks use independent polynomial moments through degree `2*n-1`,
standard-library Legendre root residuals, harmonic orthonormality/addition
theorems, analytic Gaussian radial moments and gamma-function normalization.
Orders include odd/even values, both integration grids and the maximum domain.
Invalid domains and nonfinite outputs are rejected. Python checks retain
stdlib-only deterministic generation and independent radial/harmonic formulas.

The integration gates remain CPU/Libcint raw matrices, all-center derivatives,
all radial powers, s/p/d projectors, Cartesian/spherical d shells, complete
RHF/UHF energy/forces, replay, budget failures and CUDA error recovery.

CPU native tests: **30/30**, plus the final finite-value grid-test rerun.
CPU Python: **45 passed, 10 GPU skips**. CUDA Python: **55/55**; CUDA native:
**2/2**; Compute Sanitizer: **0 errors**. Local compiler/input/ownership checks:
**34/34**. Source-bound candidate and CPU manifests verify **3,354 files**;
the pristine baseline verifies **3,353**.

The candidate's Libcint matrix error is `3.47e-12 Eh`; complete RHF/UHF force
errors are `5.53e-13 / 4.65e-16 Eh/bohr`. CPU/GPU matrix/derivative differences
remain at or below `2.23e-16`. Planned host/device/pinned peaks are identical.
All five ECP kernel resource records are byte-for-byte unchanged: AO 56
registers/64 stack bytes, projection 28/64, contraction 72/32, consumer 38/0
and convergence check 14/0.

The original three-sample complete RHF medians are 985.16 ms (baseline) and
988.48 ms (candidate). Original UHF samples are 981.46/984.69/984.89 ms versus
984.51/1164.98/1209.93 ms. The latter triggered a predefined five-sample ABBA
follow-up (`grid-performance.sh`); the initial measurements remain retained.
These small samples are regression diagnostics, not a statistical speedup claim.

The ABBA follow-up completed all four runs with five warm samples per method
and run. Pooled baseline/candidate RHF medians are **987.93 / 988.60 ms**
(+0.068%); UHF medians are **985.51 / 984.86 ms** (-0.066%). The earlier UHF
slowdown did not recur; its cause is not established. No material endpoint
regression is observed in this bounded check. All original and follow-up
samples, GPU snapshots and exact library hashes are retained in the **52-file**
archive, including successful exit markers and the predefined run order.

## Reproduction

1. Obtain the public baseline with `git archive` at the exact commit above and
   extract the raw evidence archive. Verify all SHA-256 entries in its manifest.
2. Use separate baseline and candidate source trees. Extract the retained
   `source-overlay.tar.gz` over the candidate. The overlay can also be supplied
   as `grid-changes.tar.gz` to the retained preparation script. Check the full
   source manifest, including files outside the modified set.
3. Adapt only the absolute workspace prefix in `grid-prepare.sh`, `grid-run.sh`
   and `grid-provenance.py`. Preserve the flags, environment and test commands.
   Run preparation, `grid-run.sh cpu`, `grid-run.sh baseline`, then
   `grid-run.sh cuda`. Use sequential baseline/candidate GPU timing.
4. If reusing a build directory, explicitly regenerate the ECP header and touch
   modified native sources as the script does. Archive mtimes must not permit a
   stale generated header or object to pass qualification. For a fresh compiler
   file inventory, allow CMake's `CONFIGURE_DEPENDS` reconfiguration to complete.
5. `grid-collect.py` verifies successful exit markers, matching candidate source
   manifests and kernel-resource records before packaging. The final native
   grid test additionally rejects nonfinite harmonic/coefficient values; its
   CPU rerun and source identity are retained in `cpu-grid-recheck*` logs.
   Run `grid-performance.sh` before collection to reproduce the ABBA follow-up;
   collection requires its successful exit and verifies each measured library.

## Ownership accounting

The generated header grows from 15,768 to 21,910 bytes (+6,142). The CUDA adapter
remains conservatively scientific at 272 nonblank/noncomment lines (net zero);
the host `ecp.cpp` translation unit decreases from 232 to 219 such lines (-13).
The latter includes retained oracle code and is outside the CUDA-only ledger.
Formula migration is not misreported as a CUDA LOC reduction. Other retained
generated-file measurements remain intact and the aggregate is reconciled with
the exact ECP header bytes.

See the [current contract](../../../docs/ecp.md) and
[ownership decision](../../../.agents/notes/implemented/architecture/2026-09-17-ecp-host-grid.md).
