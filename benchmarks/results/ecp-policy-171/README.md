# ECP generated admission policy qualification

This change moves the fixed integration-grid dimensions and finite absolute-error
admission predicate into the compiler-owned ECP header. It preserves the
independent CPU oracle/fallback, numerical formulas, thresholds, scheduling,
FP64 accumulation and resource budgets. Refs #171 and #349. It does not qualify
new elements, DFT/ECP forces or a performance improvement.

## Measured source and retained evidence

Baseline: `2bf9a2a4784fc79f6048ec9c4e74701e1caa92ec`.
Candidate: `f10e7346839f059521c4a4800ae1a0655e21f1fd`.
The initial follow-up commit adds only documentation and evidence. Both measured trees came
from exact Git archives, with no scientific source overlay. The baseline used
fresh CPU/CUDA build directories; the candidate reused them after refreshing
source mtimes to force generation and recompilation.

The subsequent CuMetal compatibility fix uses unqualified `fabs`, as the other
generated arithmetic does, to resolve the CUDA device overload. CuMetal's
`std::fabs` wrapper is host-only. The original records retain their measured
revision; the [compatibility results](compatibility/summary.json) separately
qualify `37d3c1230d8a6db08952aceac73d455e82bd4a32` with new header/library
hashes. The only changed source input is `integral/ecp_policy.py`, and the only
generated-header change is the `std::fabs` to `fabs` spelling.

Compatibility requalification passes CPU native 2/2, CUDA native 4/4, CPU
Python 101/17 explicit CUDA skips, the focused CUDA ECP suite 10/23 non-CUDA
deselections, and all 33 resource tests. The same 80 boundary cases pass on
host/device; all three sanitizer runs have zero errors, including both complete
RHF/UHF replay cases. Independent matrix/energy/force gates and unchanged
resource-peak gates pass again. All 30 raw members and 1,463 source inputs verify
against the downloaded archive and exact Git blobs. The fixed source also
passes [CuMetal Apple GPU CI](https://github.com/jinzhezenggroup/vibeqc/actions/runs/35428000824)
and [NVIDIA sm_120 compilation plus CPU/Python CI](https://github.com/jinzhezenggroup/vibeqc/actions/runs/35428000713).
The Apple CI exercises its normal runtime test; ECP numerical qualification is
the real-FP64 NVIDIA run, not an implied Apple ECP capability claim.

To repeat this follow-up, use the same toolchain and commands below with the
compatibility commit, saving outputs separately. The measured follow-up reused
the CPU/CUDA build directories after extracting that exact archive and touching
the changed emitter. Replace the broad CUDA Python selection with
`python -m pytest tests/python/test_ecp.py -q -k cuda`; keep the resource and
three sanitizer commands. Compare against the original candidate endpoints and
require identical generated headers after reversing only the `fabs` spelling.

`summary.json` records the test results, generated-header identities, numerical
errors and matched resource peaks. The four `*-endpoints.json` files retain
original samples and independent Libcint/PySCF comparisons. Timing samples are
diagnostic only. `*-source-identity.json` records every measured source input;
symlink contents are checked against the recorded target Git blob.
`verification.json` records local Git-object and downloaded-archive checks.
`toolchain.json` records the GPU, driver, CUDA and compiler versions.

The generated CPU/CUDA headers must match within each revision. Removing only
the new admission-policy fragment from the candidate header must reproduce the
baseline header byte for byte. Every native suite must pass without a skip.
The CPU Python suite intentionally skips explicit CUDA tests; the subsequent
CUDA suites enable them. The three Compute Sanitizer runs cover the 80-case
policy kernel, native error/recovery paths and complete RHF/UHF replay.

`raw-evidence-manifest.json` binds all original logs, exit markers, generated
headers and scripts by size and SHA-256. These raw files and their archive stay
outside Git. The compact records here plus the source commits and commands
below suffice to repeat the qualification; no archive download is required.

## Results

CPU native tests pass 2/2; candidate CUDA native tests pass 4/4. Python results
are 101 passed/17 explicit CUDA skips on CPU, 76 passed/88 non-CUDA deselections
in the CUDA ECP selection, and 33/33 resource tests with CUDA enabled. Two PySCF
deprecation warnings concern the reference's `remove_linear_dep_` helper.
All three sanitizer runs report zero errors; the complete-HF run passes both
RHF/UHF replay cases. Both host/device policy tests exercise 80 boundaries.

Libcint matrix errors are at most `3.47e-12 Eh`; complete-HF energy errors are
at most `2.45e-15 Eh` and force errors at most `2.46e-12 Eh/bohr` across the
candidate CPU/CUDA endpoints. Resource peaks match the baseline exactly, and
the generated-arithmetic byte comparison passes. All 51 raw manifest members
and 1,460 baseline/1,463 candidate source inputs were independently verified
after downloading the archive.

## Reproduction

Use Linux with CUDA 12.9, an RTX 4090 (`sm_89`), Python 3.11, NumPy and PySCF
2.14.0. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
and `VIBEQC_PROFILE=off`. Use separate CPU/CUDA build directories and export
`PYTHONPATH` to the archive's `python` directory. For each exact revision above:

```bash
cmake -S "$source" -B "$build" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA="$enabled" \
  -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build "$build" --target vibeqc vibeqc_ecp_projector_tests \
  vibeqc_ecp_capability_tests --parallel 3
# CUDA only: build vibeqc_ecp_cuda_error_tests; candidate CUDA also builds
# vibeqc_ecp_policy_cuda_tests. The measured CUDA build used --parallel 5.
export VIBEQC_LIBRARY="$build/libvibeqc.so"
ctest --test-dir "$build" -R 'vibeqc_ecp_' --output-on-failure
python tools/benchmark_ecp.py --device "$mode" --repeats 2 \
  --output "$phase-$mode-endpoints.json"
```

Here `enabled` is `OFF`/`ON`, `mode` is `cpu`/`cuda`, and `phase` is
`baseline`/`candidate`. Run from the source root. Complete both baseline runs
before overlaying the candidate archive; touch every source input before
reconfiguring/rebuilding. Save each phase's generated header and library hash
before the next build overwrites them.

Candidate CPU suite:

```bash
python -m pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py \
  tests/python/test_ecp_validation.py tests/python/test_hf_resources.py \
  tests/python/test_ks_resources.py tests/python/test_cuda_ownership.py -q
```

Candidate CUDA suites, with the CUDA library selected:

```bash
export VIBEQC_ECP_CUDA_TEST=1 VIBEQC_RESOURCE_CUDA_TEST=1
python -m pytest tests/python/test_ecp.py tests/python/test_ecp_f.py \
  tests/python/test_ecp_f_projector.py tests/python/test_ecp_heavy.py \
  tests/python/test_ecp_stuttgart.py tests/python/test_ecp_multicenter.py \
  tests/python/test_ecp_dft.py -q -k cuda
python -m pytest tests/python/test_hf_resources.py \
  tests/python/test_ks_resources.py -q
compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$build/vibeqc_ecp_policy_cuda_tests"
compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$build/vibeqc_ecp_cuda_error_tests"
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  tests/python/test_ecp.py -q -k real_cuda_matrices_complete_hf_and_replay
```

The measured sanitizer was CUDA 12.8's `compute-sanitizer`. Admission boundaries
are independent literal cases on host and device: just below/at/above each
threshold, both signs, NaN/Inf, equal finite extremes, subtraction overflow,
signed zero and subnormals. Matrix errors must be below `2e-9`, complete-HF
energy errors below `2e-8` and force errors below `2e-6`. Baseline/candidate
resource peaks must match exactly; changes in independent energy/force error
magnitudes must be below `1e-12`/`1e-10`, respectively.

## Ownership accounting

`ownership-delta.json` separates physical edits from semantic reclassification.
There are 7 removed scientific-classed physical code lines and 7 added runtime
lines. Another 283 unchanged adapter lines move from scientific to runtime.
Thus the ledger delta is scientific -290/runtime +290, but this is **not a
290-line deletion**. The native adapter remains, and the independent CPU
`ecp.cpp` oracle/fallback is deliberately retained.
