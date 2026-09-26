# DF ownership retirement

This bundle covers the second retirement phase of #231. It compares the clean
phase-2 baseline `4f36c6ecf99fd27a07b26da7a073313fcd3aa679` against the clean
retirement candidate `3afbd95b2f33e49025d3fcad5d8d22e0fe4d2fac`. Both exact
histories are retained in `njzjz-bot/vibeqc`, at
`evidence/issue-231-df-baseline` and `evidence/issue-231-df-paired`.
The native source identity and timing-library checksum are bound to every
sample and original-object resource record. The candidate includes removal
of the former production response, its selectors, allocations and estimates.

All **90/90** case/phase non-regression gates pass; the worst median ratio is
**1.007245172** against the unchanged 1.02 ceiling. Maximum energy and force
errors are **1.70530257e-13 Hartree** and **5.88396346e-12 Hartree/Bohr**,
respectively. This supports structural retirement without a significant-speedup
claim. All subsequent native/GPU, independent raw/source and sanitizer gates pass.

## Measurement scope and rejected complete run

The first complete inventory-ordered comparison (Slurm 9229, source `635dc2c`)
passed every numerical gate and 89/90 performance gates. OH/def2-SVP UHF,
spherical, batch one, energy-only failed: ratio **1.033615423 > 1.02**. That
complete rejected run is retained losslessly in `rejected-inventory-run.json.gz`,
with explicit rejection and per-file checksums in `rejected-inventory-run.json`.
It is not accepted promotion evidence. The driver stopped at that failure, so
its later native/GPU/oracle/sanitizer stages did not execute.

`oh-diagnostic.json` retains every isolated worker, the diagnostic drivers and
complete extracted profiler reports with checksums. An isolated five-sample
diagnostic with the same OH preamble returned ratio 0.999955. Graph-node Nsight traces found unchanged cuSOLVER work accounting for
about 76% of GPU time and DF bulk kernel times of 6.206/6.265 ms. These observations
support a process-state/timing-drift explanation; they do not establish its cause
or replace a complete comparison. Bulk lane/block/symmetry experiments were not
adopted into production.

The final comparison (Slurm 9234) explicitly uses `--process-scope case`.
It runs the complete shared ABBA selection order for each case before advancing;
the historical inventory ordering can separate corresponding workloads by
minutes. All 18 cases, five samples per version, physical parameters, phase and
energy-only measurements, numerical tolerances and the 1.02 ceiling are unchanged.
Both process scopes remain reproducible. The final benchmark driver checksum and
process scope are bound to each worker and checked by publication. The compact
archive reconstructs all 180 original per-case process records byte for byte;
`process-files.json` retains each original hash and CPU integrity tests recheck it.

The candidate's original optimized native library and objects were built at
`635dc2c6e0314dd45656e5503754159dd8bd1794`. The measured child `3afbd95` changes
only benchmark/publication tools and their tests. All native compilation inputs,
the library checksum and retained object hashes are identical across these two
revisions. `validation.json` records this original-build identity explicitly.
These objects are original timing artifacts, not reconstructions. CPU integration
checks at `635dc2c` are reused only within that identical native-input scope.

The matrix contains 18 through-f RHF/UHF DF workloads, Cartesian and spherical
bases, batches of one and three, constrained source budgets and resident
plans, and water/OH with def2-SVP. Both sides hold the separately gated native
one-electron force route fixed. Each source has five samples, paired within
each case. Every sample includes cold, unchanged, moved and restored geometry phases. Each workload
also has a separate fresh energy-only singlepoint measurement. All 90
case/phase decisions use the original 1.02 median-ratio ceiling, with energy
atol 3e-10 Hartree and force atol 3e-9 Hartree/Bohr. Unrelated speedups cannot
offset a regression. The shared significance/noise decisions are retained;
structural non-regression is not a significant-speedup claim.

`samples.json` losslessly retains inputs, timings, energies, forces, residuals,
iteration counts, resource plans and observations. `summary.json` reports every
decision. `publication.json` binds the endpoint evidence by checksum and records
source fetch refs and reproduction arguments. `resources.json` contains records
for the original optimized `cuda_rhf.cu.o` and `df_derivatives.cu.o` resource dumps from
both timing builds, including object hashes, compiler identity and source/library
binding. These are original objects, not later reconstructions.

The candidate's value bulk/source/metric stacks are 320/208/176 bytes; its raw
derivative stacks are 936/1088/864 bytes. The weighted response uses 255 registers
and an 856-byte stack. Kernel resources do not establish a speedup or whole-HF
peak-memory reduction. DF metric diagnostics describe the value/J/K plan's
estimates; separately budgeted force staging and opaque allocations are outside
those diagnostics. Per-case whole-HF ledger limits and scope exceptions remain
in the samples. Cases beyond the public AO inventory have no total-budget claim.

`validation.json` retains exact CPU/GPU test logs and XML as text with their
original checksums, suite totals, sanitizer conclusions and source identities.
`raw-source.json` retains all 24 independent libcint/NumPy raw/source comparisons,
including coordinate derivatives and public transforms. Physical input files
are retained in `fixtures/`; the exact measured source regenerates the
independent references. These checks use the same final library as the endpoint
workers. Historical opt-in derivative evidence remains in its original archive.

## Complete integration and checkpoint validation

The accepted endpoint matrix and 18 native CUDA tests passed on Slurm 9234, but
its combined Python process stopped after 97 passing tests when a checkpoint
subprocess ran out of device memory. The remaining raw/source and sanitizer
stages did not execute in that job. Slurm 9235 reproduced the failure with about
21 GiB retained by the parent CUDA context, while fresh checkpoint processes ran
successfully. This is the already documented constraint on running large CUDA
suites alongside checkpoint subprocesses; final validation runs every complete
suite in a separate process.

Isolating checkpoint tests also exposed a pre-existing bitwise assertion on the
ordinary atomic one-electron force accumulation. Twenty independent runs fail
9 times on baseline `4f36c6e` and 10 times on candidate `3afbd95`, always by
5.55e-17. The existing serial generated response passes 20/20 runs on each
version. Validation uses child `ab3092d8686fa31ee7880d0ae09ff4231d86506d`, which
selects that schedule for this one bitwise assertion. H2, the exact energy and
force assertions, all numerical tolerances and every other default-policy
checkpoint test are unchanged. No production code or native compilation input
changes. This test source is retained at `evidence/issue-231-df-validation`.

Slurm 9238 reruns all 18 native CUDA tests and all 143 Python integration cases
across the six complete modules. The collected and executed node inventories
match exactly. Slurm 9239 completes the 24 independent raw/source cases and both
sanitizer runs; stage-specific provenance retains both jobs. `validation.json` records its distinct test revision
and the identical native source/library binding to the endpoint matrix.
`checkpoint-validation-diagnostics.json` and its lossless compressed companion
retain the earlier failed validations, memory samples and all 80 diagnostic
bitwise trials. The diagnostic failures are not accepted integration evidence.

## Reproduction

Fetch the recorded baseline, endpoint candidate and validation refs into
separate clean directories. Build the baseline and endpoint candidate with CUDA 12.9.1, sm_120, Release and
`VIBEQC_CUDA_FAST_COMPILE=OFF`:

```bash
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_CUDA_FAST_COMPILE=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
cmake --build "$BUILD" --parallel 4
```

Export absolute `ROOT` (endpoint `3afbd95`), `VALIDATION_ROOT` (`ab3092d`),
`BUILD`, `BASELINE_ROOT`, `BASELINE_BUILD`, `OUTPUT_DIR`, `PYTHON` and `CUDA_HOME`. Use a Python environment with VibeQC's test dependencies,
NumPy and PySCF. Run the retained reproduction driver through a finite allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=02:00:00 bash benchmarks/results/cuda-ownership/df/validate-gpu.sh
```

The script preserves Slurm device visibility, fixes BLAS thread counts, rejects
dirty/mismatched source and libraries, runs the full matrix and then the native,
Python, raw/source and sanitizer gates. Existing output directories are rejected.
With successful worker outputs, collect original object resources and publish
a new bundle using the current repository's publication tool:

```bash
python benchmarks/results/cuda-ownership/df/collect-resources.py \
  --comparison "$OUTPUT_DIR/endpoints" --baseline-build "$BASELINE_BUILD" \
  --candidate-retained "$BUILD" --cuda-home "$CUDA_HOME" \
  --output "$OUTPUT_DIR/resources.json"
python tools/publish_cuda_ownership.py \
  --comparison "$OUTPUT_DIR/endpoints" --resources "$OUTPUT_DIR/resources.json" \
  --destination benchmarks/results/cuda-ownership/new-df-comparison \
  --baseline-ref refs/heads/evidence/issue-231-df-baseline \
  --candidate-ref refs/heads/evidence/issue-231-df-paired
```

Publication independently recomputes numerical and all performance gates from
the workers. It never trusts the comparison's accepted flag or overwrites an
existing bundle. CPU integrity tests reconstruct both archived domains and
reject corrupted samples and resource bindings.
