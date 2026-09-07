# Reproducible validation gates (CG01)

This protocol supports integral, tensor, and correlated-method development
without requiring a GPU to write or test CPU code. It adds fixtures and evidence
registration; executable production methods remain RHF and UHF. PySCF/libcint
is used only by the saved reference-generation script. Ordinary tests consume
committed data and require neither PySCF nor a CUDA toolchain.

## Existing tests and oracle independence

Every existing native and Python test is retained, including its tolerances.
The new small-fixture tolerances do not supersede stricter existing tests.

| Existing tests | Oracle and independence | What a pass establishes |
| --- | --- | --- |
| `tests/native/test_cartesian_integrals.cpp`, `test_density_fitting.cpp`, `test_spherical.cpp` | Handwritten CPU integral/derivative implementation in `src/integrals/s_integrals.cpp`, separate from generated CUDA; spherical transforms share production normalization code | CPU integrals, symmetries, derivative/DF contracts, and selected pinned libcint comparisons; shared transforms are not an independent transform oracle |
| `test_rhf.cpp`, `test_uhf.cpp`, `test_batch.cpp`, `test_cpp_batch.cpp`; Python calculator/batch/torch tests | Native HF plus pinned external energies/forces and invariants | Method/API endpoints, batching, error handling; self-comparisons alone do not establish independent accuracy |
| `tests/python/test_codegen.py`, `test_expr.py`, `test_integral_contracts.py`, `test_production_recurrence.py`, `test_rys_table_extraction.py`, `test_ppps_resident_rys.py` | Generator DAG, emitted-source checks, Rys versus symbolic recurrence | Structural/algebraic consistency; common geometry/Boys/normalization code can share bugs |
| Native mixed-precision/eigensolver/AOT tests and CUDA portions of Python tests | Generic CUDA versus handwritten CPU or pinned references; generic CUDA is separate from generated schedules but shares native dispatch/normalization | Only actually executed CUDA numerical/resource cases; skipped CUDA tests are not evidence |
| Existing benchmark/source tests (`test_benchmarks.py`, CUDA DF source and resident benchmark tests) | Pure runner/schema/source assertions; CUDA benchmarks explicitly invoked separately | Runner correctness and emission, not GPU execution |
| `tests/python/test_validation.py` | New saved PySCF/libcint values and analytic derivatives, compared with the generator host evaluator and native CPU HF | Independent small-fixture numerical acceptance, sign/order/version corruption detection, reproducible protocol behavior |

The host adapter in `tools/vibeqc_validation/integrals.py` is not another oracle:
it contracts the existing generator evaluator and compares it with libcint.
Its primitive normalization is explicit and tested against a different library.
The spherical f fixture is pinned for downstream consumers; this adapter only
executes Cartesian tensors. Existing native spherical tests remain in force.

Generate the capability table using the existing catalog:

```bash
python -m tools.vibeqc_validation.capabilities --output /tmp/capabilities.json
```

Each of the 55 canonical shell classes has separate **representation, source,
compilation, numerical, endpoint, and production** fields. The 34 f-containing
rows reuse #135's class matrix. Source support is attributed to the committed
emitter catalog; compilation and numerical stages await attached measurements.
Manifest selection is reported as historical state, not inferred acceptance.
FPPS is explicitly provisional until #135 supplies equivalent acceptance.
This protocol never edits the production manifest or compiles the full f
matrix in ordinary PR CI. The JSON table is a projection of the catalog, not a
second independent matrix that needs manual maintenance.

## Numerical conventions and fixture layers

`tools/vibeqc_validation/fixtures.py::CONVENTIONS` is the machine-checked
convention record included in every fixture:

- Coordinates are Bohr; energies are Hartree; gradients/forces are Hartree/Bohr.
  Forces are **minus** nuclear-coordinate gradients. libcint `int2e_ip1`
  differentiates the electron coordinate, so moving-center derivatives negate
  it. Each quartet center is differentiated through an explicit permutation;
  coincident coordinates retain distinct shell-center slots.
- Cartesian AOs use descending x power, then descending y power. Thus d is
  `xx, xy, xz, yy, yz, zz`. Each contracted component has unit self-overlap.
  Input coefficients multiply individually normalized primitives; each
  contraction is normalized afterward. libcint Cartesian arrays are converted
  using their overlap diagonal, with the scale saved in the fixture.
- Real spherical AOs follow libcint phases: p is `x,y,z`, and higher l follows
  `m=-l,...,+l`. They have unit self-overlap. Never infer the loaded angular
  momenta from an AO count or a basis name: `angular_momenta_loaded` is inspected
  with `bas_angular` and checked against every explicit requested shell.
- ERIs use chemists' `(ij|kl)`, stored in C order with l fastest. Physicists'
  `<ij|kl>` is chemists' `(ik|jl)`. Derivatives have leading `(center, xyz)` axes.
- RHF uses `P=2 Cocc Cocc.T`, `F=h+J[P]-K[P]/2`. UHF keeps separate alpha/beta
  densities with occupation factor one. The data contain real spatial orbitals;
  any spin-orbital expansion orders all alpha orbitals before all beta orbitals.
- The Hamiltonian is all-electron nonrelativistic Coulomb, with frozen core zero
  and explicit nuclear charges. Auxiliary-center identities and parent-atom
  bindings must be saved independently of orbital-center indices; the present
  four-center fixtures contain `auxiliary_centers=[]`.
- Canonical orbitals are ordered by energy within each spin. Their largest
  absolute AO coefficient is positive, with the first index breaking ties.
  No rotation inside degenerate subspaces is implied. Save actual coefficients
  and occupations; phase alignment alone does not resolve degeneracy.

The small layer uses PCG64 seed 138, asymmetric centers, contraction lengths
one through three, exponents between 0.4 and 2.0, and distinct, coincident, and
all-same-center cases. Same-center fixtures are **raw integral** cases; they do
not define an SCF molecule with overlapping charged nuclei. Exact near-zero
and translation-invariance checks accompany combined absolute/relative gates.

The molecular layer includes H2, He, H2O, NH3, CH4, and open-shell HF+ (UHF
doublet). NH3 includes CCSD amplitudes and a nontrivial perturbative triples
correction. All use the exact bundled STO-3G coefficients, copied into the
fixture; PySCF's rounded built-in basis tables are not substituted. These six
small HF endpoints run on CPU in normal tests. Larger def2-SVP/def2-TZVP
molecules remain in `benchmarks/_cases.py` and `real_molecule_gate.py` for manual
GPU endpoints. An f-shell endpoint must actually contain l=3 shells.

## Reproducing and checking references

With an already available PySCF/NumPy/threadpoolctl environment (no automatic installation):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/generate_validation_references.py \
  --output /tmp/validation-generation-1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/generate_validation_references.py \
  --output /tmp/validation-generation-2 --compare /tmp/validation-generation-1
```

Both commands expose `--help`. Each generation saves the script/definition,
basis-pack and libcint library hashes, Python/PySCF/NumPy versions, NumPy's BLAS
configuration, actual loaded BLAS/OpenMP versions and thread counts, SciPy/h5py
versions, one-thread policy, geometry, basis coefficients, SCF thresholds,
occupation/phase conventions, data hashes, and timestamps. The manifest pins
each complete reference record. `stability.json` traces both generations and
reports per-block differences; generation fails if the stability gate fails.
Mathematical identity includes array ordering and conventions, but excludes
fixture names, basis aliases, timestamps, and RNG history once the explicit
numbers have been saved. Reordering dictionary keys does not change a hash.

The committed two-generation stability result is measured on one dependency
stack, not a claim of cross-BLAS bit reproducibility. Regenerate and inspect
stability before upgrading references. Independent CC residual evaluation is
a downstream requirement: the saved PySCF amplitude-update diagnostic is
explicitly labeled as such and cannot substitute for an independent residual.

Initial gates for new, scale-controlled fixtures are:

| Quantity | Proposed acceptance |
| --- | --- |
| Small FP64 raw integrals/derivatives | `atol=1e-11`, `rtol=1e-10`, per element |
| Same-Hamiltonian CC correlation energy | absolute error <= `1e-8 Eh` |
| Independently evaluated CC residual | <= `1e-9` |
| Gradient against independent analytic reference | max error <= `1e-6 Eh/bohr`, target `1e-7` |

The small HF endpoint test uses `1e-10 Eh` and `1e-7 Eh/bohr`; the existing He
and molecular gates retain their stricter historical tolerances. These are
requirements, not a statement that VibeQC implements CC or CC gradients.

`finite_difference` records at least three positive distinct steps and every
error; it does not promote the most favorable step. The HF example uses 0.01,
0.003, and 0.001 Bohr, showing the truncation-error curve against independent
analytic gradients. Hold method, approximation, screening and frozen-core
policy fixed. Tighten SCF, CC and Lambda thresholds before differencing (the
reference SCF energy/gradient thresholds are `1e-13`/`1e-11`, CC energy/update
thresholds `1e-13`/`1e-11`; downstream CC gradients must also record Lambda
convergence at `1e-11` or tighter). A plateau or unstable reference requires
investigation, not a relaxed acceptance threshold.

## Execution tiers

Continue using the standard commands:

```bash
cmake -S . -B build -G Ninja -DVIBEQC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
VIBEQC_LIBRARY=$PWD/build/libvibeqc.so python -m pytest tests/python -q
```

The existing CPU/Python CI jobs discover the new tests automatically. CUDA CI
continues its separate compilation job. Its `VIBEQC_CUDA_FAST_COMPILE=ON` setting
is a compilation smoke check only; never use that build for resource/performance
acceptance. `validation_gate.py run` wraps these existing commands and archives
their outputs instead of replacing ctest, pytest, autotune, or the f-shell runner:

```bash
python benchmarks/validation_gate.py run --tier cuda-compile --subject release-build \
  --output /tmp/compile.json -- cmake --build --preset cuda-release-sm120 --parallel 2

python benchmarks/validation_gate.py run --tier gpu-numerical --subject f-matrix \
  --unavailable 'GPU job has not been allocated' --output /tmp/gpu-not-run.json

VIBEQC_LIBRARY=$PWD/build/libvibeqc.so python benchmarks/validation_gate.py hf \
  --case h2 --device cpu --repeats 5 --finite-difference --output /tmp/hf-cpu.json
```

For actual GPU work on this machine, preserve Slurm's assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env VIBEQC_LIBRARY=$PWD/build/cuda-release-sm120/libvibeqc.so \
  python benchmarks/validation_gate.py hf --case h2 --device cuda --repeats 5 \
  --output /tmp/hf-cuda.json
```

Use `run --tier gpu-numerical` inside the same allocation to wrap an existing
autotune command with `--local`, or let autotune request its own finite Slurm
allocation with `--partition main --gres gpu:5090:1`. Use `--attach PATH` to
register its original versioned output (including #135/#136 evidence) by hash.
The existing large endpoint runner is wrapped identically with `--tier endpoint`.
No full f-shell matrix is added to ordinary CI. Missing hardware/tools become
`not-run` with reasons. An unallocated GPU invocation also records `not-run`.
A wrapped numerical process exiting zero is recorded as command success only:
it might have skipped tests, so inspect/attach its actual numerical evidence.

## Shared results and performance protocol

`tools/vibeqc_validation/schema.py` defines the `vibeqc.validation` version-1
envelope. `new_evidence`, `block_error`, `attach_artifact`, and `write_evidence`
are the registration API for downstream tasks. Every record includes revision,
equation/IR/source/schedule identities, device/toolchain, actual selected backend,
settings, raw timings, allocated/peak bytes, compilation cost, residuals,
per-block errors, stages, and failure/not-run reasons. Unknown versions, nonfinite
numbers, missing-GPU success, and performance passes lacking provenance are
rejected. Existing autotune/endpoint artifacts retain their original schemas.

`measure_interleaved` reuses the existing AOT gate's ABBA order and timing
summary. It requires at least five measurements **per side**, synchronizes before
and after each sample, and hashes identical inputs. Optional `prepare` runs
outside timing. Separate cold native plan construction, retained unchanged
geometry, changed geometry, energy-only, and energy-plus-force workloads. State
clearing, process startup, imports and JIT compilation must be explicitly scoped
by the caller; the HF example's cold interval includes native plan creation and
destruction, not Python process startup or imports.

Kernel comparisons require an explicit fixed density/amplitude hash. Solver
comparisons require every iteration's energy and residual, keyed by timed sample,
plus independently evaluated final residuals and their numerical gate. Raw
samples are never discarded. Runtime assessment reports median, min/max, raw
samples and relative median absolute deviation (MAD). Improvement must exceed
2% and twice the sum of the relative MADs. Differences below this floor remain
inconclusive; this descriptive rule is not a statistical confidence interval.

Promotion also requires measured memory and compilation cost within explicit
`settings.promotion_limits.peak_bytes` and `.compile_seconds` budgets. A runtime
win alone cannot pass. Compilation, independent numerical and endpoint stages
must already pass; no automatic manifest edits are performed.
GPU promotion additionally requires explicit `settings.fast_compile=false`;
the HF example records the selected library hash and adjacent CMake build
settings when available, including any compilation-smoke-only flags.

The HF example compares identical native implementations as an A/B protocol
control. It validates all samples, freezes post-cold warm-start density, and
checks changed geometry against native one-shot execution (explicitly a replay
consistency check, not an independent displaced libcint reference). The public
HF API always computes forces and exposes neither full iteration histories,
final orbitals/density, nor allocator peaks. Those fields and energy-only work
are honestly unavailable. The example therefore demonstrates numerical and
endpoint evidence plus timing collection while **prohibiting performance or
production promotion**. It is not a speedup claim or CC implementation.
