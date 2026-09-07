# User-local CUDA autotuning

Untuned GPUs use `portable_cuda` and the generic CUDA implementation immediately.
Autotuning is optional and never runs during installation. A local profile
contains a validated native library and its evidence, so later calculations can
use the generated kernels without rebuilding them.

## Run a workload-first search

Install the optional reference dependency with `pip install -e '.[autotune]'`.
Tuning requires NVCC, PTXAS, CUOBJDump, CMake, Ninja, a C++ compiler, and the
matching VibeQC source checkout. The baseline native library must be a Release
build with `VIBEQC_CUDA_FAST_COMPILE=OFF`. Rebuild it after changing native code,
generator code, or Python runtime policy; the command checks its source identity
before starting a search. A wheel installation can pass `--source-dir` pointing
to the matching source checkout. CPU use and profile management do not require
CUDA tools or a GPU allocation.

XYZ coordinates default to Angstrom and are converted to Bohr. Use `--units
bohr` for a Bohr XYZ file. Charge, multiplicity, RHF/UHF method, Cartesian or
spherical representation, and batch size are explicit options. The command uses
logical CUDA device 0 within the allocated visibility.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=02:00:00 env VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
  vibeqc autotune --quick water.xyz --basis def2-svp

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=04:00:00 env VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
  vibeqc autotune --full water.xyz --basis def2-tzvp --batch 4 \
  --budget-seconds 14000 --export local-profile.zip
```

The same commands are available through `python -m vibeqc`. On systems without
Slurm, run inside the allocation provided by the local scheduler. The tuner and
its child processes preserve `CUDA_VISIBLE_DEVICES`; they do not request a
second allocation. Set `CUDA_PATH` or `--nvcc` to select a toolkit.

Quick mode profiles the actual final-density shell tasks and ranks classes by
measured active primitive work. It targets 97% coverage, at most eight classes,
and at most four schedules per consumer, plus an official comparison baseline
when required. Full mode targets all measured classes and up to sixteen
schedules per consumer. Neither mode compiles classes with zero measured work.
Use `--coverage`, `--max-classes`, `--max-candidates`, `--compile-jobs`, and
`--budget-seconds` to bound the search. Coverage and the remaining uncovered
fraction are recorded even when a limit ends the search.

Very small workloads can use cached ERIs instead of direct shell-class kernels.
When that execution path supplies no direct-work counters, tuning records the
reason, compiles no classes, and preserves the current configuration.

Exact official consumers are retained by default. `--retune` explicitly includes
them in the search. `--portable-baseline` starts from the generic CUDA path even
on an officially tuned device, which is useful for validating the untuned-device
workflow on available hardware. It does not mislabel the actual GPU architecture.

Release compilation of the native fallback can dominate a first local build.
Subsequent proposals reuse the same build directory and existing compiler cache.
The default two-hour tuning budget is a limit, not an expected runtime. A budget
exhaustion, interrupt, or failed gate leaves the previously installed profile
active. Candidate logs and a report remain in the printed run directory.

## Acceptance and failure handling

The workflow reuses the developer schedule tuner and its target, compilation,
resource, and repeated on-device timing gates. It additionally runs the shared
libcint fixtures against all four direct/persistent RHF/UHF wrappers of each
proposed consumer. Fixtures cover asymmetric and coincident centers, Cartesian
and spherical conventions, pair reversals, and shell/atom permutations. Forces
must also satisfy translation invariance. Numerical gates use the established
combined absolute/relative floors rather than raw relative errors near zero.
After building each candidate library, the same all-spin gate runs against the
exact native object, preserving its actual source hash, resources, and launch
behavior before endpoint promotion.

Each proposed consumer is built into a candidate native library and compared
with the currently accepted build in fresh processes. Balanced ABBA ordering
collects at least four repeats per side (six by default). Process startup, cold
SCF, and warmup are excluded; the measured replay includes complete energy and
analytic forces from a frozen converged density. All samples are retained and
SCF iteration counts must match. Promotion requires at least 1.02x median
endpoint speedup and a positive 95% bootstrap lower speedup bound above 1.0,
alongside energy, force, and translation parity. A faster isolated kernel alone
cannot enable a local profile. When a compatible local profile is already active,
the final proposal must also beat that incumbent before replacing it.

Fock and force proposals are gated independently. An older official row with an
implicit Fock mapping cannot accept a force-only replacement until an explicit
Fock schedule has been accepted, because its value mapping otherwise changes
with the force schedule. This rejection is recorded. A newly tuned Fock mapping
does not enable an unvalidated mixed-precision companion; those calculations
retain the generic mixed-Fock route.

No winner is a normal outcome. Failed compilation, exhausted budgets, numerical
errors, noisy timings, and slower endpoints preserve the working configuration.
Only complete accepted bundles are activated through an atomic index update.
Unfinished staging directories are never read by runtime selection.

## Reuse and diagnostics

Profiles live under `$XDG_CACHE_HOME/vibeqc/profiles`, defaulting to
`~/.cache/vibeqc/profiles`. `VIBEQC_PROFILE_CACHE` can select another directory.
Python CUDA calculators use this precedence:

1. A compatible explicit `VIBEQC_PROFILE=/path/to/accepted/bundle`.
2. A compatible validated local profile selected by its exact identity.
3. The baseline library's exact official profile or portable CUDA fallback.

`VIBEQC_PROFILE=off` selects the baseline library directly. `VIBEQC_LIBRARY`
locates that baseline and does not disable local profile reuse. CPU calculators
do not consult CUDA profiles or probe hardware.

The identity includes GPU name/compute capability and relevant hardware limits,
CUDA runtime/toolkit/driver versions, host architecture/libc, native ABI, source
and codegen identity, schema, and FP64 consumer policy. Each tuned kernel has
source and schedule hashes, all promotion stages, and matching numerical and
endpoint evidence. Binary, manifest, and evidence files are hashed. An available
NVCC/PTXAS must match the recorded compiler identities; reuse of an immutable
accepted binary does not require those compilers to be installed.

```python
from vibeqc import Calculator

calculator = Calculator(method="rhf", basis="def2-svp", device="cuda")
print(calculator.profile_diagnostics)
```

Diagnostics report `official`, `local`, or `portable`, the selected identity,
tuned consumers, and incompatible-cache rejection reasons. `vibeqc profile
diagnose` probes the allocated GPU and prints the same selection. Static
management commands require no GPU:

```bash
vibeqc autotune --show-profile
vibeqc profile show
vibeqc autotune --clear-profile
```

Clearing profiles deactivates them while preserving immutable binaries already
loaded by live processes. Those processes retain their selected library.

## Homogeneous cluster reuse

Export/import carries the binary, manifest, and evidence together:

```bash
vibeqc profile export /path/to/accepted/bundle local-profile.zip
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:02:00 vibeqc profile install local-profile.zip
```

Installation checks the same hardware, source, ABI, toolchain, and artifact
contracts as runtime selection. It probes the imported binary as well as its
metadata before activation. UUID and PCI address are excluded so identical GPUs
on different nodes can share a profile. A schedule JSON by itself cannot be
installed because it contains no executable kernels or matching evidence.

Local results describe the user's measured workload. Official repository
profiles remain the source for published benchmark claims. Native C/C++ clients
can load the exported `libvibeqc.so` explicitly; automatic cache selection is
provided by the Python CUDA calculator interface.
