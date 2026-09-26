# Generated f-shell validation

Issue [#135](https://github.com/jinzhezenggroup/vibeqc/issues/135) separates
representation, release compilation, numerical validity, endpoint acceptance,
and production selection for all 34 canonical f-containing classes.
`tools/validate_f_shells.py` catalogs the common recurrence/schedule capabilities,
exact registry indices, all consumers, source/IR hashes, generator ABI, and
manifest selection. A manifest entry does not establish acceptance.

The legacy `fpps` force selection is **provisional** until it satisfies the same
acceptance as this matrix. Earlier FSSS/FPPS compile smoke and the f-free FSPS
tuning workload do not establish full f-shell correctness or performance.
Selection is reported explicitly; validation does not silently alter dispatch
or mechanically promote 34 classes.

The provenance audit traces FPPS selection to commit
`c0683c5b0a66b6330d16117ab8a4dd812956843b`. Its `docs/shell_codegen.md` records
water/def2-TZVP profiling, a 2.27x isolated speedup, 15-sample endpoint speedups
of 1.0022x/1.0054x/1.0060x at batches 1/4/8, and a maximum force difference of
7.17e-13 hartree/bohr. This historical reason for selection is preserved; it
does not supply the missing current-source, independent, all-consumer matrix.

## CI and manual tiers

Every Python test run checks all 34 classes, complete component ordering,
three independent derivative centers plus translation recovery, all eight
RHF/UHF Fock/force direct/persistent wrappers, and byte-identical regeneration.
A host C++ regression executes the emitted f/f value-pair routine against
independent Gaussian moments. It detects the missing three-pair Wick
contractions that the GPU matrix exposed in the original ten `ff*` Fock
classes. Value and gradient consumers now share the complete matching logic.
The highest FFFF force order also exposed a 32-bit intermediate overflow in
the Wick multiplicity ratio: 13! exceeds unsigned even though the final
coefficient is small. Its intermediate uses 64 bits and an exact-integer host
regression checks all multiplicities through order 13.

```bash
python tools/validate_f_shells.py --tier source --output build/f-source.json
python -m pytest tests/python/test_f_shell_validation.py -q
```

CUDA PR CI compiles only `fsss,fsps,fpps,fdfd,ffff`. The separate release resource
step always uses `-O3`; the production fast-compile CI step cannot supply resource
acceptance. Per-class caches include source, ABI, target, schedule, flags, and
NVCC/PTXAS/CUOBJDump versions, with verified object/log hashes on cache hits.
Reports retain registers, stack, spills, local/shared memory, source/object/cubin
sizes, compile times, missing wrappers, and rejected resources. CUDA 12.9 sm_120
cubins report 1024 extra shared bytes beyond PTXAS/runtime user allocation; both
measurements are retained. Spilling is distinct from launch legality.

```bash
python tools/validate_f_shells.py --tier compile --smoke \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --compile-jobs 2 --compile-timeout 900 --output build/f-smoke.json
```

The full matrix is a manual release gate. Compilation needs no GPU. The
numerical tier internally submits a finite `main`, `gpu:5090:1` Slurm job per
class, preserving scheduler visibility and checking actual architecture,
launch attributes, and positive occupancy for every wrapper. Install NumPy
and PySCF in the invoking Python environment. A native CUDA host driver links
against the exact cached generated object; it copies no integral implementation.

```bash
OMP_NUM_THREADS=1 python tools/validate_f_shells.py --tier numerical \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --compile-jobs 2 --compile-timeout 900 \
  --slurm-time 00:20:00 --runtime-timeout 1500 \
  --output build/f-full.json --archive build/f-full-evidence
```

Fourteen fixtures per class use asymmetric centers, unequal primitive
contractions including a negative coefficient, and deterministic nonzero
RHF/UHF densities. They cover Cartesian/spherical conventions, both primitive
pair reversal bits separately and together, coincident physical atoms, atom
relabeling, and all eight symmetry-equivalent external shell layouts with
canonical task offsets. Libcint independently supplies values and all-center
derivatives; explicit ERI orbit loops supply the contraction oracle. Spherical
expected values come directly from spherical libcint, while Cartesian projected
densities and output Fock matrices exercise the generated ABI.

All analytic blocks use `atol=rtol=2e-10`, with force-sum translation checks.
FSSS/FSPS/FPPS additionally displace all 12 coordinates at both signs of three
fixed steps (0.01, 0.003, 0.001 bohr), comparing the fixed-density generated Fock
energy derivative at `atol=rtol=1e-6`. Reports retain every error block, failed
state, mathematical fixture hash, source/object/driver identity, toolchain,
actual device, and Slurm command. The archive index hashes complete per-class
JSON files; binary caches and raw profiler traces remain outside Git.

## Real endpoints and promotion

`benchmarks/f_shell_endpoints.py` supplies Cartesian water/def2-TZVP (48 AOs,
one loaded f shell), and a hydrogen-bonded two-monomer fragment of the documented
WATER27 tetramer in spherical def2-TZVP (86 AOs, two loaded f shells). It inspects
loaded native and PySCF shells and contraction sizes; a basis name cannot satisfy
the gate. Run both batch 1 and batch 4. The complete tetramer (172 AOs, four f
shells) supplies an additional batch-1 stress endpoint. Its batch-2 and batch-4
failures are retained: 4,528,320 logical tiles per system at 192 bytes per task
cross the runtime's 1 GiB descriptor threshold when batched, invoking a bounded
path without complete f-containing Fock/force coverage. This is an unsupported
execution regime, not an accepted production benchmark or a reason to relax
tolerances.

The FPPS force candidate and generic baseline share the remaining production
selection and identical Fock settings. The existing capacity-prime/frozen-dm0
protocol executes six ABBA repeats with strict SCF and screening settings.
Different SCF iteration branches invalidate performance acceptance. Independent
CPU PySCF energies and analytic forces are checked for every batch geometry,
outside the timing region, at 1e-9 hartree and 1e-7 hartree/bohr. The explicit
endpoint non-regression budget is 2%; a slower candidate is a rejected
promotion, not a numerical failure.

Use a release class-mode library configured with `VIBEQC_AOT_UNIT_MODE=class`,
`VIBEQC_CUDA_FAST_COMPILE=OFF`, and the actual `120-real` target. Record its
CMake settings, library/source hashes, and build duration with the evidence.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 env OMP_NUM_THREADS=1 PYTHONPATH=python \
  VIBEQC_LIBRARY="$PWD/build/f-shell-endpoint/libvibeqc.so" \
  python benchmarks/f_shell_endpoints.py --case water-def2-tzvp \
  --batch 1 --repeats 6 --output build/f-water-b1.json
```

Reports contain pre-screening canonical primitive work and native active
final-density counters. Capture actual device time in a separate Slurm run
with Nsight Systems, using the same Python/library environment:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=build/f-water-profile python benchmarks/f_shell_endpoints.py \
  --case water-def2-tzvp --batch 1 --profile --output build/f-water-profile.json
nsys export --type=sqlite --output=build/f-water-profile.sqlite build/f-water-profile.nsys-rep
python benchmarks/f_shell_device_time.py build/f-water-profile.sqlite \
  --output build/f-water-device-time.json
```

The trace contains one warm candidate replay, excluding initialization and the
reference. Measured kernel durations give exact generated-class times. Generic
angular orders 9–12 necessarily contain f; orders 3–8 mix f and s/p/d work.
Their unresolved time produces explicit lower/upper f-time fractions for
Fock and force. Primitive counts are never substituted for measured device
time, and profiled timings are not endpoint speed claims.

Promotion combines the same source/schedule's isolated numerics, resources,
actual class work/time, unprofiled endpoint results, and compile/binary costs.
Mixed generic timing intervals cannot justify a precise individual-class
speed claim. Slower/spilling classes may remain unselected.
