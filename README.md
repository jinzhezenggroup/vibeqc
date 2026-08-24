# QCE

QCE is the working name for a native electronic-structure engine. The public
naming prefix is intentionally provisional until the final project name is
chosen.

The current vertical prototype implements RHF and UHF energies and analytic
nuclear gradients for all-electron contracted **s through f** Gaussian bases.
The independent CPU oracle accepts CCA Cartesian or real spherical AOs; the
device-resident CUDA RHF/UHF path currently covers the validated Cartesian
s-p-d-f scope. It includes a versioned C ABI, a C++ core, Python bindings
through `ctypes`, and native ragged fleet batches. CUDA spherical execution,
ROHF, density fitting, and generated angular-momentum-specialized quartet
kernels remain explicit roadmap items rather than silently falling back to an
unvalidated implementation.

## Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
ctest --test-dir build --output-on-failure
```

For a CPU-only reference build, pass `-DQCE_ENABLE_CUDA=OFF`.

## Python smoke test

```bash
PYTHONPATH=python QCE_LIBRARY=build/libqce.so python examples/h2.py
```

On the target cluster, request GPU execution through Slurm rather than opening
a device directly, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  env PYTHONPATH=python QCE_LIBRARY=$PWD/build/libqce.so.0.1.0 \
  python examples/h2.py
```

Coordinates are expressed in Bohr and energies in Hartree. `forces` are
`-dE/dR` in Hartree/Bohr.

## Ragged batch and warm starts

```python
from qce import Calculator

calc = Calculator(method="rhf", basis="sto-3g")
systems = [
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    [("He", (0, 0, 0))],
]

with calc.prepare_batch(systems, warm_start=True) as batch:
    first = batch.execute(strict=True)
    second = batch.execute(strict=True)  # reuses one density per topology

print(first.energies)
print([item.forces.shape for item in first.items])  # [(2, 3), (1, 3)]
```

The native plan keeps every system ragged, buckets compatible workloads,
returns results in input order, and records a separate status for every item.
Bad coordinates or a nonconverged SCF item do not abort valid neighbors.

## Scientific scope and limitations

- RHF requires an even electron count and multiplicity 1. UHF derives
  integral alpha/beta occupations from electron count and multiplicity.
- Contracted `s`, `p`, `d`, and `f` shells are executable. The ABI carries an
  angular-momentum field and rejects `g` and higher shells. The CPU reference
  supports CCA Cartesian and PySCF/libcint-ordered real spherical functions;
  CUDA currently requires Cartesian AOs and reports spherical execution as
  unavailable instead of falling back to the host.
- The Python package bundles data-only STO-3G, def2-SVP, and def2-TZVP packs
  for H-Ar, generated from the pinned Basis Set Exchange reference. Named basis
  calculations remain Cartesian by default for compatibility; pass
  `basis_representation="spherical"` on the CPU reference for standard pure
  def2 semantics.
- No ECP, symmetry, finite-temperature occupations, DFT, or post-HF methods.
- The analytic gradient differentiates integral formulas analytically and
  assembles variational RHF/UHF gradients with their Pulay terms. It does not
  backpropagate through SCF iterations.
- CUDA 12.9 and `sm_120` are supported. With `device="cuda"`, integral values,
  SCF matrices, batched eigensolves, convergence state, energy, and analytic
  forces execute on the GPU and report `cuda`; the CPU path is used only when
  explicitly requested.
- The current CUDA SCF includes device DIIS and a device-tail-launched CUDA
  Graph loop. Fixed-topology plans retain their arena, Graph, stream, and
  eigensolver state; small AO buckets use a specialized device Jacobi solver.
  UHF stores alpha/beta matrices adjacently per system, solves both spin Fock
  matrices on device, and applies one combined DIIS residual per physical
  system.
- Contracted primitives are stored once per physical shell. Ragged
  system-to-shell, shell-to-AO, shell-to-primitive, and AO-to-shell metadata
  remain in the topology arena; Cartesian component normalization is separate,
  so d/f contractions are not duplicated for every AO component.
- AO buckets through 16 functions retain ERIs in their persistent arena for
  fleet replay. Larger buckets use O(N^2)-memory GPU Schwarz bounds and fused
  screened direct J/K inside the SCF Graph. Direct s/p/d/f topologies enumerate
  only the eightfold-symmetry-unique shell and AO quartets, evaluate each ERI
  once, and atomically scatter its RHF/UHF J/K contributions. A fixed,
  topology-derived AO-quartet tile multiplicity exposes large d/f component
  groups across multiple blocks without storing an O(N^4) task array. The
  two-electron force contracts the same unique quartets and differentiates only
  the four participating shell centers. Every bucket uploads one
  fixed-topology packed AO-pair table; one-electron integrals and their force
  terms use only the triangle, and direct buckets reuse it for Schwarz bounds
  and the direct scheduler. Ragged canonical shell-pair and shell-quartet
  topology plus shell-level Schwarz maxima are resident. Geometry-dependent
  shell-quartet compaction is implemented fully on device and awaits allocated
  GPU regression/performance validation. Generated quartet specialization,
  CUDA spherical consumers, and DF J/K are still required before making
  production performance claims.

UHF uses the same memory policy: small buckets retain ERIs, while larger
buckets build spin-resolved Fock matrices directly from O(N^2) Schwarz data.
Coulomb consumes the total density and exchange consumes only the matching
spin density. Its analytic two-electron force applies the identical screening
decision used by the energy path.

## GPU4PySCF comparison boundary

The reproducible same-process microbenchmark is
`benchmarks/compare_gpu4pyscf.py`. It deliberately includes three different
artificial regimes plus a Cartesian water/def2-SVP case. On an RTX 5090,
representative warm energy+gradient medians are about
2.44 ms for QCE versus 63.0 ms for GPU4PySCF in the validated 8-AO s/p case,
where native plan replay avoids much of the Python/CuPy dispatch overhead. In
the artificial 18-AO s/d/f direct-J/K case, a recent allocated run measured
about 28.8 ms for QCE versus 658 ms for GPU4PySCF. In the 21-AO He3 s/d case,
the symmetry-reduced tiled-quartet path measured about 37.9 ms for QCE versus
330 ms for GPU4PySCF in the same allocation. Both engines, especially
GPU4PySCF, can show substantial clock/dispatch variation. Run these
explicitly with `--case sp8`, `--case sdf18-direct`, and
`--case he3-sd21-direct`; use `--case water-def2-svp` for the named-basis case.

The same harness also includes `h2plus-uhf2` and `heh-sdf18-uhf`. On the same
RTX 5090, H2+ warm UHF energy+gradient replay measured about 7.0 ms for QCE;
GPU4PySCF showed substantially higher and more variable dispatch time. For the
18-AO direct-UHF case QCE was stable near 156--163 ms, while GPU4PySCF ranged
from roughly 150 ms minimum to over 700 ms median across runs. These artificial
cases verify execution regimes and startup behavior, not production UHF
leadership.

None of these cases establishes realistic basis-set leadership. GPU4PySCF
already has mature higher-angular-momentum Rys kernels, screening, spherical
basis support, and highly optimized shell-sorted direct J/K. QCE now switches
larger AO buckets to an O(N^2)-memory screened direct path, but that path still
lacks GPU4PySCF's generated Rys kernels, mature AO-level active-task
compaction, CUDA spherical consumers, and standard spherical named-def2 GPU
execution.
No broad performance claim is valid until both engines run the same realistic
molecules, basis, precision, convergence, and energy/gradient workload on the
same allocated GPU.

`benchmarks/compare_gpu4pyscf_batch.py` measures homogeneous fixed-topology
batches at configurable sizes. QCE submits one native bucket; because
GPU4PySCF currently exposes a single-molecule SCF API, the harness retains one
GPU object and warm density per system and executes them sequentially inside
the same synchronized batch boundary. Reports must state this interface
difference rather than presenting the result as two equivalent batch APIs.

Pass `--output path/to/result.json` to any script in `benchmarks/` to retain
raw timing samples together with the Git commit and dirty state, Python and
package versions, runtime library path, benchmark parameters, and (for the
GPU4PySCF comparison) CUDA device properties. Performance claims should cite
one of these artifacts rather than only a copied median.
