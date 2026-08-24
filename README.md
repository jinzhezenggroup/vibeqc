# QCE

QCE is the working name for a native electronic-structure engine. The public
naming prefix is intentionally provisional until the final project name is
chosen.

The current vertical prototype implements RHF and UHF energies and analytic
nuclear gradients for all-electron contracted **s through f** Gaussian bases.
The independent CPU oracle and device-resident CUDA RHF/UHF path accept CCA
Cartesian or PySCF/libcint-ordered real spherical AOs through the validated
s-p-d-f scope. The project includes a versioned C ABI, a C++ core, Python
bindings through `ctypes`, and native ragged fleet batches. ROHF, density
fitting, and component-unrolled/Rys quartet kernels remain explicit roadmap
items rather than silently falling back to an unvalidated implementation.

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
  angular-momentum field and rejects `g` and higher shells. Both backends
  support CCA Cartesian and PySCF/libcint-ordered real spherical functions.
  CUDA keeps public spherical matrices on device, maps direct-J/K densities to
  normalized Cartesian source AOs, and maps the resulting Fock matrices back
  without a host fallback.
- The Python package bundles data-only STO-3G, def2-SVP, and def2-TZVP packs
  for H-Ar, generated from the pinned Basis Set Exchange reference. Named basis
  calculations remain Cartesian by default for compatibility; pass
  `basis_representation="spherical"` for standard pure def2 semantics on
  either backend.
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
  eigensolver state. AO matrices through 16 use a low-overhead serial device
  Jacobi solver, sizes 17--32 use batched cuSOLVER, and larger matrices use a
  cooperative Graph-native Jacobi kernel. The 48-AO Cartesian
  water/def2-TZVP path is numerically validated against PySCF.
  For 33--256 AOs, that Graph-native path uses parallel cyclic sweeps with
  disjoint round-robin rotations instead of rescanning the full matrix before
  every pivot; larger matrices retain the unbounded maximum-pivot fallback.
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
  the participating shell centers; ERI translational invariance evaluates only
  `N_unique-1` centers and reconstructs the final derivative from their
  negative sum. A force-only three-component forward scalar carries x/y/z
  derivatives through one recurrence per evaluated center instead of
  repeating the complete ERI value path for every axis. Every bucket uploads
  one fixed-topology packed AO-pair table; one-electron integrals and their
  force terms use only the triangle, and direct buckets reuse it for Schwarz
  bounds and the direct scheduler. Ragged canonical shell-pair and shell-quartet
  topology plus shell-level Schwarz maxima are resident. Geometry-dependent
  shell-quartet compaction is implemented fully on device and has passed
  allocated RTX 5090 energy/force and throughput validation for the exact
  `sdf18-direct` and Cartesian water/def2-SVP batch workloads. Direct consumers
  dispatch 55 symmetry-reduced s/p/d/f shell classes inside 13 angular-order
  Graph nodes. Compact 256-quartet descriptors expand virtually into eight
  one-warp subtiles, exposing independent blocks despite the recurrence
  kernels' high register footprint without multiplying arena metadata.
  Consumers canonicalize ERI arguments by pair symmetry and size Hermite
  workspaces from the exact shell pairs; canonical `(p s | s s)` uses a
  closed two-term first-order contraction instead of generic coefficient
  arrays. The total-order-2 `(d s | s s)`, `(p p | s s)`, and
  `(p s | p s)` classes use at most four sparse Hermite terms per pair and
  generate their Coulomb derivatives directly from `F0`/`F1`/`F2`. This
  reduces their per-thread Fock/force stacks from 1,072/3,184 bytes to
  464/888 bytes. Total-order-3 `(f s | s s)`, `(d p | s s)`,
  `(d s | p s)`, and `(p p | p s)` quartets use exact 1/2/4/8-term quantum
  products plus same-axis Gaussian contractions and direct `F0`--`F3`
  Coulomb derivatives. Their Fock/force stacks fall from 1,568/5,200 bytes to
  512/1,256 bytes. Total-order-4 `(f p | s s)`, `(d d | s s)`,
  `(f s | p s)`, `(d p | p s)`, `(d s | d s)`, `(d s | p p)`, and
  `(p p | p p)` quartets extend the same generated expansion with all
  same-axis single and disjoint double Wick contractions plus direct
  `F0`--`F4` Coulomb derivatives. Their Fock/force stacks fall from
  2,488/8,608 bytes to 752/2,024 bytes. Real-spherical direct buckets apply
  `C^T D C` before those Cartesian-source quartets and `C F C^T` afterwards,
  eliminating repeated sparse term products from Fock and force recurrences.
  Remaining component-unrolled/Rys kernels, broader named-basis gates, and
  DF J/K are still required before making broad production performance claims.

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
The matching standard pure-basis workload is available as
`--case water-def2-svp-spherical`; the harness sets QCE to real spherical AOs
and PySCF/GPU4PySCF to `cart=False` from the same case metadata.
Larger primitive and angular workloads are available as
`--case water-def2-tzvp` and `--case water-def2-tzvp-spherical`; these are
profiling and regression targets with clean allocated artifacts in both AO
representations.

The same harness also includes `h2plus-uhf2` and `heh-sdf18-uhf`. On the same
RTX 5090, H2+ warm UHF energy+gradient replay measured about 7.0 ms for QCE;
GPU4PySCF showed substantially higher and more variable dispatch time. For the
18-AO direct-UHF case QCE was stable near 156--163 ms, while GPU4PySCF ranged
from roughly 150 ms minimum to over 700 ms median across runs. These artificial
cases verify execution regimes and startup behavior, not production UHF
leadership.

Named-basis UHF is available in matching Cartesian and standard pure forms as
`--case oh-def2-svp-uhf` and `--case oh-def2-svp-spherical-uhf`. Both use the
same OH doublet geometry and spin occupations; only the AO representation
changes.

The single-system cases above do not establish broad realistic basis-set
leadership. GPU4PySCF already has mature higher-angular-momentum Rys kernels,
screening, spherical basis support, and highly optimized shell-sorted direct
J/K. QCE now switches larger AO buckets to an O(N^2)-memory screened direct
path with device-compacted angular-order partitions, but it still lacks
component-unrolled or Rys shell kernels and DF J/K. Standard spherical
named-def2 GPU execution is numerically validated, but its performance has not
yet been published. Broad claims require more molecules, bases, batch shapes,
and convergence regimes on the same allocated GPU.

`benchmarks/compare_gpu4pyscf_batch.py` measures homogeneous fixed-topology
batches at configurable sizes. QCE submits one native bucket; because
GPU4PySCF currently exposes a single-molecule SCF API, the harness retains one
GPU object and warm density per system and executes them sequentially inside
the same synchronized batch boundary. Reports must state this interface
difference rather than presenting the result as two equivalent batch APIs.

Seven clean artifacts from the exact shell-class implementation establish
performance leadership for their exact homogeneous workloads on one RTX 5090.
`sdf18-direct` at batch 16
measured 67.709 ms for QCE versus 2499.369 ms for sequential GPU4PySCF
(36.91x), with maximum energy/force differences of `6.35e-14`/`4.87e-14`.
Cartesian water/def2-SVP at batch 8 measured 1202.903 ms versus 7300.320 ms
(6.07x), with differences of `1.72e-12`/`5.29e-13`. Standard real-spherical
water/def2-SVP at batch 8 measured 2313.694 ms versus 7331.882 ms (3.17x),
with differences of `1.78e-12`/`4.24e-13`. Standard real-spherical
water/def2-TZVP at batch 4 measured 911.532 ms versus 12099.769 ms (13.27x),
with differences of `1.01e-12`/`8.81e-13`; its 48-AO Cartesian counterpart
measured 2069.985 ms versus 12084.960 ms (5.84x), with differences of
`1.21e-12`/`9.24e-13`. Open-shell OH/def2-SVP UHF at
batch 8 measured 571.419 ms versus 7244.466 ms (12.68x), with differences of
`1.24e-12`/`2.11e-13`. Its standard real-spherical counterpart measured
1369.543 ms versus 7304.363 ms (5.33x), with differences of
`1.09e-12`/`4.13e-11`. See the raw
[RTX 5090 benchmark artifacts](benchmarks/results/README.md) for samples,
versions, device metadata, and reproduction commands. These results do not
imply an equivalent native GPU4PySCF batch API or generalize beyond the named
workloads.

Pass `--output path/to/result.json` to any script in `benchmarks/` to retain
raw timing samples together with the Git commit and dirty state, Python and
package versions, runtime library path, benchmark parameters, and (for the
GPU4PySCF comparison) CUDA device properties. Performance claims should cite
one of these artifacts rather than only a copied median.

Allocated batch comparisons can also enforce regression gates with
`--minimum-speedup`, `--maximum-energy-error`, and `--maximum-force-error`.
The harness always writes the requested JSON first, including the thresholds
and any failure messages, then exits with status 2 when a gate fails. This
keeps failed measurements inspectable while making scheduler/CI jobs fail
closed.

The formal realistic-size matrix is `benchmarks/real_molecule_gate.py`:
real-spherical def2-SVP WATER27 water tetramer at exactly 96 AOs and S4 water
octamer at exactly 192 AOs, each at batch 1 and batch 4. All four points enforce
explicit accuracy gates: `3e-11 Eh`/`3e-11 Eh/bohr` at 96 AOs and
`1e-10 Eh`/`5e-10 Eh/bohr` at 192 AOs. They use
`1e-12 Eh`/`1e-10` SCF energy/density tolerances and a matching `1e-14`
direct-J/K screening threshold; both engines receive up to 100 SCF iterations.
GPU4PySCF uses separate `1e-9` and `1e-8` orbital-gradient thresholds for the
96- and 192-AO points, respectively, avoiding its large-system numerical-noise
floor. The 96-AO points also require at least parity with GPU4PySCF now; the
192-AO speed gate activates after complete DF J/K. Run
it inside one allocated GPU and place output outside the measured clean
worktree:

```bash
python benchmarks/real_molecule_gate.py \
  --repeats 3 --output-directory /tmp/qce-real-molecule-gate
```
