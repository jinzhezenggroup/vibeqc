# Cross-basis HF initialization

VibeQC can solve a source HF calculation, project its occupied density into a
different AO basis, and converge the actual target Hamiltonian. Projection is
explicit and opt-in. It does not change the target basis, electron count, spin,
integrals or convergence criteria, and a converged source is never reported as
a converged target.

```python
from vibeqc import Calculator, projected_singlepoint

atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]  # Bohr
source = Calculator(basis="sto-3g")
target = Calculator(basis="def2-svp")
run = projected_singlepoint(target, source, atoms)
print(run.target.energy, run.target.forces)
print(run.diagnostics["total_seconds"])
```

`ProgressiveResult` owns the source/target result records, timing/projection
diagnostics and an immutable copy of the final target density. Source solve,
projection, target setup, target execution and full elapsed time are recorded
separately. The source endpoint currently also computes forces; that cost is
included even though projection consumes its retained density.

## Metric contract

The native CPU `cross_overlap(target, source, atoms)` adapter evaluates the
rectangular matrix `S_ts[mu_target, nu_source]`. Each side independently uses its
normalized contracted Cartesian or real-spherical s/p/d/f basis. The evaluator
shares the independent Gaussian overlap recurrence and sparse AO expansions
with the existing integral oracle. It writes only the requested rectangle,
without a combined-basis square matrix, ERIs or nuclear derivative tensors.
Raw diagnostics may provide independent `source_atoms`; prepared initialization
requires the same current geometry and ordered nuclei.

For metric-orthonormal source occupied columns `C_s`, projection computes

```
P = S_tt^+ S_ts C_s
G = P^T S_tt P
C_t = P G^(-1/2)
D_t = occupation * C_t C_t^T
```

`project_occupied` accepts this matrix contract directly. `occupation=2`
produces an RHF density. UHF projects alpha and beta independently with
`occupation=1`, including empty spin channels. Arbitrary real occupied rotations
and phases leave the projected density unchanged. Matrix-only entry points
validate the numerical state; the prepared interface additionally validates
the nuclei, core treatment, electron count, spin and model provenance.

`project_density` reconstructs occupied orbitals by diagonalizing
`S_ss^(1/2) D_s S_ss^(1/2)` in the retained source metric space. Its eigenvalues
must match the declared 0/2 or 0/1 occupations. Fractional occupations, wrong
electron counts, non-Hermitian densities and unsupported discarded metric
components are rejected. No virtual orbitals are invented or transported, and
correlated amplitudes are outside this interface.

`ProjectionPolicy` defaults to a relative spectral threshold of `1e-10`, numerical
validation tolerance `1e-7`, maximum projection residual `0.5` and 4096 AOs per
matrix. Eigenvalues at or below the relative threshold times the largest value
are discarded reproducibly. The projection residual is
`sqrt(max(0, 1-lambda_min(G)))`: the worst occupied direction's lost norm before
reorthonormalization. Diagnostics also record source/target ranks, minimum
projected norm, normal-equation residual, metric orthogonality, electron trace
and metric idempotency. Lost occupied rank rejects the entire item's proposal.
Projection does not extend the target SCF provider's conditioning capabilities.

## Prepared batches and fallback

```python
with source.prepare_batch([atoms]) as small, target.prepare_batch([atoms]) as large:
    small.execute(strict=True)
    report = large.initialize_from(small, strict=False)
    result = large.execute(strict=True)
```

The fleets must have matching item counts and the same RHF/UHF method. Each item
must retain identical ordered nuclei, charge, spin populations, core treatment
and source-density geometry. All-electron/ECP changes are not inferred; ECP
execution remains unsupported. Basis sets may be non-nested and use different
representations. Source and target may use different conventional/DF
approximations, since the target always rebuilds its own operators.

The target must be fresh or have its warm starts explicitly cleared. All
accepted densities are staged and validated before transactional native seed
installation. `strict=True` rejects the whole initialization before mutation
when any item is invalid. With `strict=False`, a failed/missing source, excessive
projection loss or incompatible per-item state leaves that item cold while
valid neighbors retain their projected seeds. Native warm-start safeguards
retry the ordinary cold guess after target stagnation. No failed source's older
retained density can silently substitute for its current failed result.

An accepted proposal has `target_verification="pending_execute"`. Target
execution records its actual status, residuals, energy, iterations and seed
origin (`basis_projection`, `cold`, or `cold_fallback`). A projected seed cannot
be saved as a converged target checkpoint before a successful target solve has
replaced it. Loading another checkpoint supersedes the corresponding projection
origin. Changed target geometries subsequently use the normal prepared-plan
invalidation and warm-density safeguards.

Overlap and metric projection are explicit host setup operations, including
when source/target calculations select CUDA. Source and target solve placement
is retained in their results. This is not a device-resident basis projection,
and it does not add DFT or complex/ROHF state transfer.

## Bounds, counters and validation

`maximum_host_bytes` defaults to 256 MiB for projection's dense numeric
workspace and staged seed copies. A conservative bound is checked before
allocating overlaps and eigensolver operands. Existing target `ResourcePlan`
headroom is checked separately. Source/target HF arenas and implementation-owned
BLAS/Python allocator overhead are outside this numeric workspace bound.
`cross_overlap` separately limits the rectangular output with `maximum_bytes`.

`BatchItemResult.fock_builds` exposes the existing CPU counter of physical Fock
evaluations, including final rebuilds; a joint UHF alpha/beta evaluation counts
once. The additive native query leaves the existing result descriptor ABI
unchanged. CUDA, unexecuted items and incompletely counted warm-to-cold retries
report unavailable (`None`), rather than inferring builds from iterations.

Pinned PySCF 2.14.0/libcint fixtures cover all Cartesian/spherical combinations,
independent geometries and unsorted s/p/d/f shell ownership. Complete RHF/UHF
direct/DF target energy, force and density references are stored in
`tests/data/basis_projection_reference.json`. CI reads these without PySCF;
regeneration uses:

```bash
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/generate_projection_references.py
PYTHONPATH=python:. VIBEQC_LIBRARY=$PWD/build/libvibeqc.so \
  python -m pytest tests/python/test_basis_projection.py \
  tests/python/test_cross_overlap.py tests/python/test_progressive_hf.py -q
```

GPU endpoint tests require an explicit finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'PYTHONPATH=python:. \
  VIBEQC_LIBRARY=$PWD/build-cuda/libvibeqc.so VIBEQC_PROJECTION_CUDA_TEST=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/python/test_basis_projection_cuda.py -q'
```

`benchmarks/basis_projection_gate.py` compares the full source+projection+target
workflow with target-only cold and unchanged-geometry warm baselines using five
interleaved pairs. It retains every timing, stage cost, final energy/force/density,
residual, iteration count and available physical Fock count. It flags different
target densities/energies/forces and records signed energy differences instead
of counting another SCF root as an equivalent speedup. Per-iteration histories
are not exported by this public solver and are explicitly marked unavailable.
No default promotion follows from reduced target-stage iterations alone.

## Measured total cost

The [archived reports](../benchmarks/results/basis-projection-rtx5090/validation-summary.json)
record Release builds on an AMD EPYC 7K62 and NVIDIA RTX 5090 with CUDA 12.9.86.
Slurm job 9077 passed 13 native checks, five enabled CUDA projection cases and
41 CUDA checkpoint regressions. The CUDA projection cases cover Direct/DF
RHF/UHF and a three-item prepared batch with a changed geometry. CPU reference
tests additionally cover diffuse near-duplicate Gaussian conditioning and
independent target densities, energies and forces.

Median milliseconds from five interleaved pairs follow. Each pair is
`target-only baseline / full source+projection+target`. The warm baseline keeps
its converged target plan; the candidate always pays for a fresh source solve.

| Device | Transfer | Cold baseline / full | Warm baseline / full |
| --- | --- | ---: | ---: |
| CPU | RHF STO-3G → def2-SVP | 732.856 / 745.885 | 724.022 / 736.281 |
| CPU | UHF STO-3G → def2-SVP | 716.419 / 727.265 | 715.767 / 726.692 |
| CPU | RHF STO-3G → STO-3G | 10.193 / 21.757 | 9.945 / 21.636 |
| CPU | RHF def2-SVP → STO-3G | 10.173 / 722.383 | 9.835 / 722.883 |
| CUDA | RHF STO-3G → def2-SVP | 41.061 / 43.646 | 26.081 / 42.877 |
| CUDA | UHF STO-3G → def2-SVP | 34.634 / 45.775 | 26.552 / 44.970 |
| CUDA | RHF STO-3G → STO-3G | 4.477 / 10.092 | 1.754 / 9.982 |
| CUDA | RHF def2-SVP → STO-3G | 4.687 / 47.230 | 1.757 / 47.169 |

All final target states agree with their cold baselines within the recorded
energy/force/density gates. This comparison does not establish global HF
stability. CPU small-to-large RHF reduces target Fock builds from 11 to 8, but
the complete workflow costs more. None of these cases demonstrates an overall
speedup; projection remains an explicit initialization option. The shared
performance assessor's `not-run` status means its promotion gate was not met;
the raw timings and executed validation are retained in every report.

Reproduce a report by selecting `--device cpu` or running the CUDA command
inside the same finite Slurm allocation used for the tests:

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY=$PWD/build/libvibeqc.so \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/basis_projection_gate.py --device cpu \
  --case h2-rhf-small-large --repeats 5 --output projection-cpu.json
```

The archive includes the exact GPU execution script, source and binary hashes,
fixture provenance and all eight reports. CPU and CUDA benchmark libraries
share source identity `460a6b09c3d470f7638fe6a3a8e92498d6ccaf29c3aac28d3f4fa03dbd0f5baa`.
