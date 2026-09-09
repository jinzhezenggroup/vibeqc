# HF accuracy evidence (#173)

The CPU experiment uses eight independently referenced HF endpoints, separately
requested energy and force targets, and fresh native solves for each numerical
setting. The requested comparison targets are `1e-6 Eh` for energy and
`1e-7 Eh/bohr` for the maximum absolute force component.

The records are [report.json](cpu/report.json), [scientific states](cpu/states.npz),
[replayed operator audits](cpu/replay.json), and the
[empirical calibration](cpu/estimator.json). Array checksums, shapes, complete
model identities, controls, physical residuals, actual errors, and failed solves
are retained. No PySCF dependency is needed to consume the pinned references.

## Numerical and calibration results

All eight strict CPU endpoints passed `1e-9 Eh` energy, `1e-8 Eh/bohr` force,
and `1e-8` orthonormal-commutator gates. Their largest errors against the pinned
PySCF references were about `1.85e-13 Eh` and `2.04e-12 Eh/bohr`.

Energy convergence, density convergence, and their coupled changes are swept
over `1e-9`, `1e-7`, `1e-6`, and `1e-5`. Each endpoint also retains a deliberately
failed one-iteration solve. A fixed-density operator audit and a separately
relaxed strict solve are distinct records. CPU screening is not swept: the
existing dense CPU oracle evaluates unscreened integrals.

Water and methane form the training families. Hydrogen, ammonia and neutral
HF supply the accepted holdout observations. Every setting for a molecular
family stays on the same side of the split. Helium (no virtual gap), UHF HF+
(unsupported spin method for this calibration), and def2-SVP H2 (different
basis family) are rejected explicitly.

| Holdout measure | Result |
| --- | ---: |
| Energy/force observations evaluated | 72, from three molecular families |
| Observations covered by the empirical envelope | 72 |
| Outside-domain states rejected | 36 |
| Missed requested tolerances | 0 |
| Over-conservative predictions at `1e-9` energy/force targets | 4 |
| Over-conservative predictions at `1e-6` energy/force targets | 1 |

These are correlated observations at one geometry per chemical system, not
72 independent molecules. This initial coverage is **not certification** or
evidence of broad chemical generalization. The estimator reports
`estimated_below_target`/`estimated_above_target`, never a guaranteed pass.

The versioned domain is CPU FP64 conventional Cartesian RHF with the verified
bundled STO-3G data, H/C/N/O/F nuclei, at most 12 AOs and 10 electrons, overlap
condition at most 30, occupied-virtual gap at least 0.1 Eh, and physical residual
at most `1e-2`. Electron-trace and metric-idempotency errors must be at most
`1e-8`; nuclear separations must be at least 0.5 Bohr. Gap and overlap filters
are validity checks, not inverse-Jacobian bounds or electronic-stability tests.

For example, ammonia at the loosest coupled iteration settings differs from
the strict target by about `2e-12 Eh` but `4.4e-7 Eh/bohr`. Energy agreement
alone cannot satisfy an independently requested force tolerance.

## Model changes and costs

[Metric experiments](metric.json) vary the auxiliary metric threshold at fixed
nuclei and basis. Raising it to 0.2 changes the retained rank from two to one.
The RHF energy changes by about 0.214 Eh and the UHF maximum force component by
about 0.031 Eh/bohr. The comparison reports `changed_model`; it does not relabel
the new fitted operator as numerical accuracy for the original target.

`solve_seconds` measures the complete native energy-plus-force probe.
`audit_setup_seconds`, each `physical_audit.audit_seconds`, `features_seconds`,
and prediction `evaluation_seconds` disclose the independent audit and
estimation overhead. `total_seconds` includes the suite through state-archive
publication. These single-pass diagnostic timings describe error versus cost;
they are not a performance-promotion gate or a claimed production speedup.
The audit deliberately uses tiny dense CPU tensors and rejects larger systems.

## Reproduction

Build the CPU library normally, then run from the repository root:

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.validate_accuracy --output /tmp/accuracy-cpu

PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.vibeqc_numerics.replay /tmp/accuracy-cpu/report.json \
  --output /tmp/accuracy-cpu/replay.json

PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.vibeqc_numerics.metric_experiment --output /tmp/metric.json
```

The added HF and def2-SVP H2 references were generated twice using PySCF
2.14.0 with identical numerical data. The pinned inputs/provenance and the
two-generation comparison are in `tests/reference_data/accuracy/`. Regeneration
requires an explicitly prepared PySCF/NumPy/threadpoolctl environment:

```bash
PYTHONPATH=python:. python -m tools.generate_accuracy_references \
  --output /tmp/accuracy-references
```

## CUDA and arithmetic experiments

CUDA 12.9.86/sm_120 execution on an NVIDIA GeForce RTX 5090 passed all eight
independent strict endpoints. Maximum reference differences were `1.07e-13 Eh`
and `5.19e-12 Eh/bohr`. The [CUDA report](cuda/report.json) retains 160 converged
variants, eight deliberate failures, and [replayable audits](cuda/replay.json).
Independent screening sweeps changed energy by zero at the reported precision
and forces by at most `3.42e-15 Eh/bohr` on this small fixture set. That result
does not establish coverage for larger screened workloads. The largest coupled
iteration force error was `1.50e-6 Eh/bohr` despite an energy error of only
`1.20e-11 Eh`.

The [arithmetic report](arithmetic/report.json) compares 96 existing experimental
mixed-Fock requests (`1e-6`, `1e-3`, `0.1`) with identical FP64-requested iteration
settings. Every state converged and its [operator audit replayed](arithmetic/replay.json).
Matched energies were identical at reported precision; matched forces differed
by at most `5.48e-15 Eh/bohr`. Actual FP32 work counts are unavailable and recorded
as null. These observations are therefore inconclusive for arithmetic-error
coverage, and do not establish an FP32 accuracy budget or execution speedup.
The CPU estimator rejects CUDA/mixed requests; #174 owns the execution/refinement
controller and its actual FP32-work instrumentation.

All real-device runs used Slurm on this host, with the allocated device visibility
preserved. With a CUDA library in `build-cuda`, reproduce using:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 env PYTHONPATH=python:. \
  VIBEQC_LIBRARY="$PWD/build-cuda/libvibeqc.so" \
  VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m tools.validate_accuracy --backend cuda --output /tmp/accuracy-cuda

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 env PYTHONPATH=python:. \
  VIBEQC_LIBRARY="$PWD/build-cuda/libvibeqc.so" \
  VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m tools.vibeqc_numerics.precision_experiment \
  /tmp/accuracy-cuda/report.json --output /tmp/accuracy-arithmetic
```

Both archives replay with the CPU command above, substituting their report paths.
