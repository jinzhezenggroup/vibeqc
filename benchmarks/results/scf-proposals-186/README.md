# SCF proposal baseline evidence (#186)

This archive measures the native CPU proposal interface using 15 independent
reference cases: H2/STO-3G, H2/def2-SVP, water, methane, and HF+ UHF, each at
geometry scale 0.98, 1, and 1.02. Whole molecule families stay in one split,
including H2's two basis sets. The pinned PySCF 2.14.0 reference generator
produced byte-identical archives in two independent runs.

All 115 converged configurations meet the declared 1e-9 Eh energy and
1e-8 Eh/bohr maximum force gates. The largest observed differences are
2.274e-13 Eh and 2.178e-10 Eh/bohr. All 15 deliberately iteration-limited
failures remain in the report and traces. These comparisons do not establish
the intended electronic state: native stability remains `not_evaluated` and
intended-state status remains `unverified`. Independent competing-start and
internal-stability results are recorded separately.

| Baseline | Cases | Mean iterations | Total physical Fock builds |
| --- | ---: | ---: | ---: |
| Cold DIIS | 15 | 9.133 | 167 |
| Fixed point | 15 | 19.867 | 328 |
| Safeguarded DIIS | 15 | 19.933 | 613 |
| Safeguarded mixing | 15 | 47.333 | 1435 |
| Diagonal OV preconditioner | 15 | 34.933 | 1204 |
| Same-geometry warm density | 15 | 2.000 | 60 |
| Previous-geometry projected orbitals | 10 | 7.800 | 98 |
| Two-geometry density extrapolation | 5 | 7.400 | 47 |
| Existing FleetPlan warm density | 10 | 7.400 | 94 |
| Deliberate one-iteration failure | 15 | 1.000 | 15 |

Counts describe one execution per configuration, including rejected proposal
trials and final Fock rebuilds. One joint UHF build counts once. Every repeated
configuration has identical work counts across its three runs. Same-geometry
warm results require a prior converged solve, whose cost is recorded separately.
Projected/extrapolated states have more initial information than cold DIIS.
No existing FleetPlan fallback occurred; its older ABI cannot report all work
after fallback, so the runner reports that count as unavailable in such cases.
These deterministic proposal baselines add Fock work here and are not a
performance promotion or a learned-acceleration claim. Native Newton/SOSCF is
not available; the diagonal OV action is not a second-order solver.

## Timing conditions

The CPU Release build ran on an AMD EPYC 7K62 with affinity fixed to CPU 8,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, Python 3.13.9 and NumPy 2.5.1.
A CUDA compiler process tree was restricted to CPUs 0-3 and could run
concurrently; shared caches and memory make these timings diagnostic.
`report.json` retains all samples, source hashes, and the native library hash.

The median across cases of disabled-interface/legacy-interface latency is
0.9961, with range 0.9550-1.0011. Both interfaces execute identical final results.
The recorded comparison with the pre-change library at parent revision
`78cb50a268c238c75f0612c8cefb7b3ab2dd6f12` additionally confirms bit-identical
energies, densities, forces and iteration counts for all 15 cases. This small
CPU suite diagnoses no material disabled-path overhead; it is not a universal
overhead bound.

Timed solves include native integral preparation, iterations, rejected trials,
fallback, and analytic forces. Raw-system ownership setup, independent audits,
reference generation and disk export are outside the timing interval. Three
repeats apply to the main cold/proposal/transport baselines and the legacy
comparison; same-geometry warm, FleetPlan warm and deliberate-failure rows
retain one diagnostic timing each. Snapshot counterfactual timings measure
one snapshot, without full-solve DIIS evolution or the trajectory's failure
budget. They must not be used as complete-solve speed measurements.

## Artifacts and validation

`report.json` contains 130 configuration records. Each of the 15 cases has
`cold`, `proposal`, and `failed` JSON/NPZ traces, for 45 traces and 451 snapshots.
Every trace passes the bounded, checksummed loader. Reconstructing each source
and independently contracting its target operator gives maximum differences
of 1.990e-13 Eh in energy, 1.422e-14 Eh in Fock entries, and 1.061e-14 in AO
commutator entries. `validation.json` records these checks, artifact hashes,
and the pre-change comparison data. Every individual artifact is below 1 MiB.

Validation also includes 12 native CPU test executables, a full Python suite
(1,159 passed, 188 skipped), and a final 36-test focused suite after the last
failure-isolation and policy refinements. A CUDA 12.9 production sm_120 build
succeeded. Native, UHF, density-fitting and proposal suites all passed under
a Slurm RTX 5090 allocation. CUDA proposal callbacks remain explicitly
unsupported; the GPU checks cover capability rejection and existing paths.

## Reproduction

From the repository root, configure a CPU build and a Python environment with
NumPy, then run the benchmark into a new directory:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build --parallel 4
env PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 8 python -m tools.validate_scf_proposals \
  --output /tmp/scf-proposal-reproduction --repeats 3
```

Use a CPU available to the current allocation if CPU 8 is unavailable. Reference
regeneration additionally requires `pyscf==2.14.0`; run
`python -m tools.generate_scf_proposal_references --help` for the explicit
output option. Ordinary execution neither requires PySCF nor exports data.
See `docs/scf_proposals.md` for the physical invariants, acceptance policy,
failure budget, ownership, and trace/replay contracts.
