# Direct-HF integration qualification after #456

The reviewed production source is `e4f2f67fc5bdf6bc661e9dac4b92b1c59116162e`.
It integrates #456's Graph/bucket owners while preserving #427's captured
mixed-stage and mandatory exact-FP64 refinement distinction. All repository
hooks and 133 initial host tests passed before the source was frozen.

## Production and numerical evidence

The clean **Release / default AOT / stable-shards / sm_120** build uses
CUDA 12.9.1, with fast-compile disabled. Slurm 10210 completed the build;
Slurm 10215 ran the independent checks and clean paired matrix on an exclusive
RTX 5090 (driver 580.95.05). The library SHA-256 is
`1fb2545d5790c0cfade19b558613e41e0dd21045ffa307db90a6a06f2197341b`.

All **242 host regressions**, the public independent-CPU OH regression,
and the complete independent accuracy probe passed. All **four endpoints,
seven interleaved pairs each** passed their original energy/force/speed gates.
No outlier, geometry, iteration budget, failed observation or timing was removed.
Only 96 AO uses the previously qualified stock full-Fock reference (`direct_scf=False`,
gradient tolerance `1e-11`); 192 AO retains incremental Fock and `1e-8`.
Native tolerances remain energy `1e-12`, density `1e-10`, screening `1e-14`,
with 100 maximum iterations. The complete reference work is timed.

| AO / batch | Max energy error (Eh) | Max force error (Eh/bohr) | Force gate | Ordinary speedup | Iteration-matched speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 / 1 | 3.80851e-12 | 2.31696e-12 | 3e-11 | 5.514x | 5.514x |
| 96 / 4 | 4.718e-12 | 2.42273e-12 | 3e-11 | 9.820x | 9.82x |
| 192 / 1 | 3.18323e-12 | 5.29965e-11 | 5e-10 | 4.324x | unavailable |
| 192 / 4 | 6.59384e-12 | 2.35374e-10 | 5e-10 | 4.813x | unavailable |

These are complete endpoint comparisons against sequential single-system
GPU4PySCF objects, not isolated-kernel speedups. An unavailable iteration-matched
statistic is not replaced by ordinary timing. The 192-AO points have no speed
admission floor; both 96-AO points retain the 1x iteration-matched floor.

## Native-suite failures: reproduced on the exact integration base

**Neither raw full native suite is labelled passed.** Candidate: 62/65 pass.
The exact integration base `f1f1250ef4d6688b1b5bd1e3bb57e5e3075f4d68`, including
#456 but not #427, was separately rebuilt with the same production configuration.
Slurm 10219 ran all 64 base tests: 61 pass, with exactly the same three failures:

- `vibeqc_mp2_contract_tests`: the stub-oriented test rejects a successful CUDA
  derivative on a real GPU ("must not run without the explicit GPU gate").
- `vibeqc_density_fitting_tests`: retained-value raw-upload/weight accounting
  assertion. This is not the older numerical response-arithmetic diagnosis.
- `vibeqc_ks_cuda_tests`: the CUDA OH occupation cycle does not converge.

Every shared native result matches; #427 adds one passing force-convergence
test and introduces **no new native failure**. The base library SHA-256 is
`4d5e3f4e9bd59c6d1b9bf098bb0cb523f249b5b01b5e67b45747f63138e299dd`.
The exact candidate OH test/fixture were also copied unchanged to test the base:
it fails the original `1e-9` force gate (largest violating error about `1.55028e-9`),
whereas the candidate passes cold/frozen/changed-geometry/energy-only checks.

After the completed candidate timing job, an independent copy of its complete
build tree was saved. The base was checked out and incrementally rebuilt by
CMake/Ninja using only correctly reusable objects; all base GPU work finished
before the original candidate source and byte-identical build were restored.
No live build or benchmark was subjected to a concurrent source mutation.

The first baseline summary adapter used the wrong location for CTest's relative
JUnit output. CTest had written it beneath its build directory. The exact XML
was copied to the expected audit path and parsed without rerunning any test.
The original adapter traceback, both XML paths, hashes, and recovery record
remain retained; the measurement itself was not replaced.

The disposition is **LGTM for #427's direct-HF change**, based on independent
force acceptance and exact-base non-regression. It is not a waiver or successful
relabel of the failing native suites, and it does not close #206 or #240.
The historical failed paired campaigns and reference-only qualification remain
separate and unchanged.

## Retention and reproduction

`manifest.json` maps each original logical report/log/script to a lossless gzip
payload with both decoded and stored byte hashes. Every paired scalar, full
force array, timing, convergence record, diagnostic trace and failed result
is retained. Compiled libraries/objects are not stored in Git. Run the CPU-only
integrity/consistency audit from any directory:

```bash
python benchmarks/results/issue206-direct-force-state/integration-e4f2f67/verify.py
```

To repeat the production matrix, build the recorded source with the
`cuda-release-sm120` preset and the recorded CUDA compiler, then run on a finite
exclusive NVIDIA allocation with that library and `PYTHONPATH=python:.`:

```bash
python benchmarks/real_molecule_gate.py --density-fitting none --repeats 7 \
  --reference-gradient-tolerance 96=1e-11 --reference-full-fock 96 \
  --output-directory /path/to/a/new/empty/campaign
```

The exact validation and baseline scripts are retained, including their original
node3 paths. The supplemental accuracy script refers to original retained CPU
oracle paths; its report contains the exact geometries, CPU oracles and every
sample. The production matrix and public OH test are repository-self-contained.
Do not run profiling or compilation concurrently with endpoint timing, or treat
the `srun` launcher's memory statistic as compiler peak memory.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Concurrent final-head integration

Before publishing the evidence, another review merged master through
`e9f7e127f6603d421b1a6e777b8f01eecb7a2fb2`. That update contains #588's
CPU-tuning limit admission checks and #589's CI-generated ownership report.
It changes no file under `src/`, `include/`, `cmake/`, or `tools/`, and does not
change `CMakeLists.txt`, CUDA generators, runtime code or benchmark runners
relative to the measured source. Its code and tests were independently reviewed
and integrated without restoring the deliberately removed ownership snapshot.
The final combined tree passes **279 host regressions**, including CPU tuning,
and all repository hooks. The first additional host invocation omitted the
explicit library path and failed one benchmark import; setting `VIBEQC_LIBRARY`
to the unchanged frozen library corrected the environment without changing a test.
The GPU results above remain attributed to the exact measured source `e4f2f67`,
not misrepresented as a newly rebuilt later documentation/CI/tuning-only head.

Agent: ChatGPT
Model: GPT-6 Astra Pro
