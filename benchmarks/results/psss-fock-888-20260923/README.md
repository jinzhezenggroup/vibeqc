# psss Fock retirement qualification (#888)

The candidate removes the dedicated handwritten psss value/Fock path and caps
packed generated Fock grids in units of queue claims. The native baseline and
candidate use the same landed #945 coverage repair and production Release/sm_120
configuration. Exact revisions, scientific-source identities, library checksums,
and Slurm jobs are retained in each workload record.

The retirement gate is candidate/baseline <= 1.02 for every cold, warm,
changed-geometry and changed-warm energy-plus-analytic-force endpoint median.
Three ABBA cycles retain all repetitions. The RHF matrix uses STO-3G and def2-SVP,
batches 1/3, and fixed/resident/paged execution (seven warm repeats). UHF OH
spherical def2-SVP uses batches 1/4 and fixed/paged execution (five repeats).
The larger spherical def2-SVP water tetramer uses fixed execution (three repeats).
This is a structural-retirement non-regression gate, not a significant-speedup
promotion claim. Preparation-plus-cold medians are also retained.

`matrix.json` summarizes all 17 configurations. Companion files retain one
reference tensor per phase and every measured timing, energy/force maximum and
RMS error, and iteration count in original process order. Error reductions include
all force components in every molecule, including ragged batches. No sample is
filtered. Per-class shell/tile/AO/primitive work comes from separate final-density
profiling replays; iteration parity protects the semantic SCF work comparison.

`regressions.json` records 20 strict allocated-GPU tests without skips, including
the original #945 reproductions in default/forced execution and independent
libcint RHF/UHF comparisons with psss/fsss generated-Fock registry gaps. Raw
independent psss-gap outputs and quantitative acceptance errors are retained.
`resource-summary.json` is a separate intrusive Nsight replay; its timings are
not included in clean endpoint comparisons. Full traces, interrupted/pre-merge
runs, build products, and routine logs remain ignored local artifacts.

Rebuild both recorded source revisions with the `cuda-release-sm120` preset.
From the candidate checkout, using Python with NumPy/PySCF and pytest installed:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  python benchmarks/results/psss-fock-888-20260923/ragged_endpoint_gate.py \
  --base /path/to/native-baseline --candidate "$PWD" \
  --basis sto-3g --batch 3 --schedules fixed,resident,paged \
  --cycles 3 --repeats 7 --output .artifacts/888-ragged.json
```

Repeat with `--basis def2-svp` and batches 1/3 for the other RHF points.
`direct_endpoint_gate.py` accepts the same roots plus
`--cases oh-def2-svp-spherical-uhf --batches 1,4 --schedules fixed,paged
--cycles 3 --repeats 5`; for the larger holdout use
`--cases water-tetramer-def2-svp-spherical --batches 1 --schedules fixed
--cycles 3 --repeats 3`. Pass explicit `--base-library`/`--candidate-library` if
libraries are outside the preset build directories. Both runners reject source
identity mismatches. The final JSON numerical/performance flags must be checked;
process exit success alone is not a passing performance gate.

For the focused oracle tests, run pytest under the same finite Slurm allocation
with `PYTHONPATH=python:.`, `VIBEQC_RESOURCE_CUDA_TEST=1`, and a source-matched
`VIBEQC_LIBRARY`; select `test_bounded_direct_high_l_cuda.py` and
`test_bounded_psss_fock_retirement_cuda.py`. Preserve Slurm device visibility.

The profiler replay uses `VIBEQC_PSSS_RESIDENT_BRA=1`,
`VIBEQC_BOUNDED_DIRECT_STREAMING=none`, and `VIBEQC_PROFILE=off`. Under Slurm,
run each source-matched library with `nsys profile --trace=cuda,nvtx
--cuda-graph-trace=node --sample=none --cpuctxsw=none`, followed by
`python tools/validate_weighted_eri_endpoints.py --worker
'{"method":"rhf","basis":"sto-3g","batch":3,"repeats":1,"schedule":"resident"}'`.
Export SQLite with `nsys export --type sqlite` to inspect launch grids and
register/shared/local storage. Keep this intrusive replay separate from clean
ABBA timing.
