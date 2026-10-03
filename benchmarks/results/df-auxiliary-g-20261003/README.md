# Auxiliary-g native DF values

`qualification.json` qualifies the internal **value** source extension to
orbital f / auxiliary g. It does not qualify native hundreds-AO molecular
CCSD(T), Lambda or forces. Both retained large supplied-orbital probes have
`accepted: false`; their failed factor/block gates remain visible.

The measured library and generated-header hashes are frozen in the report.
Raw arrays, binaries and full logs remain ignored under
`.artifacts/g-source/` in `/data/jzzeng/qc-df-cuda-g-source-20261003`; no external
archive or release was published. Timings are individual qualification samples,
not speedup claims. Source capacity counters are conservative numeric bounds,
not observed physical GPU peaks. The large probes exclude RHF and CCSD solving.

## Gates and reproduction

Build with CUDA 12.9, `sm_120`, Release, explicit CXX/CUDA `ccache` launchers and
`CCACHE_BASEDIR` set to the checkout root. This qualification disables AOT shell
and stationary-force generation. Use an independent PySCF 2.14.0 environment,
`PYTHONPATH=python:.`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`GENERATIVEQC_LIBRARY=<checkout>/build-cuda/libgenerativeqc.so`, and
`CUDACXX=<CUDA>/bin/nvcc`.

Run each GPU command through a finite Slurm allocation, preserving assigned
visibility, e.g. `srun --mail-type=NONE --partition=main --gres=gpu:5090:1
--nodes=1 --ntasks=1 --time=00:20:00 ...` on the qualification host.

- With `GENERATIVEQC_DF_G_CUDA_TEST=1` and
  `GENERATIVEQC_DF_CC_SOURCE_CUDA_TEST=1`, run
  `pytest -q tests/python/test_df_auxiliary_g.py
  tests/python/test_df_cc_molecular_source.py`: **131 passed** (58 host,
  57 GPU primitive, 16 native source/molecular). Repeat the same command under
  unfiltered `compute-sanitizer --tool memcheck --target-processes all
  --error-exitcode 99`: **131 passed, zero errors**.
- Host compiler/derivative/admission checks:
  `pytest -q tests/python/test_df_codegen.py tests/python/test_df_derivatives.py
  tests/python/test_df_policy.py tests/python/test_df_cc_source_program.py
  tests/python/test_rccsd_cache_admission.py`: **120 passed**.
- Link `tests/native/df_value_probe.cpp` to that same library and CUDA runtime.
  Run `tools/validate_df_source.py --probe <probe> --directory <ignored-output>
  --cases g-cartesian g-spherical g-dependent g-cartesian-spherical`:
  **32/32 pass**. The probe checks raw/transformed/metric derivative rejection
  without output mutation on every g source. A separate spherical, two-item,
  ragged-tile run under memcheck also has **zero errors**.
- Build/run `generativeqc_hf_resource_layout_tests`: assertions are active in
  Release; metadata-only packing admits Cartesian g, omits unused SCF/task
  state and leaves ordinary SCF g rejection intact.

The [decision note](../../../.agents/notes/implemented/numerics/2026-10-03-df-auxiliary-g-values.md)
records the rejected Rys6 shortcut, ownership boundaries, failed large-source
qualification and next conditioning investigation. The report retains the
independent oracle versions and explicit numerical tolerances.
