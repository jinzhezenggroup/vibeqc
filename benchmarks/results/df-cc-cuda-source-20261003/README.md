# Native CUDA DF-CC source qualification

`qualification.json` records eight molecular source cases and two complete
internal H2 energy calls on an RTX 5090. These are correctness/resource samples,
not performance promotion or hundreds-AO CCSD(T)/force evidence. The regular
public DF descriptors remain unsupported.

The source oracle uses committed independent raw three-center/metric data,
NumPy symmetric metric whitening and full contractions. Cases cover H2, water,
LiH and f-shell HeH, with and without a duplicated auxiliary s shell. The
duplicate changes Q and exercises truncated metric rank. The complete H2
oracle is a determinant Hamiltonian diagonalization: CCSD is exact for two
electrons. Production numerical work runs in the native CUDA library.

The retained values use Eh and seconds. Capacity fields are conservative
numeric reservations, not observed physical peaks. Contraction summands count
semantic scalar products, not hardware FLOPs. Molecular timing covers the
entire internal method call (conventional RHF, DF source, CCSD), excluding
input normalization and the independent oracle. These ordered single samples
include different CUDA first-use costs and must not be compared as speedups.

The CUDA 12.9 Release build uses architecture 120, explicit C++/CUDA ccache
launchers and checkout-root `CCACHE_BASEDIR`. AOT shell and stationary-force
artifacts are disabled; the generic source and reference routes are exercised.
The missing generic registry query found by library loading is fixed in this
change. Source-file, generated build-identity and binary hashes bind the run.
The measured scientific snapshot is `6855da896`; the subsequent integer-type
portability fix preserves these historical source/binary hashes.

After building `generativeqc` in `build-cuda`, reproduce from this checkout:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc '
    export GENERATIVEQC_LIBRARY=$PWD/build-cuda/libgenerativeqc.so
    export GENERATIVEQC_DF_CC_SOURCE_CUDA_TEST=1
    export PYTHONPATH=python:.
    python -m pytest -q tests/python/test_df_cc_molecular_source.py \
      --basetemp=.artifacts/df-source/source-final
    compute-sanitizer --tool memcheck --target-processes all --error-exitcode=99 \
      python -m pytest -q tests/python/test_df_cc_molecular_source.py \
        --basetemp=.artifacts/df-source/memcheck-final
  '
```

Tests retain per-case source/molecular JSON in those ignored temporary
directories. Memcheck runs the complete suite without kernel filters. All ten
tests pass and it reports zero errors. The conventional RCCSD/RCCSD(T) public
and force-resource suites pass 45 tests; three live PySCF analytic-oracle tests
are skipped because PySCF is absent. The committed-oracle energy/force tests
and exact force-budget tests run normally.
