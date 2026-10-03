# CPU KS final-maximum closure

This is a selective correctness publication, **not a lossless raw backup**.
Full-precision selected histories, metrics, input, source/runtime identities,
negative outcomes and portable reproduction scripts are retained here. Complete
raw arrays, snapshots, binaries, supervisors and failed/partial logs remain local.
No new archive scheme, fixture or external storage is introduced.

## Decision and scope

Accept the narrow CPU final-closure correction: require the physical commutator
entrywise maximum alongside the existing strict RMS, energy and density tests.
The maximum gate is inclusive `max <= min(1e-8, density_tolerance)`. Rejected
ordinary RKS closure restarts from the actually evaluated density/factor, clears
DIIS, and spends only the original remaining primary-iteration budget. Incremental
RKS retains its pre-existing separate budgets; UKS uses its existing bounded
stabilized closure. No tolerance, cap, numerical method, mirror optimization or
snapshot acceptance policy was relaxed.

Restarted-stage history uses the reserved -1 C marker and Python null only for
its unavailable first energy delta. Unexpected NaN/Inf are still failures. These
recording changes do not participate in convergence arithmetic or alter the ABI.

Frozen measurement: b3c9f339de410c9c167d56a03fa7b5ed18cde825, tree
25d0c2bbad37264b435fda787a960ae8a9ef73e2, native library SHA-256
31d19e38ba3107685c704bc93997fb1599865bbabfd2956da953ff4a7128d002.
The publication union has its own source/build identity in provenance.json; the
large measurement is not relabeled as a measurement of that newer union.

## Observations and limits

- Original public 1a4acc51 source returned 17 iterations/19 Fock builds, but strict
  final snapshot failed. SCF success alone was not export success
- Corrected source used 21 iterations/25 Fock builds under the unchanged 150
  primary-iteration limit. The first 17 history rows match exactly. Null energy
  deltas occur at rows 1 and 18. Pre/post snapshot diagnostics are identical,
  zero export Fock builds were added, and all native owners closed
- Final native maximum commutator: 9.626743846524732e-13 <= 1e-10. Two completed
  two-build finalizations are inferred from source-reviewed counters; per-audit
  maxima/intermediate rejected matrices were not exposed or remeasured
- The original independent historical/native coordinate comparison passes but
  its molecular-weight comparison fails at 23 far-tail points. That audit remains
  failed. The separate same-coordinate check constructs independent Python raw
  measures and Becke partitions at untouched native coordinates, never using
  native weights as input, divisor or fitted normalization
- An intervening same-coordinate attempt silently lost its explicit grids when
  PySCF radii_adjust was assigned. All its physical results are invalid and are
  excluded from acceptance. The corrected audit configures policy first, binds
  exact arrays, prohibits rebuilding, and verifies 12 before/after identities
- Corrected independent fixed-D audit: energy error 1.8758328224066645e-12 Eh,
  density maximum 6.3687943807622105e-12, Fock maximum 5.531999683816105e-13,
  commutator maximum 1.006750238730092e-12. All original and additional gates pass
  without a native execute or SCF solve; the previously retained oracle D is used
- All 331776 tail points were traversed in 1296 nonempty 256-point tiles without
  clipping. Counts are execution receipts bound to reviewed source and grids;
  individual rho samples were not retained and were not independently replayed
- Saved independent F exactly equals H+J+Vxc. Saved F/D/S commutators and metric
  differences reproduce exactly. Native-density energy reconstruction uses saved
  H/J plus recorded XC integral. Oracle J/Vxc/Exc were not retained: its energy
  reconstruction remains source-bound execution evidence, not a second integral

There is no performance/default-promotion, universal-export, CUDA/COSX,
large-force/response, warm/composed or large changed-geometry qualification claim.
The correction performs additional primary/finalization work. Small lifecycle,
force and response controls do not imply the unmeasured large compositions.

## Preserved outcomes

provenance.json pins original raw receipt hashes separately from normalized
selected values. The original failed snapshot, cross-node grid failure, invalid
wrong-grid run, earlier 10680aa1/preparation a4965b77, nonfinite parser regressions,
initial hook/build attempts and expected device skips were retained locally.
The first closure preparation was never used for a large molecular execution;
the diagnostic sentinel was corrected and qualified before the measured run.
The public samples share the exactly equal 17-row historical prefix once rather
than copying it. They retain every full-precision value in both histories.

## Verify without chemistry

From this repository, with PYTHONPATH=python:.

    python -B benchmarks/results/cpu-ks-final-closure-20261003/reproduce.py verify

This uses the existing publication reader/validator, checks selected scalar gates,
work/history/grid invariants and visible negative outcomes. It does not reconstruct
raw arrays that are not shipped here. Normal regression tests remain offline.

## Reconstruct exact measured source

Use a full clone containing public base 9c54107caa03f20d05e756ca7e3cd66fb13dabcf
and this publication's 13 exact correction postimages. The local measured b3 commit
object itself is not required. The helper validates each postimage, creates a new
worktree on public base, and checks the exact measured tree before committing:

    python benchmarks/results/cpu-ks-final-closure-20261003/reconstruct_source.py \
      --repository "$PWD" --destination /tmp/ks-corrected-source

Add --baseline with a different destination to restore original public
1a4acc519eb881cc19d418de65ecca72324359d4. No helper fetches automatically or overwrites
existing destinations. If shallow, explicitly fetch the documented public base
first. Source reconstruction is not numerical revalidation.

Configure each new source separately; never reuse the other source's library:

    ccache --version
    export CCACHE_BASEDIR=/tmp/ks-corrected-source
    cmake -S /tmp/ks-corrected-source -B /tmp/ks-corrected-source/build/cpu-revalidation \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DGENERATIVEQC_ENABLE_CUDA=OFF -DGENERATIVEQC_ENABLE_CXX_PCH=OFF \
      -DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas -DGENERATIVEQC_BUILD_TESTS=ON
    taskset -c 2 cmake --build /tmp/ks-corrected-source/build/cpu-revalidation -j1

Use the recorded Python 3.12, PySCF 2.14.0/Libxc 7.0.0 and OpenBLAS runtime for
closest reproduction. Configure OpenBLAS via the project's supported pkg-config
setup. Hashes identify the measured runtime; new builds/runs get new identities.

## Explicitly requested large reproduction

The following commands are provided, **not executed during publication**. Run
serially in new ignored destinations. Each scientific child is pinned to one CPU,
requires six thread caps=1, 7 GiB available/2 GiB disk, 8 GiB address-space cap,
6.5 GiB supervised RSS cap and 900-second timeout. Failures and logs are preserved.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    export PYTHONPATH=/tmp/ks-corrected-source/python:/tmp/ks-corrected-source
    P="$PWD/benchmarks/results/cpu-ks-final-closure-20261003/reproduce.py"
    python -B "$P" native --source /tmp/ks-corrected-source \
      --library /tmp/ks-corrected-source/build/cpu-revalidation/libgenerativeqc.so \
      --output "$PWD/.artifacts/ks-repro/native"
    python -B "$P" oracle --source /tmp/ks-corrected-source \
      --output "$PWD/.artifacts/ks-repro/oracle"
    python -B "$P" independent --source /tmp/ks-corrected-source \
      --native "$PWD/.artifacts/ks-repro/native" \
      --native-supervisor "$PWD/.artifacts/ks-repro/native.supervisor.json" \
      --oracle "$PWD/.artifacts/ks-repro/oracle/result.json" \
      --output "$PWD/.artifacts/ks-repro/independent"

The separate oracle mode explicitly makes one new PySCF SCF solve using the exact
input basis/grid; independent mode makes zero SCF/native calls and audits fixed
saved densities. Its portable intake requires this explicitly fresh oracle receipt,
matching input/source-tree identities and declared density hash; missing provenance
is not treated as historical evidence. The original retained oracle is represented
by historical checksums/measurements here, not an input that readers must obtain. This differs from the original qualification, which reused a
previous oracle. A new reproduction must not be called the original raw record.
To reproduce the original failure, run native mode against the separately built
baseline source/library and a new output; it is expected to fail at snapshot.
The full original focused command is retained in run_small_controls.sh.
Its frozen result is 152 passes / 11 device skips plus 9 native suites.
run_union_controls.sh retains the separate refreshed-union native/control command. The earlier 701 local
union passed 398 controls / 9 device skips and 9 native suites; it remains a separately
identified superseded integration. The refreshed 837 union additionally includes
the VWN empty-spin repair and its focused host tests, existing VWN/provenance and
hybrid/B3LYP consumers. Its exact totals, device-skip names/reasons and build identity
are in provenance.json, along with 160 storage tests and 19 portable-harness controls.
Small deterministic controls can be run without these large commands:

    pytest -q tests/python/test_cpu_ks_final_closure.py tests/python/test_ks_stage_baseline.py
    ctest --test-dir build/cpu-revalidation --output-on-failure -R 'generativeqc_(dft_density_source|uks_state|ks_final_state)_tests'

Portable scripts retain the measured native capture, fixed-density numerical
statements, same-coordinate weight helper and explicit-grid guard. Only invocation,
path/provenance binding and a separate opt-in oracle generator are adapted. New
scripts receive source-only/mock validation; no new full PBE96 execution is claimed.
