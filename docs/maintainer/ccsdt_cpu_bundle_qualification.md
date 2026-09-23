# RCCSD(T) CPU response bundle qualification

The internal conventional RCCSD(T) force endpoint uses the shared native CPU
TensorIR executor. CC/Lambda programs and fixed-orbital weights use its existing
prewarm owner. Each requested perturbative-triples VJP now prewarms the exact
virtual-tile program set before executing those tiles. A direct single-program
execution remains available to callers that do not prewarm.

Run the complete-endpoint comparator from a Linux checkout with the CPU native
library and `.[reference-test]` installed:

```bash
export VIBEQC_LIBRARY="$PWD/build/libvibeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python tools/benchmark_ccsdt_cpu_bundles.py \
  --case nh3 \
  --cache-root build/ccsdt-nh3-fresh-cache \
  --output build/ccsdt-nh3-report.json
```

Use new cache and output paths for each qualification. The comparator runs the
same fresh RHF → RCCSD → corrected Lambda → total orbital response → analytic
gradient endpoint in four independent processes: separate triples tile artifacts
and bundled triples tiles, each with cold and warm disk cache. Only the triples
tile prewarm is disabled for the separate route. It records complete endpoint
wall time, artifact construction wall time (source emission through load), cold
compiler/linker duration, TensorIR program executions and scalar-work estimate,
shared-library counts and sizes, peak and retained process RSS, and logical
response reservation. It fails if the routes differ in TensorIR execution
inventory, accepted CC/response identity, orbital response actions, or the pinned
PySCF 2.14.0 analytic gradient gate. A warm run must compile no new artifact.

The [PR evidence workflow](../../.github/workflows/ccsdt-cpu-bundle-evidence.yml)
runs H2O and NH3 with a clean cache for each route and retains the JSON reports.
Measured RSS includes Python, loaded libraries, source generation and the native
endpoint; the reported logical reservation remains the separate numeric-buffer
contract. Artifact construction wall time includes source emission and cache
integrity/load work, so it is not interchangeable with compiler/linker duration.
