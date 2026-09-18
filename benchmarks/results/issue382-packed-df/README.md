# Packed DF derivative execution — #382

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

The qualified 768-AO RTX 5090 endpoint falls from **9.810 to 7.311 s**
(25.5%) while the derivative weight handoff falls from 452,984,832 to
226,787,328 doubles. The generated derivative equations, basis, metric map,
precision and numerical gates are unchanged. This is the component-ledger
update for parent [#206](https://github.com/jinzhezenggroup/vibeqc/issues/206);
the parent performance gate remains open.

## Endpoint qualification

All VibeQC samples start from the same frozen post-cold density and execute
three SCF iterations. Slice A uses five interleaved full/symmetric pairs;
the final automatic qualification uses three samples at 192/384 AOs and five
symmetric/automatic pairs at 768 AOs. Timings exclude tracing/counter atomics.

| AOs | Full ordered, s | Slice A symmetric, s | Final automatic, s | Auto route |
| --- | ---: | ---: | ---: | --- |
| 192 | 0.269758 | 0.207434 | 0.207233 | symmetric dense |
| 384 | 3.261971 | 2.831671 | 2.837180 | symmetric dense |
| 768 | 9.810031 | 7.294232 | 7.310908 | packed, 256 AO rows |

Final 768 symmetric/automatic medians are 7.295448/7.310908 s: packing trades
about 0.2% latency for half the weights. RHF, spherical def2-SVP, identical
orbital/auxiliary basis, metric relative cutoff `1e-10`, energy/density
convergence `1e-12`/`1e-10`, and the original screening settings are retained.
Independent acceptance remains `1e-9 Ha` and `1e-8 Ha/Bohr`; final maximum
errors are `3.96e-11 Ha` and `1.30e-10 Ha/Bohr`.

Automatic packing requires the qualified 768/768-AO rank-160 RHF occupied
response on RTX 5090 and its existing validated density/factor provenance.
The promoted resident 192–384-AO sm120 route uses symmetric dense consumption.
Unsupported/corrected states keep dense response; generated shell consumers
can fold it. `full`, `symmetric`, `packed`, generic execution, and AO block
sizes `64|128|256|384` remain explicit diagnostics/fallbacks. No scientific
fallback or tolerance is changed.

## Force comparison and #206 component ledger

GPU4PySCF 1.8.1 / PySCF 2.14.0 / CuPy 14.2.0 were rerun on the same GPU,
geometries, representations and auxiliary basis. Its complete warm samples
take one SCF iteration. **Do not use cross-engine complete-warm ratios as an
iteration-matched speedup.** The force columns below exclude SCF in both
engines, but VibeQC exposes an instrumented force stage, not a public clean
force-only entry point. That interval begins after final-state selection and
includes one-electron/Pulay work, DF response, transfers and force return.
GPU4PySCF's clean force-only call computes a full analytic gradient from a
converged state and returns it to host. All five raw samples are retained.

| AOs | VibeQC force stage, s (traced) | GPU4 force-only, s (clean) | VibeQC 3-center, s (traced) | GPU4 fused ip1, s (profiled) | GPU4 complete warm, s (clean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192 | 0.145970 | 0.210760 | 0.074419 | 0.072275 | 0.346231 |
| 384 | 2.415182 | 0.400230 | 0.535996 | 0.183427 | 0.612209 |
| 768 | 3.660065 | 1.349590 | 2.729120 | 0.664643 | 2.139804 |

Native component numbers are stream-event intervals and can include stream
idle time. GPU4 ip1 is summed `sum_ejk_int3c2e_ip1_kernel` GPU activity, not its
whole DF gradient. These measurements identify the remaining boundary; they
are not equivalent isolated-kernel speedup claims. The 384-AO response
producer remains a separate bottleneck despite faster derivatives.

| 768-AO component | Full ordered | Slice A | Final packed |
| --- | ---: | ---: | ---: |
| Complete warm, clean median (s) | 9.810 | 7.294 | 7.311 |
| Force stage, intrusive (s) | 6.166 | 3.634 | 3.660 |
| Three-center derivative interval (s) | 5.252 | 2.721 | 2.729 |
| Raw upload, final trace (s) | — | — | 0.236 |
| Raw transpose, final trace (s) | — | — | 0.013 |
| Occupied projection products, final trace (s) | — | — | 0.276 |
| Occupied metric/weight GEMMs, final trace (s) | — | — | 0.043 |
| Pseudo-density expansion, final trace (s) | — | — | 0.144 |
| Metric Frechet + metric derivatives, final trace (s) | — | — | 0.016 |

Host/GPU intervals overlap and must not be added. `summary.json` retains all
native groups, residuals, resource metadata and exact counters. Slice A's
component measurements are in `slice-a/768-profile.json`; final counters use
a separate instrumented run. Clean timings are never taken from a profiler.

## Exact symmetry and storage

The issue's `384^3 = 56,623,104` triples describe a single dense panel.
The occupied 64-AO panel layout actually executes 57,212,928 triples because
four auxiliary shells straddle panel boundaries. Slice A preserves those
panels; packed panels align to whole shells within the same cap and reach the
exact triangular bound `(384*385/2)*384 = 28,385,280`.

| 768-AO executed quantity | Full | Symmetric dense | Packed |
| --- | ---: | ---: | ---: |
| Logical orbital shell pairs | 147,456 | 73,920 | 73,920 |
| Shell triples | 57,212,928 | 28,680,960 | 28,385,280 |
| Primitive products | 350,896,128 | 176,127,744 | 175,132,672 |
| Nonzero Cartesian component products | 1,943,764,992 | 975,696,384 | 973,859,328 |
| Public weights loaded, including zeros | 452,984,832 | 452,984,832 | 226,787,328 |
| Weight handoff bytes | 3,623,878,656 | 3,623,878,656 | 1,814,298,624 |
| Peak weight panel doubles | 37,748,736 | 37,748,736 | 18,898,944 |
| Peak weight panel bytes | 301,989,888 | 301,989,888 | 151,191,552 |
| Auxiliary panels | 12 | 12 | 13 |

Dense folding reads `Wij+Wji` for off-diagonal shells and retains all ordered
AOs inside a diagonal shell. Packed storage contains `Wii`, `Wij+Wji` for
`i>j`, auxiliary-major. Its diagonal shells consume only lower AO pairs.
Independent nonsymmetric sparse adjoints and mixed spherical/Cartesian
representations validate these multiplicities.

The producer computes `C sym(U)` followed by lower rectangular AO products;
it never constructs a dense auxiliary panel and then compresses it. The final
256-row expansion computes 301,989,888 rectangular entries, including unused
upper triangles inside diagonal blocks, and submits 3,072 pseudo-density
products. Its rectangular scratch peak is 196,608 doubles. Dense-panel and
full-dense-weight-tensor counters are both zero. This is an exact symmetric
packing improvement, not a claim that all response arithmetic is halved.

Slice C's requested sink is already present: generated first derivatives
contract the packed weights immediately into the atomic gradient. No full
coordinate-by-integral derivative tensor, second derivative formula family,
or new response IR is needed. Runtime scheduling and BLAS adapters remain
native; the compiler retains scientific derivative ownership.

## Transfers, memory and remaining execution work

The native warm force response still uploads 3,623,878,656 bytes of raw values
at 768 AOs. Its three borrowed J/K buffers still occupy 10,871,635,968 bytes;
private response scratch with diagnostic counters is 37,865,040 bytes.
Weight-panel savings therefore do **not** imply halved total device allocation.

| AOs | Native response H2D/D2H bytes | GPU4 force-only H2D/D2H bytes | Native sampled peak MiB | GPU4 sampled peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 192 | 56,943,456 / 624 | 127,224 / 13,447 | 6,578 | 1,002 |
| 384 | 4,016,752,288 / 1,200 | 229,632 / 24,399 | 8,258 | 2,034 |
| 768 | 3,628,698,912 / 2,352 | 432,712 / 46,303 | 23,434 | 9,360 |

Transfer scopes differ: native counters cover the DF response; GPU4 counts
all CUPTI copies inside the synchronized force-only NVTX range. Native
final-state downloads and factor validation are separately retained in the
trace groups. The 384-AO native route repeats raw-panel transfers, unlike
the qualified 768 occupied route. GPU4 complete-warm transfer counts can be
reconstructed from the SCF and force ranges in each profile summary.

Process memory is sampled every 100 ms in separate diagnostic runs, including
cold setup; peaks are sampled lower bounds, not allocator high-water marks.
Native resident bytes are queried while the prepared owner is still live:
6,782,189,568 / 8,426,356,736 / 21,252,538,368 at 192/384/768 AOs. GPU4 live,
reserved-pool and process-resident readings are retained separately in its
clean records. Final teardown samples are not labeled resident-after-force.

The shell kernel still groups by angular class, uses subgroup shared atomics
for Cartesian weights, and revisits metadata once per panel. Packing removes
equivalent shell-pair evaluation without adding primitive-count grouping or
a common-class specialization. `kernel-resources.json` retains 192 static
class/schedule resource records for both baseline and changed kernels:
register counts span 94–255 in both, with per-class shared/stack data.
These are pressure indicators, not an occupancy or load-balance measurement.
GPU4 semantic shell/primitive counts are unavailable; its launch/resource
records remain separate from VibeQC's semantic counters.

## Rejected packing schedule

The initial 64-row packed route gave 7.558 s versus symmetric dense 7.319 s.
Its pseudo-density expansion cost 374 ms versus 135 ms dense while derivative
time remained about 2.72 s. Screening 64/128/256/384 rows gave expansion times
372/202/144/190 ms. Five clean 256/384 comparisons gave 7.305/7.356 s.
Increasing rows to 256 computes more diagonal overshoot but reduces the
pseudo-density product count from 9,984 to 3,072. The negative experiment is
retained under `slice-b/`; `blocks/` records the selection evidence.

## Validation and reproduction

Final Slurm job 9667 passed both native executables and all **63** focused GPU
tests (72.51 s). The native shell oracle covers 108 layout/schedule/panel/
representation combinations, s/p/d/f and sparse nonsymmetric adjoints.
Complete HF cases cover RHF/UHF, zero beta rank, geometry/batch replay,
corrected-state fallback, finite discarded metric modes, multiple AO blocks
and panels, and invalid-control recovery. Earlier Slice A/B qualification
passed 24/37 GPU cases. Compute Sanitizer job 9665 reported zero errors for
the shell oracle and 96-AO packed producer. The independent generated
libcint oracle and benchmark reference contracts passed 148 CPU tests.

See [reproduction/README.md](reproduction/README.md) for build, Slurm and
profiling commands. `manifest.json` pins source/library/runner hashes and
versions. Native source snapshots apply to base `f170308`; binaries and full
Nsight captures stay outside Git, with their hashes retained. Repeated
`basis_metadata` is losslessly interned in each JSON's catalog; all timing
samples, forces, energy errors, work counters and controls remain present.
The reduction verified expansion back to the original parsed JSON.

The [decision note](../../../.agents/notes/implemented/performance/2026-09-15-packed-df-derivative-pairs.md)
records invariants and revisit conditions. The ownership delta is retained in
`ownership-delta.json`; native method-adapter growth is conservatively counted
as handwritten scientific code, without claiming new generated equations.
