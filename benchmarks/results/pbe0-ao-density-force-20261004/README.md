# PBE0 AO force screening: frozen evidence, no default promotion

The common JSON/gzip reader and `publication.json` bind numerical samples to
their measured source. No log archive, build product or profile trace is stored.
The Agent Note at `.agents/notes/proposed/2026-10-04-pbe0-force-ao-density-refinement.md`
records the rationale, gates, regression and precise source/count meanings.

## Distinct sources

- Unmodified master: 9c54107caa03f20d05e756ca7e3cd66fb13dabcf,
  source 989f486f1ddf4308eb888ec6f6d90497c77a88b32f4c5ef3cbece83ad7a53dc8,
  library 73c54dc18529365e25b9d4bfb70717002170d37932becffe5193e5548f530e78.
- Prototype at the same base: `prototype-source.patch`,
  source f580d9107b02c9ed7a529702f9a0b7fa096e0706a356013631c8b5b9098ab618,
  library 8cb2ec0a8fbc7700f7a0371e2dc48c28f0552d1e0b9de3da8ee0739c7dc2a6b8.
- Separate producer observer at dc6ea9940ab51b58a985ea0f804ee6a3f8f7179f:
  `profile-source.patch`, source d884bfc505651b350580996c16b933094083bc8ad307e536286636de695bd440,
  library 71b43977c436b8ec9d55e0b48fd582315b7a710948b242d72e2c6b540f7c08ac.

The latest-master PR rebuild/requalification is separate; these historical
records are never rewritten to claim timing of a newer source or binary.

## Retained observations and limits

`master-*.json.gz` retains all 72 native +72 fresh reference endpoints.
`prototype-*.json.gz` retains all 168 native +84 fresh reference endpoints,
including the H2CO holdout. Each contains the complete original numerical record,
its receipt/outcome and original raw-file SHA256. JSON compaction preserves
numbers and ordering, not the original whitespace/byte serialization.

Every native E/F point passes every matching-geometry reference pair at the
unchanged 1e-8 Eh /1e-7 Eh/Bohr gates. Native 48/96 warm medians change from
18.017958/78.015489 to 17.216479/74.788140 seconds. This is **ordered endpoint
observation**, not an interleaved causal speedup estimate. The 48-atom cold case
regresses. All cold/moved costs and iterations are retained; force-only screening
does not explain different SCF trajectories. Old absent Fock counts stay null.

`source-timings.json.gz` contains separately interleaved integral-source replay,
26.730820→23.532602 s median, not full endpoint timing. `producer-counts.json`
records the separate real-producer observer: 595,220,532→385,596,186 admitted
high-order AO quartets. Explicit all-center and unique-center Dual3 gradient
calls are distinct counters, not primitive/root/FLOP counts; low-order work is
unobserved. Atomic profiling time cannot substitute for clean timing.

`controls.json.gz` preserves the water/water-cation finite-difference results and
the original failed OH controls, plus both full 200-iteration OH diagnostic
histories. Water-cation success does not repair or qualify OH. Full routine
sanitizer/build logs and JUnit remain ignored; the Agent Note records their
scoped outcomes, not blanket acceptance of a newer binary.

## Reproduction

Use an isolated checkout at the recorded base, apply only its named source
patch, then check the native source identity before using the associated
artifact. Build with the repository's explicit CXX/CUDA ccache launchers and
checkout-root `CCACHE_BASEDIR`; do not reuse a library with a different digest.
The library digests identify measured builds, not binary artifacts shipped here.

Run all real-GPU work in a finite Slurm allocation, preserving its assigned
`CUDA_VISIBLE_DEVICES`, for example:

```sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:00:00 \
  python -m benchmarks.readme_pbe0 reference --atoms 96 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
  --repeats 5 --output .artifacts/reference.json
```

Run `native` with the same arguments and `--reference .artifacts/reference.json`.
Set `GENERATIVEQC_BOUNDED_FORCE_AO_DENSITY=0` or `1` explicitly; keep
`GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=0` for this isolated experiment.
The measured wrappers additionally record actual GPU4PySCF `XCfun.on_gpu` and
the loaded native binary. `prototype-holdout-runner.py` adapts only molecule
registration/protocol labeling; `def2-svp-hco.json` retains its basis provenance.
Preserve the upstream basis-data license recorded in that input and the
repository's `LICENSES/bse-data-BSD-3-Clause.txt`.

Historical runner/probe sources use `.py.txt`/`.cpp.txt` to preserve their exact
measured bytes rather than reformatting them as current executable source. Copy
them to the corresponding language suffix in an isolated reproduction checkout.

**Fixed-density replay limitation:** the exact 13,759,241-byte text density input
for `profile-probe.cpp` remains local/ignored. Its SHA256 and accepted-state
provenance/independent source arrays are retained in `producer-sources.json.gz`.
This is not a self-contained offline replay package for the exact census.
Re-solving the provided molecule generates a new accepted density and must
receive a new input identity; it cannot silently inherit the recorded counts.
No production/default promotion or independent-review approval is claimed.
