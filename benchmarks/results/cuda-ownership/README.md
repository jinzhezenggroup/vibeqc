# CUDA ownership retirement evidence

The two completed phases retain independent, reproducible evidence:

- [One-electron values](one-electron/publication.json): the first generated S/T/V
  promotion and retirement delivered by #252.
- [DF values and derivatives](df/README.md): shared generated bulk/source/weighted
  mathematics and removal of the coordinate-wise CUDA HF response in #255.
  Its complete 18-case, five-sample paired comparison passes all 90 gates (worst
  ratio 1.007245172); native/GPU integration, independent libcint checks and both
  sanitizers pass. The earlier rejected full comparison and subsequent diagnostic
  evidence remain explicitly labeled and retained alongside the accepted bundle.

The versioned [current ownership report](../../../docs/cuda_ownership_current.json)
and [physical accounting](df/physical-lines.json) distinguish scientific/runtime
edits from unchanged lines that changed category. Neither kernel resources nor
removed plan estimates are claims of measured whole-HF peak-memory savings.

## First-phase one-electron evidence

The `one-electron` bundle supports the generated S/T/V value promotion in #231.
It compares clean candidate `fdc7f40737fae85e9a25d771158ffdd1cace2b00` with clean
handwritten baseline `1ba6f175caed656729e0863880251cc315232b1c`, retained at
[`njzjz-bot/vibeqc: evidence/issue-231-baseline`](https://github.com/njzjz-bot/vibeqc/tree/evidence/issue-231-baseline).
The publication manifest binds both exact revisions and their fetchable source
repositories/refs. Both libraries used Release, CUDA 12.9.1, sm_120 and
`VIBEQC_CUDA_FAST_COMPILE=OFF` on an RTX 5090 in Slurm job 9179.

Five samples per source used the shared ABBA ordering. Each case has separate
cold (including preparation), unchanged, moved and restored geometry timings.
All 80 case/phase gates passed the existing 1.02 median-ratio ceiling. Maximum
energy error was 7.1055e-15 Hartree; maximum force error was 9.437e-15
Hartree/Bohr, against unchanged tolerances of 3e-10 and 3e-9. The worst median
ratio was 1.019427. These results support structural retirement without a
significant-speedup claim. The shared evidence API accepts the numerical scope;
its significant-performance and production stages remain `not-run`. The
separate non-regression assessments and retirement decision are explicit.

The inventory contains 16 s/p/f cases spanning RHF/UHF, Cartesian/spherical,
direct/DF and batches of one/three, plus four larger s/d/f cases: Cartesian RHF
direct batch one, spherical UHF direct batch three, and spherical RHF/UHF DF
batches one/three. Nineteen cases have explicit shared budgets. The 18-AO
Cartesian direct case preserves its legacy unbudgeted scope because public HF
inventory v1 supports at most 16 AOs; no total-budget guarantee is claimed for
that case. Every scope note, resource plan and observation is retained.

`samples.json` losslessly interns repeated input, build and resource records by
canonical hash. Its runs retain all individual timings, energies, forces,
residuals and iteration counts. `summary.json` retains the shared timing/noise
assessments for every case and phase. `evidence.json` contains every numerical
error block and provenance. `resources.json` reports native kernel resources:
generated thread/shell-warp schedules use 180/200 registers and 176-byte stacks,
with zero local/shared memory, unchanged by the generic-runtime extraction.
The historical handwritten value kernel used 146 registers, an 11,696-byte
stack and 1,024 bytes of shared memory. Resource counts alone do not imply a
speedup. Full build wall time was not captured.

`publication.json` binds all selected files by checksum and contains the
reproduction argv. Check out the exact measured source revisions and build
matching optimized libraries. Run the comparison through a finite Slurm
allocation with `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; preserve its
assigned device visibility. Publish a new result directory with:

```bash
python tools/publish_cuda_ownership.py \
  --comparison .artifacts/ownership-reproduction \
  --destination benchmarks/results/cuda-ownership/new-comparison
```

The publisher recomputes numerical and non-regression gates from the original
workers before calling the common publication API. The three kernel resource
JSON inputs are recorded under `.artifacts/231-*-resources.json`; copy the
retained entries there for an archival replay. Each record binds the worker
revision, native source identity, build settings, timing-library hash, object
hash/size and CUDA compiler identity. Publication rejects mismatched records.
The baseline records come from the original optimized objects. The candidate
resource record is explicitly an exact-source kernel reconstruction: its timed
linked library was replaced during retirement. Regenerating the scalar/policy
headers at `fdc7f40` and compiling `src/scf/cuda/one_electron_values.cu` with the
recorded Release/C++20/sm_120 flags reproduces every archived kernel symbol and
resource count. The record does not claim that reconstructed object was the
original timed binary. Use `cuobjdump --dump-resource-usage` on the matching
objects when collecting a new measurement. Publication never overwrites an
existing bundle. The CPU integrity tests reconstruct workers entirely from
retained data, reproduce all gates, and reject corrupted measurements.

The subsequent retirement removes the handwritten double-value kernel and
one-electron/DF value dispatches. Shared recurrences remain for separately owned
Dual derivatives and other integral domains. Independent Libcint checks remain
the scientific oracle; generated-schedule parity is labeled separately.
