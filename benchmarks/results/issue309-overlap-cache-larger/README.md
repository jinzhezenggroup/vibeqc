# #309 larger preparation ablations

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

Five interleaved pairs for each candidate and endpoint on a clean source-bound
RTX 5090 library, using a finite Slurm allocation. Both domains use RHF,
spherical def2-SVP and the same orbital auxiliary basis, a 1 GiB DF allowance,
and frozen post-cold warm densities. Baseline restores eager core guesses and
overlap decomposition; candidates separate the two preparation optimizations.

| AO / batch | Candidate | Clean warm endpoint | Baseline (s) | Candidate (s) | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 96 / 4 | lazy-core | energy | 0.921091 | 0.626548 | 1.470 |
| 96 / 4 | lazy-core | force | 2.549948 | 2.246968 | 1.135 |
| 96 / 4 | overlap-cache | energy | 0.921236 | 0.619651 | 1.487 |
| 96 / 4 | overlap-cache | force | 2.541576 | 2.239687 | 1.135 |
| 96 / 4 | combined | energy | 0.921163 | 0.325583 | 2.829 |
| 96 / 4 | combined | force | 2.532724 | 1.954343 | 1.296 |
| 192 / 1 | lazy-core | energy | 3.258497 | 2.209151 | 1.475 |
| 192 / 1 | lazy-core | force | 6.383553 | 5.331633 | 1.197 |
| 192 / 1 | overlap-cache | energy | 3.271309 | 2.160213 | 1.514 |
| 192 / 1 | overlap-cache | force | 6.389121 | 5.279518 | 1.210 |
| 192 / 1 | combined | energy | 3.264697 | 1.110122 | 2.941 |
| 192 / 1 | combined | force | 6.388515 | 4.215996 | 1.515 |

All SCF iteration/retry branches match. Energy and complete-force replay gates
pass at 1e-9 Eh and 1e-8 Eh/Bohr against separately prepared cold endpoints.
Separate intrusive traces verify every requested core/overlap solve and cache
scope, including a changed last item. Every selection still executes one
reference final-Fock solve per RHF item; no fallback solve was observed. Raw
cold construction/destruction, unchanged replay and changed-geometry samples,
including restored original geometry between changed samples, remain retained.
Profiled timings are not used in this table.

This expands the first 96-AO/batch-1 record; it does not establish independent
GPU4PySCF parity or close #206/#308/#309. The 192-AO/batch-4, 384-AO/batch-1/4,
constrained-memory and other spin/representation timing domains remain open.
Required provider integration and verified final-state reuse remain #310/#311.

The standard archive retains every result/manifest, original JSONL trace text
and reproduction script. Archive and member hashes are recorded, and every
member was restored and compared byte for byte. The native scientific source
identity and library hash match the earlier validated #314 binary. This
additional evidence does not replace its native, molecular or memcheck tests.

```bash
python -m tools.unpack_evidence benchmarks/results/issue309-overlap-cache-larger \
  --output /tmp/issue309-cache-larger-evidence
```

Rebuild the manifest source with CUDA 12.9.1, Release, architecture 120 and AOT
disabled as in the baseline recipe. The restored `reproduction.sh` records all
18 runner invocations. Adjust its checkout/interpreter and use fresh output
paths, then run through `srun --partition=main --gres=gpu:5090:1 --nodes=1
--ntasks=1 --time=01:00:00 bash <script>`, preserving scheduler visibility.
