# 96-atom preparation diagnostics and independent references

> **Checkout retention (2026-09-21):** `qualification.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue308-large-diagnostics/qualification.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue308-large-diagnostics/qualification.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

This slice retains completed diagnostics for 32 waters / 96 atoms / 768
spherical AOs with def2-SVP orbital and auxiliary bases. There is still no
converged VibeQC energy or full-force endpoint at this size. The synthetic
geometry is a 2x2 array of translated WATER27 S4 octamers, not an optimized 32-mer.

After the capture repair in #324, the 28-GiB resident plan still did not finish
cold energy after more than 15 minutes (Slurm 9521). The 100-second follow-up
(9526) sampled GPU waiting in resident three-center materialization at both
30 and 60 seconds, before SCF. The GPU stayed near full utilization with about
14,680–14,782 MiB allocated. The requested CUDA trace did not appear because its first
operation never completed. These are failed/terminated diagnostics, not endpoints.

Independent PySCF RHF DF energy and complete analytic forces converged with
energy tolerance 1e-12, gradient tolerance 1e-10 and auxiliary-basis response.
The force run energy is -2438.354141054296 Hartree and the maximum net force is
7.05e-13 Hartree/Bohr. Its checkpoint, density, overlap and force arrays are
retained for later physical comparison. CPU timing is not a GPU speed comparison.

Slurm 9529 held the first 24 AO-pair by 24 auxiliary outputs fixed while
growing the surrounding basis from 24 to 192 to 768 AOs. Values are bit-identical
for each mapping. Default primitive-mapping medians were 5.677, 5.724 and
5.881 ms across ten post-priming repeats. Dense zero-coefficient scans add
some cost but do not explain the observed large preparation delay by themselves.
The compiler reports 254 registers per value-kernel thread. Nsight Compute
could not obtain hardware counters (9530: ERR_NVGPUCTRPERM), so that report
alone does not establish an occupancy bottleneck. No source-layout change is
promoted from these diagnostics.

The qualification archive retains #323's passing mixed-UHF/provider/export
checks and #326's host-DIIS retry checks, alongside earlier failures and their
raw traces. Current memchecks report zero errors and zero leaks. The manifest
records exact source/library identities, scopes and counts. All archive members
were restored and compared byte-for-byte; original incomplete result files
remain unchanged. This slice does not close #308, #310 or #206.

## Archive storage correction

`reference.zip` was removed from the current tree when restoring the hard
1 MiB file limit. Its exact bytes remain in commit `daa2da0867877c94c40f379ffe1f3db6e3036ef8`; the
[storage migration](../retention-size-limit/migration.json) pins its SHA-256
and size. Existing numerical conclusions and measured identities are unchanged.
Restore the historical archive to an ignored working directory with:

```bash
python tools/restore_retained_evidence.py benchmarks/results/issue308-large-diagnostics/reference.zip
```

Archive restoration is only needed for historical raw-run inspection. New runs
keep full logs, profiles and retries outside Git.
