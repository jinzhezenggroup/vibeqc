# Final-state and rejected combined ablations at 192 AOs

> **Checkout retention (2026-09-21):** `endpoints.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue206-final-state-192/endpoints.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue206-final-state-192/endpoints.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

Slurm job 9503 ran the WATER27 S4 water octamer: 24 atoms, 192 spherical
AOs, def2-SVP orbital/auxiliary basis, RHF, batch one, resident DF and FP64.
The finite allocation used one RTX 5090 on `main`. Source commit and native
library hashes are pinned in the manifest. No compilation or substantial
CPU work overlapped the clean timing window.

Four final-state invocations passed: separate traced/clean energy/full-force
comparisons, each with five interleaved pairs for cold, warm, output-selected
and changed-geometry workloads. Warm energy medians were 91.899 ms with forced
ordinary-device rebuilding and 46.009 ms with verified retention. The effect
passes the 2%/noise gate. Warm complete force medians were 3.463 s and 3.425 s;
that difference is not significant under the declared gate. Cold and changed
geometry improvements are also not significant. All raw samples remain.

All four combined-host invocations failed the exact SCF iteration/retry
branch gate. Their logs, manifests and raw traces are retained alongside the
passing runs. They support no combined speedup claim. The original runner
had not written endpoint rows before rejecting; those rows cannot be recovered
from host timing traces. The accompanying diagnostic change retains such rows
in a separate rejected artifact, preserving the failure and withholding timing
promotion. It does not loosen a numerical or branch threshold.

The archive was restored and every member compared byte-for-byte. This is
native work attribution, not an independent scientific oracle or matched
GPU4PySCF parity result. Larger dimensions, batch four and constrained memory
remain separate acceptance work.
