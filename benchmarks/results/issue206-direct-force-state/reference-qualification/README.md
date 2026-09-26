# Explicit reference qualification follow-up

**The full-Fock reference policy is qualified; native integration acceptance is
not established by this reference-only experiment.** Historical tolerance-only
and paired-matrix failures remain failures.

## Completed full-Fock GPU experiment

Slurm job 10035 completed four exact 96-AO geometries, thresholds 1e-9, 1e-10,
1e-11 and 1e-12, and three repeats: all 48 samples are retained in
[full-fock-diagnostic.json](full-fock-diagnostic.json). The stock GPU4PySCF
provider used `direct_scf=False`; no CPU replacement or native change was used
in this experiment. CPU calculations remain an independent accuracy oracle.

The 1e-11 setting passes the predeclared 3e-12 Eh/bohr qualification margin:

- Maximum CPU-oracle force error at 1e-11: 2.311040248059726e-12.
- Maximum CPU-oracle force error in the 1e-12 controls: 2.864375403532904e-12.
- Maximum force change under tightening: 9.49906819869284e-13.
- Every 1e-11 and 1e-12 sample converged; looser settings remain unqualified.

The record retains all scalar observations, exact geometries, CPU oracle forces,
package/source identities, and hashes of full force and frozen-density arrays.
The complete raw report remains at its recorded ignored node3 path and hash.
[sweep-full-fock.py.txt](sweep-full-fock.py.txt) preserves the exact executed
runner bytes; its historical input paths must be restored before replay.

## Historical failures

[incremental-diagnostic.json](incremental-diagnostic.json) retains every
completed scalar sample from the earlier tolerance-only sweep, including
nonconverged repeats and incomplete setting counts. Its 30 completed samples
failed qualification; they are not relabeled by the new full-Fock experiment.
The initial full-Fock allocation attempts 10026/10028 produced no molecular
result. All historical raw records remain separate from job 10035.

## Complete endpoint qualification

Use the qualified reference only in an explicitly labeled new campaign, after
building and validating its native source:

```sh
python benchmarks/real_molecule_gate.py --density-fitting none --repeats 7 \
  --reference-gradient-tolerance 96=1e-11 --reference-full-fock 96 \
  --output-directory .artifacts/pr427-matrix-fresh
```

Set `VIBEQC_LIBRARY` to that campaign's verified production Release/AOT library.
All four endpoints and seven interleaved pairs remain mandatory. Complete stock
endpoint timing includes the extra full-Fock/convergence work. The 192-AO
reference settings and every native, iteration, energy, force and speed gate
remain unchanged. Defaults have not changed.

The master-integration CPU library passed 230 targeted host regressions. The
production CUDA build and its native tests/full paired matrix are separate
qualification steps; neither historical 49-test native results nor this
reference-only pass authorizes a merge. Issue #206 remains open.

Agent: ChatGPT
Model: GPT-6 Astra Pro
