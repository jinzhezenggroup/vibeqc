# Explicit reference qualification follow-up

The tolerance-only reference policy is **not qualified**. The historical failed
campaign remains unchanged. The first two exact 96-AO geometries contain failed
convergence controls despite tighter requested thresholds; the partial sweep
was stopped after those decisive failures, not relabeled as complete or passing.

`incremental-diagnostic.json` retains every completed scalar diagnostic, including
nonconverged repeats, incomplete setting counts, frozen-library identity and
hashes of full raw records retained on node3. Two earlier script attempts failed
at stock API calls before usable samples; their records remain in that manifest.
The two failed attempts to allocate a fresh full-Fock experiment (Slurm jobs
10026 and 10028) produced no molecular acceptance result. No passing reference policy is established by this directory.

The committed benchmark controls preserve historical defaults. An explicitly
qualified experiment can request, for example:

```sh
python benchmarks/real_molecule_gate.py --density-fitting none --repeats 7 \
  --reference-gradient-tolerance 96=1e-11 --reference-full-fock 96 \
  --output-directory <fresh-output-directory>
```

This is a command example, **not an accepted setting or a claim that it passes**.
Full-density rebuilding applies only to the selected stock reference arms; its
entire cost remains in endpoint timing. Native settings and numerical error
gates are unchanged. The 192-AO arms keep their historical reference settings.

## Completed work and remaining qualification

Benchmark controls are committed in `c2e93d347ca4203d2efff35016a47b74210912f9`.
Both host test files passed all 53 tests; applicable repository hooks passed.
Native production code and the measured library are unchanged. This follow-up
has not rerun the native test suite and does not imply a new native test result.

The next scientific step is the separately specified full-Fock stock sweep;
its cause hypothesis has not been verified on GPU. Only after that policy meets
the declared CPU-agreement and tightening-stability criteria may a complete
four-endpoint, seven-pair campaign qualify it. The exact prepared runners and all
full raw records remain in the node3 artifact directory recorded in the JSON.
No outstanding GPU job for this follow-up remains after the bounded allocation
failures. PR #427 remains draft; issue #206 remains open.

Agent: ChatGPT
Model: GPT-6 Astra Pro
