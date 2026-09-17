# Post-#418 stock comparison: #206 remains open

The merged cooperative DF Rys path reduces the observed 768-AO warm complete
energy-and-force median below stock GPU4PySCF in this campaign. The 384-AO
complete endpoint and both large energy-only endpoints remain slower. The
original 96-AO batch-4 force gate fails on **all seven pairs**. This evidence
keeps #206 open; it does not establish general DF superiority.

## Pinned experiment

Slurm job 9902 ran ten points with seven interleaved, synchronized repeats per
engine on one RTX 5090, using a finite 45-minute allocation and preserving
Slurm's device visibility. The controller completed with exit 1 because of the
retained numerical failure; independent remaining points completed normally.
No failed point was rerun to obtain a pass. The process preflight records only
the two known orphan Nsight control agents, with no active build or profiler.

Measured source: `42060ca0dddb29c1c3b2f03fc1ee933d0e4945f2`, subsequently merged
by #428 as `9af9e08b4a79b6d7f562eb1e20fe84ce58c85a7f`. Scientific source and
benchmark drivers are identical between these commits. Frozen native identity:
`06fa1f9d7108f86edfaa4ba9285df58c20a1d191b3c20785a2500cad1798730d`.
Library SHA-256:
`ea39fae62486240a98021726d970ec80c965e8f4a687056e59636a7179faf5be`.
The build/source reconstruction is retained here and in
[`../issue418-cooperative-rys/`](../issue418-cooperative-rys/README.md).

Stock versions are GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 14.2.0, cuTENSOR 2.3.1
and NumPy 2.5.3. Separate per-geometry/batch provider audits verify the installed
cuTENSOR engine, explicit physical basis data and matching full metric ranks.
The stock implementation's normal Cholesky policy is preserved. These cells
use def2-SVP for both orbital and auxiliary spaces; they do not represent a
practical unequal-size JKFIT auxiliary basis.

Native controls select dense value storage, automatic shell policy, device
final-state validation, existing automatic exchange/projection policies and
disabled force screening. Automatic cooperative admission remains limited to
sm_120 and equal-auxiliary 384/768 AO. The 192-AO candidate result from #418 is
not selected here.

Each engine restores its own immutable post-cold converged density outside the
warm timer and receives untimed priming. No cross-engine density byte identity
is asserted. All legitimate convergence branches remain visible. Energy and
density tolerances are 1e-12 and 1e-10; screening remains 1e-14. The original
96/192 fixtures supply their unchanged numerical and stock gradient gates.
Large and holdout cells use 1e-9 Eh / 1e-8 Eh/Bohr error gates and a 1e-10 stock
gradient tolerance. Every paired energy/force array is checked independently.

## Ordinary warm observations

Times are median seconds over all seven samples. F includes energy and complete
analytic forces; E is energy only. Failed timing is retained without admission.

| Case / batch | Observable | VibeQC | Stock | All seven numerical pairs |
| --- | --- | ---: | ---: | --- |
| Water 96 / 1 | F | 0.120780 | 0.265592 | pass |
| Water 96 / 4 | F | 0.466341 | 1.053825 | **fail** |
| Water 192 / 1 | F | 0.165849 | 0.350926 | pass |
| Water 192 / 4 | F | 0.654052 | 1.396228 | pass |
| Water 384 / 1 | E | 0.296658 | 0.137303 | pass |
| Water 384 / 1 | F | 0.634143 | 0.614895 | pass |
| Water 768 / 1 | E | 1.015666 | 0.615875 | pass |
| Water 768 / 1 | F | 2.005771 | 2.140275 | pass |
| OH UHF 19 / 1 | F | 0.007741 | 0.242166 | pass |
| NH3 29 / 1 | F | 0.008845 | 0.236034 | pass |

The 768-force ordinary median is 6.28% lower than stock (stock/native 1.067).
Relative median absolute deviations are 0.249% native and 1.203% stock. Native
uses three iterations in all seven samples; stock uses one in five samples,
two in one, and three in one. These are ordinary converged solve latencies,
not fixed-work kernel measurements. The comparator's matching-branch subset
contains only **one stock sample versus seven native samples**; its 1.274 ratio
is not used as a robust performance claim. The analogous 384-energy subset
also has only one stock sample. No favorable branch is selected for the table.

This controller did not predeclare a noise/superiority test. Reported dispersion
is descriptive; no post-hoc threshold is used to claim acceptance. The
384-force median remains 3.13% slower than stock. Neither the internal #418
speedup nor the 768 ordinary observation establishes 384/768 superiority.

The 96/b4 maximum force difference is **3.128523135e-11 Eh/Bohr**, exceeding
the unchanged **3e-11** gate in every pair. The 96/b1 maximum is 2.971758777e-11
and passes. The maximum 192/b1 and b4 force differences are 2.070516258e-10
and 2.121617890e-10, within their unchanged 5e-10 gate. The earlier convergence
sensitivity diagnosis does not waive the original failed acceptance or replace
these samples with tighter-reference timings.

## Remaining acceptance

This campaign retains the original small DF cells, explicit large energy/force
cells, UHF and a non-water holdout. Exact native reservation diagnostics and
metric ranks are present, but complete stock/native operator, transfer and
resource accounting is still missing. Cold observations are single setup
observations, not a repeated cold comparison. No changed-geometry rebuild or
practical unequal auxiliary basis is measured here. No new combined ablation
or direct-path acceptance is established.

The earlier #421 DF failures and authoritative #427 direct campaign, Slurm
9890, remain unresolved and are not overwritten. #427 remains draft while its
original direct numerical gates fail. #412's negative split-Gram result is
unchanged. Further #206 work must preserve these failures and original gates.

## Audit and reproduction

`manifest.json` records 52 losslessly retained input/output, identity and
reproduction records with both original and stored hashes. Every result,
progress record, failure, provider audit and exact controller command is
retained; the native binary remains local. Gzip payloads restore exact bytes.
Slurm accounting storage is disabled on this host; the controller transcript
retains `srun`'s task exit 1 and the session's observed completion is recorded.

Recompute all 70 numerical pairs, 140 timing samples, branch counts, full-rank
checks and medians without a GPU:

```bash
python benchmarks/results/issue206-post418/reproduction/analyze.py \
  benchmarks/results/issue206-post418/raw /tmp/post418-summary.json
cmp benchmarks/results/issue206-post418/summary.json /tmp/post418-summary.json
```

The archived controller/qualifier are exact original scripts. To reproduce on
the recorded stack, restore their source paths from `manifest.json`, use the
pinned measured checkout and matching library, and run through:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:45:00 /tmp/vibeqc-308-gpu4pyscf-env/bin/python \
  .artifacts/issue206-post418/run-fresh-matrix.py
```

The controller deliberately refuses an existing result directory. Any new
experiment must use a separate location and preserve this failed campaign.
