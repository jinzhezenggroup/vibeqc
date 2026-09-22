# Issue #206 completion audit

This directory retains compact summaries for the current completion audit. The
raw CUDA/component traces remain in the ignored n5 artifact directory recorded
in `summary.json`; the JSON here is the reviewable, identity-pinned result.

Current native identity:

- source identity: `47d4c417aaa8f3ecaf61e62d96f9c0c54b5900cb20d99938abccac764d68dff3`
- library SHA-256: `8d40c060e65dd0f3bb78209149e91de7da2b68f85ed16d349c58650b0365ad65`
- target: NVIDIA GeForce RTX 5090, sm_120, CUDA 12.9.1

## Acceptance mapping

| Requirement | Current evidence | Result |
| --- | --- | --- |
| 96/192-AO DF energy+force parity and warm endpoint matrix | `summary.json`, current 7/5-repeat endpoint records | Pass |
| 384-AO larger DF endpoint | `summary.json` | Numerical pass; ordinary warm speedup 1.34x |
| 768-AO larger DF endpoint | `summary.json` | Numerical pass; ordinary warm ratio 1.23x, so not a no-slower claim |
| Resident response admission and transfer/work invariants | `resident-768.json` | Pass: resident JK scratch, zero raw panel gather, zero raw upload, packed exact work |
| Intentional constrained/fallback route | `fallback-192.json` | Pass: explicit host-weight control is classified as fallback and completes |
| Direct analytic-force floor | `direct-current.json` | Pass for the retained four-point 7-repeat floor |
| Unequal-auxiliary practical cases | Historical raw records used an older native identity and are not counted as current evidence | Not counted |
| Current unequal-auxiliary practical cases | `practical-current.json` | Pass: cc-pVDZ/cc-pVDZ-JKFIT water RHF 96/464 and OH UHF 19/93, five complete warm pairs each |
| Current practical cold/changed endpoints | `rebuild-practical-current.json` | Warm gates pass; both five-repeat changed-geometry force gates remain open by small margins |
| Current equal-basis cold/warm/changed matrix | `rebuild-equal-current.json` | Pass: current 96/192-AO batch-1/4 cells, five repeats, all warm and changed energy/force gates pass |
| Current host preparation/provider/state diagnostics | `fixed-work-ablations-current.json` | #309/#310/#311 controls retained; requested derivative-only and Gram-only fixed-work arms remain open |

This audit does not close #206: the 768-AO endpoint remains slower than the
stock comparator, the 384-AO automatic streamed low-memory experiment did not
complete within its finite 30-minute allocation, and the practical unequal-auxiliary
changed-geometry force gates remain open. Those negative results are retained in
`summary.json` rather than omitted.

## Fixed-work terminology

The host records are diagnostic controls for preparation, eigensolver-provider,
and verified-final-state work. They are not interchangeable with the issue's
derivative/Gram fixed-work arms:

| Requested arm | Current evidence | Disposition |
| --- | --- | --- |
| Baseline | `rebuild-equal-current.json` plus host baseline selections | Retained as ordinary endpoint/control evidence, not an iteration-matched cross-engine fixed-work claim |
| Derivative-only | No isolated current #394/#404 arm in this audit | Open; earlier promoted derivative work is not relabeled here |
| Gram-only | No isolated current #412 arm; rejected split-GRAM evidence remains separate | Open; no rejected candidate is counted as a saving |
| Combined | `combined-host-cross-workstream` in `fixed-work-ablations-current.json` | Host-only #309/#310/#311 composition, not derivative-plus-Gram fixed work or external parity |

The raw host JSON files live under the n5 artifact root recorded in
`summary.json`; each compact record includes the raw SHA-256 and current
library identity.

The resident sentinel and timeline runner are exercised by the hardware-free
tests `tests/python/test_issue206_resident_sentinel.py` and
`tests/python/test_issue308_response_timeline.py`.

## Rebuild protocol correction

The retained rebuild records above use protocol version 1. That runner excluded
the stock reference geometry reset from its changed-endpoint timer. Version 2
includes that reset and rejects nonconverged reference solves before gradient
execution. Historical raw records are unchanged and are not version-2 timing
evidence. See the retained protocol-correction Agent Note.
