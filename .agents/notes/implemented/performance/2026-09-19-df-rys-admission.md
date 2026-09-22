# Decision: let the DF manifest own derivative lowering admission

Status: implemented
Date: 2026-09-19

## Problem

The shell consumer already has separate residency/layout/backend admission, and
its generated manifest separately qualifies each mathematical lowering by GPU
architecture and angular class. Nevertheless, `production_target()` replaced the
actual architecture with zero unless the orbital dimension was exactly 384 or
768 and the auxiliary dimension was equal. Thus the default resident 192-AO
shell consumer silently missed qualified cooperative Rys kernels.

The cooperative-Rys campaign had retained a positive 192-AO candidate without
promoting that dimension. This change supersedes only that note's final
shape-admission boundary, not its numerical evidence or historical decisions.

## Decision and invariants

Remove the redundant dimension test and its unused basis arguments from the
shared panel/group/packet target query. The generated manifest remains the sole
owner of architecture/class qualification. Unknown architectures, missing
classes (including auxiliary f classes), unqualified mappings and explicit
`legacy` controls retain the polynomial fallback. The `candidate` control still
admits an experimental manifest; it does not make unqualified entries automatic.

Do not change the integral generator, mathematical expressions, precision,
metric cutoff/rank, force definition, final-state checks, allocations or leases.
Do not broaden the gradient bridge's separate shell/packet/packed-layout rules
in this slice. In particular, small default generic consumers stay generic.

## Rejected alternatives

Adding 192 to the whitelist would leave the same artificial cliff at other
sizes and unequal auxiliary dimensions. Repeating the old 384/768 Rys speedup
would count an already delivered optimization. Enabling every shell or packet
consumer at once would combine independent policy changes without attribution.
No new tuning framework or handwritten scientific kernel is required here.

## Evidence

The [retained campaign](../../../../benchmarks/results/issue445-df-rys-admission/README.md)
uses a fresh optimized sm_120 library, one finite Slurm allocation, independent
complete-force references, seven interleaved pairs from one frozen density,
untimed priming at each transition, and separately instrumented diagnostics.
The Direct four-center AOT bundle is disabled in both arms; generated DF kernels
remain enabled at Release `-O3`, without fast-compile flags.

At 192 AO, the baseline's `auto`/`candidate` medians are 142.749912/92.375393 ms:
35.29% less complete energy-plus-force time. Every replay takes three SCF
updates. Operator calls, logical shell/primitive/component work, response
products, transfers and charged storage match. The maximum force difference
from the independent reference is 3.19e-12 Hartree/Bohr against the unchanged
1e-8 gate. A separate intrusive force-stage measurement falls from
102.966205 to 56.245295 ms; it is not substituted for clean timing.

The 384/768 controls already select these kernels, and their differences are
below 0.04%. The unchanged generic 96-AO control has a 1.77% median difference
amid monotonic startup drift: retain it as inconclusive, not a kernel regression
or speedup. No fresh cross-engine superiority, batch-4, constrained-memory or
practical-JKFIT latency claim is made by this campaign.

An executed pre-fix regression fails on `shell_000_rys_selected == 0`, after its
independent energy/force checks pass. The new regression exercises auto ->
legacy -> auto across panel/group/packet consumers, equal 192 AO, unequal
cc-pVDZ/JKFIT and non-water UHF. Existing scientific tests also cover full
forces, original/tighter requests, cold/warm/changed geometry and low budgets.
The campaign README records the completed test counts and final-binary check.

## Consequences and revisit criteria

The numerical kernel is chosen by capability instead of a benchmark dimension.
This does not predict latency on an unmeasured architecture, nor qualify an
unavailable class. New mappings still require the existing independent numerical
and complete-endpoint evidence. Revisit the fallback if a new architecture/class
is qualified, and handle separate consumer/packet selection under #445/#444.

The 768-AO ledger assigns about 89 ms to the exclusive pseudo-density product
region out of a 1.115-s complete force stage. Eliminating only that named region
would not meet #419's 0.15-s endpoint or 10% force-stage gate. A fusion experiment
must also remove other producer/consumer work and measure the combined endpoint;
this observation is not an upper bound on all possible fusion savings.

## References

- #445, #444, #435 (remaining consumer/admission work); #419 (separate fusion).
- [Original cooperative Rys qualification](2026-09-17-cooperative-df-rys.md).
- [Current selectors](../../../../docs/developer/df_tuning.md).

Agent: ChatGPT
Model: GPT-6 Astra Pro
