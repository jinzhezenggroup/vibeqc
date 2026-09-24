# DF changed-geometry reference protocol correction

Status: implemented
Date: 2026-09-22

The issue206 rebuild runner now includes the stock reference geometry reset
inside the synchronized changed-endpoint timer, matching the native coordinate
update boundary. Only the original-geometry prime remains outside both timers.
Cold and warm/changed reference solves must converge before analytic gradients
are accepted; finite energies alone do not establish stationarity.

The evidence schema is version 2. Existing version-1 measured JSON is preserved
byte-for-byte and is not relabeled as if it included stock geometry reset.
Two executable failure-first protocol regressions cover reset-time inclusion
and nonconvergence rejection before gradient execution. No timing data was
fabricated and no scientific error threshold changed.

Mainline integration also preserves per-reason DF fallback diagnostics while
retaining this branch's full resident generated-source eligibility. This is
not a replacement for complete current-source GPU qualification.

Agent: ChatGPT (Even-PR Review a3kkcri7)
Model: GPT-6 Astra Pro
