# Decision: shrink RI-MP2 J scratch before abandoning full B residency

Status: implemented
Date: 2026-09-21

The full-resident candidate previously tried only the preferred J batch. A valid
48-byte single-virtual example was rejected, while a 128-byte two-virtual example
unnecessarily entered the blocked path. Both can retain full B with J batch one.

Apply the existing bounded halving policy to full-resident scratch first, then
retain the existing blocked fallback. Emit the same choices through the compiler's
native-header generator. Python and actual generated-C++ counterexamples reject
or choose the wrong plan before repair and agree on the valid plan afterward.
No MP2 expression, integral cutoff, denominator rule or memory cap is changed.
These are planner tests, not a new molecular GPU or performance qualification.

Agent: ChatGPT
Model: GPT-6 Astra Pro
