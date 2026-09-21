# Decision: preserve signed reduction indices and reserved scalar identity

Status: implemented
Date: 2026-09-21

The bounded Maple add frontend substituted a negative index without parentheses.
For example, add(i^2,i=-1..1) was translated into -1^2 + 0^2 + 1^2 and evaluated
to zero instead of two. Parenthesized literal substitution preserves precedence.

K_FACTOR_C, MU_GE and DBL_EPSILON already had intrinsic semantics, but source
assignments and external bindings could redefine them and were then silently
ignored. Reserve these constants at source/binding admission; ordinary lexical
function parameters can still shadow them. No additional Maple syntax is admitted.

Advance the importer semantic identity to v6. Eight failure-first regression
cases reproduce the errors before the repair; the complete selected Maple/XC
suite passes 228 tests afterward without skips, including independent SCAN and
r2SCAN E/vxc/fxc fixtures. Ruff, focused ty and compiler dependency checks pass.
No production XC source or numerical threshold changes, and no GPU result is
inferred from these host/compiler checks.

Refs: #769, #742.

Agent: ChatGPT
Model: GPT-6 Astra Pro
