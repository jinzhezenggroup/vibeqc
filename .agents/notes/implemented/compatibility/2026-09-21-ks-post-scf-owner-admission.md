# Decision: reject unowned geometry corrections in the electronic KS projection

Status: implemented
Date: 2026-09-21

A backend-neutral execution plan can describe post-SCF corrections, but that does
not provide a native electronic executor for them. Reject such corrections before
returning electronic coefficients/options, rather than silently computing a
smaller Hamiltonian. The named PBE-D4 ABI retains its independently qualified
native correction owner through an explicit family-selector projection; this is
not a blanket exception for arbitrary geometry terms.

Four direct RKS/UKS coefficients/options counterexamples fail before repair and
pass afterward. Existing named-selector rejection, append-only v4 ABI and public
PBE-D4 tests remain enabled. No energy formula, tolerance, or native ABI layout is
changed. The Calculator-level D3 owner remains a separate composition boundary.

Agent: ChatGPT
Model: GPT-6 Astra Pro
