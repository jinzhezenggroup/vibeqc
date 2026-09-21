# #438 practical JKFIT operator provenance

This slice fills trace/provenance gaps needed before changing SCF or final-state
work policy. It does not weaken convergence and does not claim an endpoint win.

- compact RHF reports whether dense or occupied exchange is selected;
- final RHF K reports the requested policy, selected provider, and a concrete
  fallback category instead of making dense fallback indistinguishable;
- force response explicitly reports final-projection reconstruction as well as
  reuse;
- the bounded DF SCF probe publishes native Fock-build count;
- issue438_practical_operator_ledger.py joins complete progress, CUDA component,
  and host eigensolve journals while keeping graph construction, CUDA timings,
  host timings, logical updates, and clean endpoint latency distinct.

The next evidence run must use current master for practical 96/464 and 192/928,
retain equal controls, and keep clean endpoint timing separate from tracing.

Agent: ChatGPT
Model: GPT-5.6 Sol
