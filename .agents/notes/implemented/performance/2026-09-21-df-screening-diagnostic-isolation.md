# DF screening diagnostics: isolation and resource admission

Status: implemented
Date: 2026-09-21

The intrusive screening histogram is not a production screening threshold. Every
endpoint wrapper now scopes its environment control: cold, prime and clean calls
disable inherited screening; only the requested diagnostic enables it. Scope
restoration runs on failures and remains outside measured endpoint time.

The diagnostic response panel uses exactly sized reusable host storage charged to
the existing response allowance before allocation or D2H submission. A drained
previous panel is freed before growth, preventing an unbudgeted two-panel peak.
The normal D2H/synchronization counters include the diagnostic overhead in addition
to the dedicated counters. Insufficient headroom fails; it does not retile the
scientific work and thereby confound timing attribution.

The report reader validates all six distance/exponent histogram axes, bin widths,
strict nonnegative integer counts and conservation, rather than accepting the
headline primitive count as proof of valid bins. Existing source-work and weight
conservation gates remain intact.

Failure-first host tests exercise the real endpoint wrapper, corrupt report bins
and the actual C++ staging helper at exact/one-byte-short boundaries. Scientific
formulas, thresholds and derivative CUDA kernels are unchanged.

Agent: ChatGPT (Even-PR Review)
Model: GPT-6 Astra Pro
