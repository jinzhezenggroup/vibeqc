# Decision: share the DF contract without importing its dense oracle

Status: implemented
Date: 2026-09-21

The factorized RCCSD solver previously imported its method contract from the
dense DF oracle. Removing the oracle re-exports from the facade alone did not
remove that transitive production dependency. The canonical immutable contract
and reference relabeling helper now live in `df_contract`; the oracle imports
those exact objects for compatibility. Neither the method identity payload nor
the scientific validation rules change.

Keeping an oracle import behind a renamed facade or relaxing the provenance
check was rejected. A clean-process regression checks the actual module graph;
an identity test checks the shared objects, and the dense/factorized numerical
suites retain independent validation coverage.

Agent: ChatGPT
Model: GPT-6 Astra Pro
