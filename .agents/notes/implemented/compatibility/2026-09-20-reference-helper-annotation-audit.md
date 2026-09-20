# Decision: concrete reference-helper annotations without new numerical references

Status: implemented source-contract repair
Date: 2026-09-20

The remaining helper exemptions are removed without replacing known interfaces
with blanket Any: CLI destinations are Path, names and hashes are str, commands
without a return use None, numerical arrays use ndarray, and manifests use
explicit mappings. Dynamic PySCF objects and heterogeneous JSON payloads retain
their honest dynamic boundaries. Annotation imports must not introduce a new
runtime PySCF dependency.

Reference metadata currently binds generator bytes. Changing annotations alters
those bytes but must not masquerade as a fresh scientific reference run.
`tests/reference_data/reference_source_annotation_audit.json` retains the exact
before-review and after-review checksums for all eleven helpers, the reviewed
source commit and executable-AST equivalence fingerprints. Here `original`
means the source at that recorded pre-review commit, not a newly inferred
historical numerical producer. The same audit records which metadata files
were rebound to the compatible annotated source.

The review compared executable ASTs after removing only annotations, type
comments and typing-only imports; numerical expressions, calls and literals
were retained. Reference arrays, mathematical inputs, library identities and
numerical thresholds were not regenerated. Existing provenance identities were
recomputed only where source metadata changed. The recorded AST fingerprints
are a same-interpreter review aid, not a new cross-version cache/ABI contract.

`test_reference_annotation_contracts.py` checks the exact annotated source bytes
and the evident CLI/name contracts without executing reference generators.
The independent existing fixture/source-integrity tests remain authoritative.
No newly computed energy, derivative, reference timestamp or external-library
measurement is claimed by this source-only audit.

Reject removing provenance checks, blindly accepting changed numeric arrays,
or restoring blanket ANN/TC exemptions. Revisit this process if a helper's
executable AST changes: that needs a genuine reference regeneration or an
independent scientific equivalence campaign, not an annotation-only rebind.

Agent: ChatGPT
Model: GPT-6 Astra Pro
