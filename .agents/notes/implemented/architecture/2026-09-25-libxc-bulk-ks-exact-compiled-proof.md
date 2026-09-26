# Decision: require exact compiled-CPU proof before bulk KS resolution

Status: implemented
Date: 2026-09-25

## Problem

The generic bulk-Libxc KS resolver already required the `compiled-cpu` stage
name, but stage admission alone did not prove that the attached qualification
payload was the canonical result for the exact generated point binding. A
malformed or stale pass envelope could therefore satisfy the stage DAG while the
resolver lost the artifact identity it was supposed to execute.

This becomes more important after production-candidate/v2: domain string, native
domain version 3, pinned density threshold, generated wrapper, executable and
interior+vacuum smoke are one execution contract.

## Decision

Add a strict `validate_qualification()` boundary to the compiled-CPU evidence
owner and make bulk KS resolution consume it after ordinary stage resolution.

The validator requires:

- the current compiled-CPU qualification schema;
- exact functional capability identity;
- canonical v4 point-binding payload;
- production-candidate/v2 and its native domain version;
- exact pinned density threshold;
- binding/result identities;
- compiler, translation-unit and executable hashes; and
- the passing interior+vacuum smoke contract.

`BulkKsResolution` advances to v2 and retains the exact compiled point-binding
identity and compiled result identity. Descriptive method names remain outside
scientific plan identity.

## Rejected alternatives

- Trusting `status=pass` plus the stage name would leave artifact provenance
  unchecked at the molecular boundary.
- Reconstructing a new point binding from only the functional name would discard
  the evidence-produced executable identity.
- Making molecular-SCF evidence a prerequisite for the candidate resolver would
  recreate the original evidence cycle.

## Invariants

- Candidate KS resolution requires compiled-CPU and production-domain stages.
- The compiled-CPU stage must also contain a canonical exact qualification.
- Molecular-SCF remains produced by executing the candidate; it is not required
  to construct that candidate.
- KS resolution retains the exact binary/binding provenance needed by the next
  evidence producer.
- No public method is promoted by this resolver change.

## Evidence

Focused tests cover canonical qualification validation, threshold tampering,
malformed qualification rejection, candidate-cycle preservation, and retention
of exact binding/result identities in the KS resolution payload.

Repository CI is the executable validation authority for this stacked slice.

## Consequences

The next molecular-SCF evidence layer can key its receipt to a
`BulkKsResolution` that already names the exact compiled artifact, rather than
combining an abstract MethodIR plan with an unrelated stage label.

## Revisit when

A future artifact registry supplies a stronger shared executable handle/receipt
that subsumes the current compiled-CPU qualification payload.

## References

- #1119
- #1121
- #1234
- #1333
- #1337

Agent: ChatGPT
Model: GPT-5.6 Sol
