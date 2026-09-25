# Decision: separate bulk KS qualification execution from promoted resolution

Status: implemented
Date: 2026-09-25

## Problem

The first bulk-KS resolver required `molecular-scf` evidence before it would
construct the MethodIR/KS plan. That stage is itself produced by running the
plan after compiled-CPU and production-domain qualification, so the requirement
made the evidence DAG circular. Separately, CPU RKS/UKS selected semilocal
mathematics through curated method-family branches, leaving no native consumer
boundary for an evidence-bound generated bulk XC artifact.

## Decision

Keep ordinary `resolve_bulk_ks` fail-closed on the full promoted stage set, but
add `resolve_bulk_ks_candidate` for qualification producers. The candidate
requires exactly `compiled-cpu` and `production-domain`; it never grants
`molecular-scf` or public-method capability.

Expose a CPU semilocal point-program descriptor containing a generated
expression identity, ingredient mask, domain version and function pointer.
RKS/UKS qualification entry points consume that descriptor through the existing
SCF iteration loops and common AO/grid rho/sigma/tau contraction. Curated method
entry points remain unchanged.

## Rejected alternatives

- Treat synthetic `molecular-scf` evidence as a prerequisite. That would make
  the evidence record self-authorizing rather than execution-derived.
- Map bulk functionals onto the legacy LDA/PBE/r2SCAN family codes. Different
  Libxc mathematics must not inherit a curated family identity or numerical
  policy.
- Add one SCF driver per Libxc functional. The point-program identity is the
  scientific boundary; the SCF loop is method-neutral within the admitted
  semilocal ingredient domain.

## Invariants

- Candidate resolution cannot satisfy or imply `molecular-scf`.
- Final bulk-KS resolution still requires retained molecular endpoint evidence.
- A point-program descriptor is executable plumbing, not production admission.
- Runtime dispatch does not branch on a functional name to choose scientific
  mathematics.
- The same RKS/UKS convergence and physical-state rules remain in force.

## Evidence

Python tests lock the two-stage candidate versus three-stage promoted resolver
contract. Native tests execute the generic CPU RKS and UKS entries with the
existing generated PW91 semilocal program, rebuild endpoint components, check
closed-shell RKS/UKS agreement, and reject an unbound expression identity.
PW91 is used only as an already-generated regression program; this change does
not claim bulk production admission for it or any AUTO_BULK_COMPONENTS entry.

## Consequences

#1119 can bind a qualified bulk AOT artifact to this point-program descriptor,
then #1121's qualification runner can execute it to create molecular-SCF
evidence without a circular prerequisite. #1120 remains the authority for
production-domain evidence, and public method promotion remains separate.

## Revisit when

Replace the function-pointer packaging boundary only if the compiler/runtime
artifact registry adopts a different stable ABI shared by CPU and CUDA.

## References

- #1119 generic bulk XC runtime
- #1120 production-domain qualification
- #1121 generic KS integration
- #1140 first evidence-gated bulk KS resolver

Agent: ChatGPT
Model: GPT-5.6 Sol
