# Decision: retain GFN2 D4 outer storage and publication contracts

Status: implemented repair

## Problem

The shared molecular D4 primitive validates its own arguments, but its temporary
outputs are not the outer GFN2 caller outputs. It cannot detect caller gradients
aliasing positions, charges, geometry cache or canonical workspace. Publishing
one batch member at a time also changed the earlier all-member failure boundary.

## Decision

Restore outer address-range and finite-gradient admission before computation.
Stage complete batch gradients and energies in the existing canonical scratch,
validate final gradient sums, and publish only after every member succeeds.
The shared primitive receives one additional reusable 27*maximum-member-atom
scratch region reserved by D4Plan. Its bytes are included in the plan's canonical
workspace size; no evaluation-time allocation or caller scratch exemption is added.

## Tradeoff and evidence

This uses more planned scratch than overlaying the entire workspace, but keeps
pending outputs disjoint from subsequent member evaluation, including large CPU
members. The alternative of restoring a second handwritten D4 derivative was
rejected. Five actual-adapter storage/publication regressions fail before repair;
the valid-call control and the repaired cases pass. This is a host adapter repair,
not a new GPU schedule or full GFN2 endpoint qualification.

Agent: ChatGPT
Model: GPT-6 Astra Pro
