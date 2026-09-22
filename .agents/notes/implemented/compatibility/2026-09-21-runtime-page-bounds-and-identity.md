# Decision: bound axis iteration and identify ordered page contents

Status: implemented
Date: 2026-09-21

## Problem

The rectangular iterator used itertools.product, which pools whole input axes.
A two-coordinate page over (200000, 3) allocated 7,993,777 traced bytes before
publication. Thus page capacity did not bound the iterator's retained state.
The independently constructible RuntimeTaskPage also hashed only its position
and count, allowing different accepted coordinate sequences to share an identity.

## Decision

Advance rectangular coordinates using an O(rank) mixed-radix counter. Preserve
row-major ordering, exact logical counts, consumer legality and tail accounting.
Include the actual ordered coordinates in the page identity. Domain identities
remain unchanged; newly introduced page identities deliberately gain the missing
work-content dependency. No scientific equation or generated tensor graph changes.

## Evidence

Two failure-first tests cover bounded first-page allocation and reordered page
identity. Existing domain/triples/stationary tests retain independent sequence
and work-count checks. These host checks are not new GPU or endpoint speed claims.

Agent: ChatGPT
Model: GPT-6 Astra Pro
