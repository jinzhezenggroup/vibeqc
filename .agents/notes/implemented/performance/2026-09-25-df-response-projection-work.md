# Decision: explicit DF response work and scratch census

Status: implemented
Date: 2026-09-25

## Problem

The full-rank bounded response revisits every fitted Q block for every P block.
With b auxiliary blocks this means b*b fitted-reader calls and b passes over
all auxiliary columns. A memory bound alone hides this work amplification.
The source-only occupied producer separately uses one raw auxiliary column at
a time although an already charged projection interval is dead at that phase.

## Decision

Keep checked, host-testable work and scratch contracts in one small shared
header. The first contract counts the existing general-density fallback;
the second bounds an occupied projection batch inside an explicitly supplied
reusable interval. The caller must prove that the interval is dead and disjoint
from retained factors. Pair-major sources charge a transpose panel explicitly.
This is resource/census infrastructure for the following stacked implementation,
not a new scientific driver, universal optimizer or automatic promotion.

## Invariants

- No numerical cutoff, density identity or final-state check changes.
- No uncharged storage, overflow wraparound or inferred whole-device capacity.
- Retained B, DIIS history and other future-live warm state are not reclaimed.
- The default runtime is unchanged in this preparatory commit.

## Evidence

A host C++20 test compiles with -Wall -Wextra -Werror and passes. It covers
5/8-block practical-shape counts, zero workspace, exact and one-element-below
budgets, ragged dimensions, zero/full ranks and overflow/invalid input.
No CUDA build, device execution or wall-time improvement is claimed.

## References

#1078, #1334, #445; response audit comment 5832638027.

Agent: ChatGPT
Model: GPT-6 Astra Pro
