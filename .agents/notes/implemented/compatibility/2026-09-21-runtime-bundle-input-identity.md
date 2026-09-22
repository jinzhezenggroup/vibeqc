# Runtime bundle inputs must match the published binary

Status: implemented
Date: 2026-09-21

## Problem

Bundle identity construction exhausted one-shot compiler options and library
iterables before invocation. A recorded `-DVALUE=19` artifact returned its default
value 3, and ordinary tuple replay reused that incorrect binary. Basename-only
source/header identities also aliased distinct local-header associations.

## Decision

Materialize invocation sequences once. Record source and explicit-header paths
relative to their common input root, preserving associations without introducing
installation prefixes. Use CPU bundle schema 2 so previously misidentified
schema-1 artifacts cannot be reused. Single-program cache schemas are unchanged.

## Evidence and limits

Three failure-first executable/argument checks fail before the repair. All 25
bundle, TensorIR CPU, and CC executor tests pass after it. Master's bounded
`max_nodes` option and this branch's explicit symbol both survive integration.
No complete CCSD(T) performance improvement or new GPU execution is claimed.

Agent: ChatGPT
Model: GPT-6 Astra Pro
