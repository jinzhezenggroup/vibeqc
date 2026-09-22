# Retained reference exporter import migration

Status: implemented
Date: 2026-09-22

## Problem

Removing compatibility imports changes exporter bytes even when mathematical
code and the canonical implementation objects are unchanged. Reference loaders
correctly rejected the stale source association. Refreshing historical numerical
values or claiming a new execution would conceal the actual provenance.

## Decision

Keep every retained reference/archive and its metadata identity unchanged.
A separate receipt binds six exact historical/current source SHA-256 pairs,
the specific import replacements, and a version-stable non-import AST hash.
Review verified old bytes against the original source commit and confirmed every
non-import AST node, annotation, docstring and literal was unchanged. The old
forwarders resolve to the canonical implementation used by the new imports.

Only test/reference source checks consult this exact receipt. Generic file
hashing, binary/artifact checks, numerical block hashes and tolerances remain
unchanged. Missing receipts, unknown source revisions or even an additional
unrecorded import are rejected. No AST-based runtime heuristic accepts edits.

## Rejected alternatives

Do not erase source checks, globally normalize file hashes, recreate removed
forwarding packages, or relabel old measurements as newly regenerated results.
The annotation-only source audit remains a historical record; its source check
uses this explicit subsequent migration rather than rewriting that record.

## Evidence

The core-CI failure had 195 test failures and 12 setup errors. Exact retained XC,
density-workload and annotation-audit checks fail before this repair. Tests also
exercise formula/import mutations, unknown hashes/paths and missing or invalid
receipts. Existing numeric fixture identities and all archives remain unchanged.

Agent: ChatGPT (Even-PR Review R8 pv1q9ewr)
Model: GPT-6 Astra Pro
