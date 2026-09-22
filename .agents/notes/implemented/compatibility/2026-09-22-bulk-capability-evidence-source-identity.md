# Bind bulk capability evidence to scientific and compiler sources

Status: implemented
Date: 2026-09-22

## Problem

Registration labels do not identify an implemented functional. Changed parameter
bindings, source digests, upstream identity, binding interpretation or compiler
bytes retained the same capability ID and accepted old compiled-stage evidence.

## Decision and invariants

Bind the scientific registration, pinned source/upstream identity and the shared
common/integral/XC compiler source inventory. Use stable logical source paths
for wheels and checkouts. Bulk queries hash the inventory once without lowering
numerical graphs or loading native/GPU/reference code. Conservative source
invalidation is intentional; old higher-stage evidence is never transferred.

## Evidence

Five admission regressions fail before repair and pass afterward. The complete
bulk/metadata/capability suite passes 496 tests. Stage dependencies, numerical
reference fixtures and the public-method manifest remain unchanged.

Agent: ChatGPT (All-PR Review)
Model: GPT-6 Astra Pro
