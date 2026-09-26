# Decision: Select one DF response representation at the owner handoff

Status: implemented
Date: 2026-09-26

## Problem

The factorized-response prototype extended source-owned occupied projection to
retained value plans, but still lent their fitted/raw tensor views. The bridge
correctly rejected this mutually exclusive combination before execution.

## Decision

After the canonical occupied factor passes the existing token/density checks,
withdraw both tensor views from the bridge call. The occupied path regenerates
bounded physical raw-source panels and owns its projection scratch. No value-plan
allocation is freed or repurposed by withdrawing these views.

Retained plans select this source-owned path only for an explicit occupied-space
request. Automatic selection retains the existing fitted-B path and its work
contract; streamed plans continue selecting source-owned projection automatically.
Invalid factors preserve the existing retained views and bounded fallback.

## Evidence

The compiled handoff regression exercises the actual selection block with retained
and streamed plans, explicit/automatic/dense requests, borrowed scratch, and valid
or invalid tokens. It checks the bridge's mutual-exclusion condition and preservation
of retained views on fallbacks. Existing algebra tests independently compare the
factorized packed exchange weights with materialized matrix products.
