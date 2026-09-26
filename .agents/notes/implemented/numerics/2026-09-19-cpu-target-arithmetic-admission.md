# Decision: CPU target flags describe an ISA, not a new arithmetic policy

Status: implemented
Date: 2026-09-19

## Problem

A custom CpuTargetInfo could append -ffast-math after the supposedly strict
-ffp-contract=off option. In a reproduced generic ssss candidate, this erased
native finite-input checks: a NaN primitive exponent returned success and thirteen
NaN value/derivative outputs. Recording the flags in an artifact hash did not
qualify the arithmetic or make that result safe.

## Decision

The initial scalar/AVX2/AVX-512 target contract accepts exactly the ISA options
belonging to its declared lane width. Extra arithmetic flags, response files,
implicit -march=native, missing ISA flags and width/flag mismatches are rejected.
Lane widths must be actual integers, not booleans or equal-valued floats. Keep
FMA as the existing explicit scientific schedule choice. Do not change generic
compiler adapters or existing scalar/libm/Boys arithmetic.

## Evidence and consequences

Eleven malformed-target/schedule regressions fail before repair and pass after
it. A strict compiled scalar candidate separately rejects the NaN primitive.
The scalar/AVX2/AVX-512 compile, tail and independent value/derivative tests remain
in the same suite. New target flags or alternate arithmetic require an explicit
versioned contract and independent qualification, not merely a new cache key.
No historical benchmark sample or numerical tolerance is changed.

Agent: ChatGPT
Model: GPT-6 Astra Pro
