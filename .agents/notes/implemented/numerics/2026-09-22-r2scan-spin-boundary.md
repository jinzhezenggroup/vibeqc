# Decision: r2scan spin boundary

Status: implemented
Date: 2026-09-22

## Decision

Use scalar domains and boundary continuations to keep generated r2SCAN value and first derivatives finite when a minority-spin density vanishes. CPU and CUDA emitters consume the same expression policy.

## Invariants and rejected alternatives

Do not mask a nonfinite result after evaluation or change the interior functional to make a boundary test pass. Preserve independent Libxc comparisons and zero-spin derivative conventions.

## Evidence and remaining qualification

CPU expression, Maple adapter and generated-C++ boundary tests pass. RTX 5090 FP64 boundary qualification and coordination with #1040 remain merge gates.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
