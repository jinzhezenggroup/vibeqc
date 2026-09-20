# Decision: unwind partial RCCSD owners and reserve simultaneously live buffers

Status: implemented repair
Date: 2026-09-20

## Failure ownership

A failed C++ constructor does not invoke its own destructor. The CUDA owner
therefore explicitly drains/releases any created stream and device allocation
before rethrowing a setup failure. Its device-scope member remains alive during
cleanup, then restores the original device during stack unwinding. Successful
owners use the same idempotent cleanup from their destructor. This preserves
original exceptions and never substitutes a CPU computation.

## Numeric capacity

CPU DIIS admission includes the current amplitude, trial, error and copied
history vector that can coexist before an old pair is removed. The Gram matrix,
original augmented system/RHS and solve_linear's by-value copies also coexist.
The conservative bound sums these capacities instead of counting only retained
history. Input vectors are charged by capacity rather than logical size.
CUDA admission additionally includes the final detached host T1/T2 result, which
is allocated before the resident owner is destroyed. Explicit budgets can now
reject an amount that previously omitted these buffers; that is not a tolerance
change or a different CC equation.

These are numeric-buffer bounds, not whole-process RSS or allocator/library/code
accounting. The separate MO/reference preparation capacity remains explicit.

## Regression evidence

The CPU test compiles the actual solver and generated equations and checks
zero-history and DIIS admission at exact/one-byte-short capacities, including
retained vector capacity. The CUDA-owner host harness compiles the production
constructor/destructor with injected CUDA API failures at all 19 setup steps,
retains device/handle counters, and verifies clean subsequent construction.
It also checks detached-result admission. These tests reproduce the respective
missing reservations and construction leak before repair. Host ASan/UBSan and
NVCC compilation are separate checks, not a claim of GPU memcheck execution.
Existing independent molecular-energy and expanded residual tests remain required.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Legacy descriptor admission

The optional correlation budget is read only after its full field is covered
by struct_size. Older ABI prefixes keep the default even when bytes beyond
the declared prefix contain an invalid budget. Present values are checked
against both signed-64-bit and size_t ranges before conversion. Two poisoned-
prefix regressions reproduce the prior unconditional-read rejection and now
pass without relying on a memory fault.
