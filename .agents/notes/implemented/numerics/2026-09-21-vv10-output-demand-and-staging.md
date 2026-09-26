# Decision: preserve VV10 output demand and asynchronous host staging

Status: implemented repair
Date: 2026-09-21

## Problem

An energy-only call still allocated feature outputs at the C boundary, forcing
local feature derivatives and their finite checks. At positive density 1e-70 and
zero gradient, the existing independent energy oracle remains finite while the
unrequested rho^5 derivative denominator underflows. Both variants failed.
The CUDA wrapper also declared its host download vector after the stream and
arena owners, allowing exception unwinding to free it before queued D2H drained.

## Decision

Propagate optional feature outputs through C staging, CPU scales and CUDA
allocation/accumulation. Preserve geometry requests independently. Retain the
existing worst-case resource admission and strictly positive density domain;
do not invent a cutoff, clip density, or relax requested-derivative validation.
Declare local host download destinations before stream/arena owners so teardown
completes queued work before those destinations expire, without a new normal-path
synchronization.

## Evidence

Two energy-only reference regressions and five compiled-host asynchronous fault
cases fail before repair. The success fault-control also remains enabled. These
host fault cases use explicit CUDA stubs, not device numerical emulation.

Agent: ChatGPT
Model: GPT-6 Astra Pro
