# Keep the selected resident DF response source alive

Status: implemented
Date: 2026-09-22

## Problem

The automatic resource policy can choose a materialized resident DF owner while
retaining a positive resolved value budget. The old post-upload cleanup treated
that positive number as evidence of source-backed tile regeneration and released
`raw.three_center` before force response. The 96-AO automatic CUDA endpoint then
converged for energy but rejected the complete force request with invalid response
dimensions. The public rejection was correct: no raw owner remained to consume.

## Decision

Preserve the already-materialized raw allocation when both response geometry
owners are bound. Keep its exact pointer identity across replay. Source-backed
metadata starts with an empty raw array and stays empty; energy-only records can
still release their uploaded host values. Uploaded transformed tensors and unused
explicit derivative tensors remain releasable in either route.

The repair changes no provider selection, raw-tensor allocation size, integral
arithmetic, convergence rule, source geometry check, force tolerance or explicit
positive budget. The retained raw vector remains visible to existing capacity
accounting. This is not permission to regenerate missing values through an
undocumented CPU fallback or to accept an empty span as a materialized tensor.

## Evidence

The exact source-retirement helper compiled against the native data type loses
the raw response owner before repair. The regression now verifies resident
pointer/extent preservation over repeated cleanup, transformed/derivative release,
energy-only release and empty source-backed behavior. The original allocated-GPU
96-AO failure was independently reproduced before this change. Subsequent endpoint
validation belongs in the associated review; this note does not relabel an
unexecuted large-scale or timing gate as passed.

Refs #970 #439.

Agent: ChatGPT (Even-PR Review R8 v29lt7wo)
Model: GPT-6 Astra Pro
