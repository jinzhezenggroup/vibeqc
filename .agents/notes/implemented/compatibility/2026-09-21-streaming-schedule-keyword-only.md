# Decision: make reduction-streaming opt-in keyword-only

Status: implemented
Date: 2026-09-21

## Problem

Inserting stream_reductions before direct_gemm in TensorSchedule and
TensorScheduleSpace reinterpreted legacy positional arguments. In particular,
an established direct_gemm=True call unexpectedly enabled the qualification-only
streaming path, despite the retained negative endpoint performance evidence.

## Decision

Make only the new streaming fields keyword-only. Existing positional arguments
retain their original meaning, and explicit stream_reductions keyword calls
continue to work. Serialized schedule fields, search-axis order, numerical
lowering and default disabled streaming policy are unchanged.

## Evidence

Four old positional-constructor cases fail before the repair and pass after it.
The combined planner/search/triples/compatibility suite passes 175 cases.
These are host tests, not a new GPU timing or performance-promotion campaign.
The negative measured streaming results remain unchanged and default search
still excludes streaming.

Agent: ChatGPT
Model: GPT-6 Astra Pro
