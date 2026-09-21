# Decision: carry MethodIR range exchange through KS options v6

Status: implemented transport slice
Date: 2026-09-22

## Context

The common MethodIR already represents WB97M-V as semilocal B97M plus explicit
short-/long-range exact exchange and VV10. Native KS options v5 could carry the
nonlocal primitive but stopped before the range-separated exchange pair, forcing
the live endpoint to lose part of the resolved scientific graph.

## Decision

Append a v6 KS-options suffix carrying the physical short-range fraction,
long-range fraction, and shared omega. Python derives these fields from the
compiler-owned KS execution plan. No method name is used as a scientific
dispatch key.

The native parser validates and snapshots the v6 suffix but deliberately fails
closed until the existing two-Fock range-separated consumer is attached. This
prevents a dangerous intermediate state in which RSH metadata is accepted but
silently omitted from the Fock build.

All v1-v5 struct extents remain valid physical prefixes. Protected-page tests
cover the new v5 boundary and full v6 object.

## WB97M-V closure boundary

This slice is necessary but not sufficient for public WB97M-V. Remaining gates
are the native RSH consumer binding, a production-qualified B97M semilocal
tail/attenuation policy, combined RSH + VV10 self-consistent execution, and
independent end-to-end energy/force validation. Public wb97m-v therefore
remains reserved.

Refs #167 #491 #396

Agent: ChatGPT
Model: GPT-5.6 Sol
