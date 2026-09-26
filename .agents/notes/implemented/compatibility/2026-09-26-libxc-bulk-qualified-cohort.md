# Decision: Keep split-hybrid extensions out of the bulk import cohort

Status: implemented
Date: 2026-09-26

## Problem

Split-hybrid work extended the shared Maple frontend and admitted const-double parameter layouts. Offline bulk regeneration then promoted 27 registrations from 221 to 248, although independent E/vxc/fxc fixtures still covered only the original 221. The core CI regeneration test failed, and accepting the larger catalog without a reference would misrepresent qualification.

## Decision

Retain the bulk importer's non-const-double layout policy and block the newly supported split-hybrid-only Maple helper closures from the bulk inventory. Preserve their pinned blocker reasons so catalog and report regenerate reproducibly. The r++SCAN source imports rSCAN transitively and must also remain blocked; unsupported BR89 definitions take precedence over their helper references. Split-hybrid extraction continues admitting its const-double layouts under its independently validated contract.

## Revisit when

Promote additional bulk entries only with independent two-spin Libxc E/vxc/fxc fixtures, audited parameter bindings and updated inventory counts. Another consumer's point/endpoint evidence does not qualify bulk derivatives.

## Evidence

`python tools/import_libxc_bulk.py --check` reproduces 221 imported and 323 blocked registrations. The metadata, catalog and source-registry tests cover deterministic products and provenance.
