# Preserve B3LYP through the shared native KS execution plan

Status: implemented
Date: 2026-09-21

## Decision

Integrate the B3LYP CPU family into the existing compiler-resolved KS plan, rather
than restoring the retired method-ID dispatch in prepared native owners. Family 3
has explicit spin, full-range exchange, CPU-only admission and SCF domain 2. The
legacy public selector maps to that plan only at the compatibility boundary.

Generic LYP remains sourced from pinned Maple. Its already-qualified B3LYP
production-tail continuation stays explicitly separate; dropping it would change
fully polarized derivatives, while restoring the general handwritten LYP fallback
would undo the canonical-source cutover. Range-separated/CUDA capability is not
inherited from the new family.

## Evidence

A fresh Release CPU library and native DFT API test pass. Four independent B3LYP
RKS/UKS reference/native force tests pass, including the public endpoint and
reconverged finite differences. The integration initially exposed an incorrect
XC-name lookup; it was corrected to use the canonical MethodIR resolver before
those tests were rerun. No GPU qualification or tolerance relaxation is claimed.

Agent: ChatGPT
Model: GPT-6 Astra Pro
