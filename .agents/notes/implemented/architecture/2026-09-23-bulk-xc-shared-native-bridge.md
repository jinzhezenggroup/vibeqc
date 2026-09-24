# Bulk Libxc shared native pointwise bridge

Date: 2026-09-23
Issue: #1119

## Decision

Automatic Libxc registrations may enter the compiler-owned shared semilocal
native lowerer only through an explicit **pointwise representation** bridge.
This bridge does not imply production-domain admission, molecular SCF, forces,
response, or public-method capability.

The first slice admits the already imported rho-only LDA and rho/sigma GGA
feature prefixes to the same scalar Graph -> derivative roots -> C/CUDA source
owner used by curated semilocal functionals.  Ordinary runtime
`build_program()` and the default native lowerer continue to reject every
`AUTO_BULK_COMPONENTS` registration until independent production-domain
evidence is attached by the admission lane.

Tau meta-GGA bulk registrations remain explicitly blocked in this slice because
the bulk inventory ABI currently retains Laplacian slots even when the Libxc
registration flags require tau but not Laplacian.  A follow-up #1119 slice must
project that imported Graph onto the canonical rho/sigma/tau ABI without
changing the pinned mathematical identity.

CPU and CUDA wrappers share the same expression identity; backend function
qualifiers remain outside the mathematical hash.

Agent: ChatGPT
Model: GPT-5.6 Sol
