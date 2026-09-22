# Decision: cut production LYP over to the pinned Libxc Maple source

Status: implemented
Date: 2026-09-21

## Problem

#744 requires family-by-family production cutover only after independent
qualification, and #745 requires the handwritten formula to be retired once
that family is cut over. LYP qualification landed in #809, but production still
constructed GGA_C_LYP from the handwritten expression in rsh_expressions.py.

## Decision

Make GGA_C_LYP production construction call the already-qualified pinned
Libxc 7.0.0 gga_c_lyp.mpl representation through a small production adapter.
The adapter binds the default parameters from the pinned gga_c_lyp.c owner
and reuses the common fail-closed Maple frontend.

Delete the handwritten LYP formula rather than keeping a silent fallback.
Record adapter bytes, importer bytes, importer semantics, entry-source hash and
transitive source hash in the functional identity whenever GGA_C_LYP is active.

## Gates

- #809 already qualified imported-vs-handwritten E/vxc/fxc parity.
- Retained independent PySCF 2.14.0 / Libxc 7.0.0 Hessian fixtures continue to
  gate both spin layouts.
- Production build_program is compared directly against the imported graph
  through the packed feature Hessian.
- Existing Scalar C/CUDA emitter qualification remains active.
- B88, VWN/VWN-RPA and P86/PZ are not cut over here; their importer slices are
  still independently reviewable.

References: #744, #745, #809.

Agent: ChatGPT
Model: GPT-5.6 Sol
