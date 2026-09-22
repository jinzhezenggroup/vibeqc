# Decision: cut P86/PZ production to pinned Libxc Maple

Status: implemented
Date: 2026-09-21

This Agent D slice is stacked on #813 and performs the #744/#745 production
cutover and retirement for LDA_C_PZ and GGA_C_P86.

- Production lowers from pinned lda_c_pz.mpl / gga_c_p86.mpl.
- The handwritten pz_epsilon, pz_correlation, and p86_correlation bodies are deleted.
- No hidden handwritten fallback remains for these components.
- Adapter/importer/source provenance is bound into functional identity.
- Production E/vxc/fxc is gated against the independent PySCF 2.14.0 /
  Libxc 7.0.0 p86-hessian fixture.
- libxc_maple.py is not modified by this cutover slice.

References: #744, #745, #813.

Agent: ChatGPT
Model: GPT-5.6 Sol
