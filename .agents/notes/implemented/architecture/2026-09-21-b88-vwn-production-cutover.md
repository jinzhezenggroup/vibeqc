# Decision: cut B88 and VWN production to pinned Libxc Maple

Status: implemented
Date: 2026-09-21

## Scope

This is the Agent D production/retirement slice stacked on the qualified
B88 + VWN importer branch (#778/#782).

- GGA_X_B88 production now lowers from gga_x_b88.mpl.
- LDA_C_VWN and LDA_C_VWN_RPA production now lower from their pinned Maple
  entry points and shared vwn.mpl include.
- The standalone handwritten VWN/VWN-RPA bodies are deleted.
- The full-range handwritten B88 production path is deleted.
- The B88 enhancement mathematics needed by GGA_X_ITYH is retained only inside
  the explicitly named ITYH path until that range-separated component is
  independently imported and qualified.
- Adapter/importer/source provenance is bound into functional identity.

The stacked PR deliberately does not modify libxc_maple.py.

References: #744, #745, #778, #782.

Agent: ChatGPT
Model: GPT-5.6 Sol
