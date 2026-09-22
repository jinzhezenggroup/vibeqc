# Preserve scaling before GFN2 electronic pair products

The CUDA cutover formed an unscaled sum of scalar potentials and unscaled
multipole products before applying -1/2. Finite inputs could overflow there
although the previous native scaled operations and final result remained finite.

The typed primal now scales each integral before multiplying each potential.
Reverse-mode adjoints remain generated from that primal. No handwritten VJP,
new cutoff, relaxed finite-input gate, or output tolerance is introduced.
Four compiled emitted-helper regressions cover scalar addition, overlap,
dipole and quadrupole products. This preserves finite arithmetic range;
it is not a full molecular CUDA endpoint or performance promotion.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro
