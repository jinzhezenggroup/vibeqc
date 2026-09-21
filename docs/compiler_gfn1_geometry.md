# GFN1 geometry compiler

Issue #837 keeps GFN1-xTB scientific equations compiler-owned whenever they
are pure functions of immutable parameters and geometry.

## Delivered pairwise slice

`python/vibeqc_compiler/geometry/gfn1.py` lowers two GFN1 geometry terms
through the existing GeometryIR / PairIR / TensorIR stack:

- exponential covalent coordination numbers (`k = 16`, inclusive 25-bohr
  cutoff, pinned GFN1 radii);
- effective nuclear repulsion
  `Z_i Z_j exp(-sqrt(a_i a_j) r^(3/2)) / r`.

Both primal quantities and Cartesian derivatives come from the same TensorIR
graph. Coordinate derivatives use generated reverse-mode VJPs rather than a
copied handwritten force implementation. The graph lowers through the shared
CPU interpreter and CUDA TensorIR planner/emitter.

Numerical element data are not handwritten into the compiler module.
`tools/parameters/generate_gfn1_geometry.py` extracts the exact required
subset from the canonical GFN1 snapshot and emits `_gfn1_data.py`; the source
registry binds the generator, input identity, and generated output hash.

## Independent qualification

The compiler graph is checked against pinned independent GFN1 references:

- the MB16-43 structure-01 coordination vector evaluated independently from
  the mctc-lib exponential-count definition and GFN1 radii;
- tblite's GFN1 effective-repulsion golden energy;
- xTBloom's independently implemented pair-force goldens;
- multi-step centered finite differences for generated CN and repulsion VJPs;
- translation conservation, cutoff/minimum-distance topology semantics, and
  CUDA source lowering.

CPU-vs-CUDA agreement is not used as the sole correctness oracle.

## Halogen correction boundary

GFN1's halogen correction is not pairwise: it depends on a
donor-neighbor-acceptor angular geometry. This slice deliberately does not
encode it as a fake pair primitive or introduce a handwritten production
kernel. The MethodIR already identifies `halogen_correction`; its compiler
lowering requires an explicit triplet/angle topology extension to GeometryIR.

The existing compiler-owned D3(BJ) path remains the GFN1 dispersion owner; no
second GFN1 D3 implementation or data table is introduced here.

## Runtime boundary

This module does not implement SCC iteration, mixing, eigensolution,
occupations, warm starts, or public method admission. Those remain runtime
policy under #837.

Agent: ChatGPT
Model: GPT-5.6 Sol
