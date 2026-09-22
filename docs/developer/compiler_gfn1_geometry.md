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

## Halogen TripletIR lowering

GFN1's halogen correction is not pairwise. The compiler builds role-ordered
`(nearest-neighbor, halogen-donor, acceptor)` triplets, with the donor as the
angle center, and lowers one pinned scalar expression through TripletIR and
TensorIR. The donor elements are Cl/Br/I/At; the acceptors are N/O/P/S. A
donor-acceptor pair is admitted at the inclusive 20-bohr cutoff. The donor's
nearest positive-distance atom is selected with tblite's strict comparison,
which makes the lowest atom index the deterministic tie break.

The angular factor is `[0.5 (1 - cos(neighbor-donor-acceptor))]^6`. The radial
factor uses the pinned GFN1 atomic radii, radius scale 1.3, donor `xbond`
strength and damping 0.44 with exponents 6 and 12. The same scalar graph
provides CPU interpretation, generated Cartesian JVP/VJP and shared CUDA
lowering. `build_gfn1_halogen_program` binds this graph to the canonical
`halogen_correction` primitive and semantic identity of a resolved GFN1
XtbMethodIR.

If the nearest neighbor is the acceptor, the pinned angular factor and its
first derivative are identically zero. That entry is omitted because generic
TripletIR requires three distinct atom roles. Coincident donor/acceptor
coordinates are undefined by the pinned radial equation and fail closed.

Topology is discrete compiler state. Moving across the 20-bohr cutoff,
changing the selected nearest neighbor, or changing its deterministic tie
break requires a rebuild. Cartesian derivatives are defined only inside one
fixed topology region; no derivative is claimed at those boundaries.

Pinned tblite energy goldens for Br2-NH3, Br2-OCH2 and FI-NCH provide an
independent oracle. The generated VJP is additionally checked against an
independent scalar transcription at several centered finite-difference step
sizes, plus rotation/translation/permutation covariance. CPU/CUDA agreement
is a backend gate, not the sole scientific oracle.

The existing compiler-owned D3(BJ) path remains the GFN1 dispersion owner; no
second GFN1 D3 implementation or data table is introduced here.

## Runtime boundary

This module does not implement SCC iteration, mixing, eigensolution,
occupations, warm starts, or public method admission. Those remain runtime
policy under #837.

Agent: ChatGPT
Model: GPT-5.6 Sol
