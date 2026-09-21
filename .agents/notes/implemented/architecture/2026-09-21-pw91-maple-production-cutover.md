# Decision: cut PW91 production mathematics to pinned Libxc Maple

Status: implemented
Date: 2026-09-21

GGA_X_PW91 and GGA_C_PW91 were already qualified through the common
fail-closed Libxc Maple frontend, but production still rebuilt the same
mathematics from handwritten formulas in rsh_expressions.py.

Production now delegates those two components to pw91_maple.py. The adapter
owns only VibeQC feature/spin packing plus the pinned source call boundary;
Libxc 7.0.0 Maple owns the PW91 mathematical definition. The old handwritten
PW91 functions remain temporarily in rsh_expressions.py but are unreachable
from production dispatch and are reserved for deletion under #745.

The functional identity records importer semantics, importer bytes, adapter
bytes, and entry/transitive source hashes. No public method capability is
promoted by this source-of-mathematics cutover.

Agent: ChatGPT
Model: GPT-5.6 Sol

Validation on the cutover branch:
- focused PW91 importer/family/Hessian suite plus cutover assertions: 38 passed;
- broader PW91/MethodIR/XC/source-registry suite: 228 passed with the CPU native library;
- H2 fixed-density total and XC energies are unchanged from the pre-cutover master;
- H2 Fock maximum absolute difference is 2.78e-17;
- full polarized PW91 order-2 graph: 1945 -> 1916 reachable nodes;
- generated scalar/CUDA source: 81974 -> 82009 bytes (effectively flat);
- median warm build_program time on node3: 251.9 -> 245.9 ms.

No numerical tolerance was weakened and no runtime Libxc dependency was introduced.
