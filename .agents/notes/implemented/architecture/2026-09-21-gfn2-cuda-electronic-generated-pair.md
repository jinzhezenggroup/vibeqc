# GFN2 CUDA electronic pair cutover

The production GFN2 CUDA Hamiltonian now consumes compiler-generated
fixed-state pair science from the #505 TensorIR contract.

## Ownership boundary

The generated graph owns, for one canonical matrix pair:

- scalar-overlap Hamiltonian shift;
- directed dipole and quadrupole Hamiltonian shifts;
- reverse-mode S/D/Q integral adjoints;
- the unrestricted spin-only overlap adjoint.

The CUDA runtime continues to own ragged topology, validation, error
publication, stream ordering, spin charge/magnetization layout conversion,
scratch storage, and SCC orchestration.
## Build integration

`tools/generate_gfn2_electronic_cuda.py` lowers a scalar runtime-bound
TensorIR graph to a generated CUDA header.  Build-time codegen is intentionally
NumPy-free, matching the existing GFN2 pair generator contract.

`gfn2_hamiltonian.cu` and `gfn2_hamiltonian_force.cu` gather runtime
topology values and call the generated primal/VJP helpers.  They no longer own
the -1/2 S/D/Q scientific equations.

ES2, ES3, AES2, H0-force and integral construction are separate scientific
owners and are not claimed by the #505 graph.

## Qualification

- #505 full-graph equivalence tests cover primal and reverse-mode adjoints.
- The CUDA 12.9 SM120 fast-compile target `vibeqc_gfn2_cuda` builds through
  device link and archive creation with the generated header.
- Production runtime source tests prevent reintroduction of the retired
  handwritten pair formulas.

Agent: ChatGPT
Model: GPT-5.6 Sol
