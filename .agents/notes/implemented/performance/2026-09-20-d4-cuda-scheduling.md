# D4 block-cooperative CUDA scheduling

Issue #493 retained a CUDA-correct D4 implementation in which one CUDA lane owned
one molecule. This slice keeps that evaluator as the oracle and adds a bounded
fixed-charge scheduler with one 256-thread block per ragged molecule.

The scheduler parallelizes coordination-number pairs, two-body dispersion pairs,
ATM triples, and the final coordination response. It uses the existing 27*N
workspace contract, does not materialize pair/triple tensors, accepts an optional
per-system active mask, and zero-publishes failed/inactive members so one bad
molecule cannot poison its peers.

A real-device parity test compares the cooperative path with the scalar oracle
and covers unequal ragged members, an empty member, an inactive member, and a
NaN-poisoned member. On an RTX 5090 with CUDA 13.0, a non-isolated micro-probe comparing
against the retained one-lane CUDA baseline measured:

| batch | scalar lane | cooperative | speedup |
| --- | ---: | ---: | ---: |
| 4 x 8 atoms | 1.133 ms | 0.075 ms | 15.2x |
| 4 x 16 atoms | 8.409 ms | 0.133 ms | 63.0x |
| 4 x 32 atoms | 60.251 ms | 0.570 ms | 105.8x |
| 4 x 64 atoms | 560.060 ms | 8.780 ms | 63.8x |
| 4 x 96 atoms | 4305.238 ms | 42.791 ms | 100.6x |

These numbers were collected while another GPU process was resident and are a
scheduler micro-probe only, not an isolated or end-to-end DFT-D4 performance claim. The complete EEQ charge/response path is still the scalar-per-molecule
qualification route and remains separate from this fixed-charge scheduling slice.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Shared partition admission

Offsets are shared ownership metadata, not a molecule-local numerical input. A
nonmonotone partition can produce overlapping workspace/output intervals even
when a later local interval is ordered. Clearing an invalid member can then race
with a valid neighbor's publication. One linear, allocation-free validation
kernel now checks the complete partition before any molecule work. Invalid
partitions return invalid_argument for every member and keep zero outputs.
Admission failures do not dereference their untrusted intervals during final
publication. A valid partition still isolates inactive/invalid/numerically failed
molecules independently. The 27*N arena and all scientific expressions remain
unchanged; the extra validation launch is not advertised as a speedup.

The host regression executes extracted production guards with explicit serial
CUDA stubs. The CUDA regression additionally covers four malformed partitions
and restoration of the original valid partition on the same buffers.

Agent: ChatGPT
Model: GPT-6 Astra Pro
