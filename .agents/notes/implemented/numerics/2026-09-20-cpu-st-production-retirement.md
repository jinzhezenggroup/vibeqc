# Decision: retire handwritten s/p/d/f CPU S/T from the production path

Status: implemented
Date: 2026-09-20
Issue: #351

## Decision

Overlap and kinetic (S/T) are the first operator family for which production CPU and CUDA now consume one compiler-owned mathematical specification. The existing one-electron value/derivative IR remains the scientific source of truth; the CPU build adds a host lowering from those same DAGs and emits a generated header for all Cartesian s/p/d/f component pairs. No second recurrence IR or integral generator is introduced.

`build_integrals`, cross-overlap, and streamed weighted one-electron derivatives use the generated S/T path for s/p/d/f. The old dynamic-Jet S/T recurrence is renamed and classified as reference code: `posthf::RawSource` continues to use it as a structurally independent oracle. Angular momentum g and above explicitly use that reference as a compatibility fallback until the generated production domain expands; it is not an unclassified fallback for the promoted s/p/d/f domain. Nuclear-attraction, ERI, and other host families are outside this retirement slice.

## Independent numerical gates

The native Cartesian test now compares the promoted s/d/f production overlap and hcore matrices element-by-element with `RawSource`, whose S/T recurrence is structurally independent. It also compares every nuclear-coordinate overlap/hcore derivative with central finite differences of that independent source. The same native gate also exercises the classified g fallback; it caught and fixed a compact component-key collision before publication. Existing Cartesian/spherical/basis tests pass. Existing compiler one-electron tests continue to validate generated value/derivative DAGs against independent Libcint/PySCF references.

## Build/resource and endpoint evidence

Retained raw evidence is `benchmarks/results/issue351-cpu-st-retirement.json`; the complete endpoint reproducer is `benchmarks/issue351_cpu_st_retirement.py`.

On node3 (AMD EPYC 7K62, one hardware thread per core), clean CPU-only Release/Ninja `-j2` builds measured 42.96 s / 317,692 KiB max RSS on master `d32caac` and 52.74 s / 413,564 KiB after promotion. The generated S/T header is 2,596,536 bytes; the shared library grows from 1,734,912 to 2,334,496 bytes. There is no new persistent runtime workspace. The change adds zero handwritten scientific recurrence LOC: the existing S/T recurrence is retained and reclassified as oracle/fallback, while all promoted scientific arithmetic is generated from the shared DAG.

For a complete 18-AO Cartesian s/d/f He-H+ RHF energy+force endpoint, 12 complete calculator executions per library (first two excluded from the median) measured 10.8843679 s on master and 11.1502246 s after promotion, a 2.44% median regression. This is recorded as a cost, not a speedup. Energy differs by 2.5e-15 Ha and the maximum force difference is below 4e-16 Ha/Bohr; both runs converge in eight iterations.

## Ownership consequence

The supported s/p/d/f CPU production path no longer owns a handwritten S/T recurrence. CUDA and CPU source their S/T mathematics from the same compiler DAG, while the independent host recurrence remains deliberately separate for validation. Generated source size, maintained-source delta, compile/resource cost, runtime endpoint effect, and the retained oracle boundary are therefore explicit before retirement.

Agent: ChatGPT
Model: GPT-5.6 Sol
