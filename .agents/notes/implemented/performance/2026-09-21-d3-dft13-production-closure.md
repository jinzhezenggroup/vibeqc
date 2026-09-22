# D3 DFT13 production closure evidence (issue #492)

Date: 2026-09-21

## Scope and capability boundary

Issue #492 closes the production D3 correction surface without changing the electronic
SCF/Fock equations. Calculator-level composition is owned by the prepared Calculator
boundary merged in PR #823: the electronic owner and retained D3 owner see the same
accepted geometry, the correction is applied exactly once, and the public result keeps
the dispersion component separate from the total energy/forces.

The public D3 capability matrix is explicit and non-transitive:

- `d3.bj-two-body`: BJ damping with `s9=0`.
- `d3.bj-atm`: BJ two-body plus the separately qualified ATM contribution.
- `d3.zero-two-body`: zero-damping two-body D3 with explicit `rs6/rs8/alp`.
- zero-damping plus ATM and unknown damping variants fail closed.

Parameter-catalog availability does not imply executable capability, and BJ capability
does not imply either zero damping or ATM.

## Independent scientific gates

Zero damping follows the pinned simple-dftd3 definition with pair-specific vdW radii,
`alpha6=alp`, and `alpha8=alp+2`. The independent oracle used dftd3 1.6.0 with
oracle-library SHA-256
`fc5b452cb2303157d431b8a8af9c36ed0d9320e67fcf915bc550a57840f49ee7` and explicit
`s9=0`.

The committed native zero-damping gate fixes independent PBE and PBE0 energy plus
analytic-gradient fixtures. PBE additionally passes centered finite differences at
`2e-4`, `7e-5`, and `2e-5` bohr, translation invariance, and zero total gradient.

The pre-existing independent ATM primitive remains a separate scientific gate.
Production composition is additionally tested as

```text
(BJ + ATM) - BJ == independent ATM oracle
```

for both energy and analytic gradient, which detects either omitted or duplicate ATM
application.

## Runtime, ABI and failure boundaries

The D3 production owner retains the established ragged offsets, changed-geometry replay,
bounded workspace and per-item status contract. The public C descriptor is append-only:
the historical BJ prefix remains accepted while zero damping and ATM require the
extended descriptor. Provider, scheduler and prepared-variant identities are published
independently.

Public native tests cover independent zero-damping values, BJ+ATM exactly-once
composition, changed geometry, energy-only execution, legacy BJ ABI compatibility,
resource-bound rejection, unknown damping rejection, and zero+ATM rejection.

## CUDA correctness evidence

Real-device qualification used an NVIDIA H100 80GB HBM3 with driver 595.58.03 and
CUDA toolkit 12.9.86 (sm90).

The promoted CUDA owner is the deterministic per-system cooperative ragged kernel.
Each logical atom owns its CN, direct-force and CN-response accumulation, and system
energy is reduced in a fixed atom order by one thread. Systems below 8 atoms retain the
one-thread reference path; systems with 8 or more atoms execute one 128-thread
cooperative block per ragged system. ATM remains a separately qualified additive stage
after the two-body contribution.

On the H100, CPU/CUDA parity passed for ragged 64- and 96-atom systems across BJ,
zero damping and BJ+ATM, including changed geometry and energy-only execution. The
public variant/exactly-once suite and the 4100-system ragged/masked replay gate also
passed on the same device.

A global unique-pair prototype using FP64 atomics was qualified for correctness but
**not** promoted. Across matched 1/4/16-system, 64--512-atom public endpoint workloads,
its median speedup relative to the cooperative kernel ranged only from 0.994x to
1.018x. That did not constitute a material endpoint improvement and would also have
replaced the atom-owned reduction order with atomic accumulation, so the prototype was
removed from the production scheduler.

## Matched endpoint promotion

The admitted promotion compares the same H100, prepared public
`vibeqc_d3_batch_execute` endpoint, geometry generator, FP64 model, 8 warmups and
30 measured iterations. The baseline changes only the cooperative threshold so the
same production code executes the original one-thread per-system CUDA evaluator.
Checksums are identical between both paths.

| fleet | atoms | property | cooperative median ms | serial median ms | median speedup | cooperative p95 ms | serial p95 ms | p95 speedup |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | 8 | gradient | 0.070980 | 0.134081 | 1.889x | 0.071901 | 0.134986 | 1.877x |
| 1 | 8 | energy | 0.054237 | 0.100821 | 1.859x | 0.054586 | 0.101640 | 1.862x |
| 1 | 16 | gradient | 0.093291 | 0.388168 | 4.161x | 0.094300 | 0.390883 | 4.145x |
| 1 | 32 | gradient | 0.137040 | 1.389206 | 10.137x | 0.138426 | 1.396139 | 10.086x |
| 1 | 64 | gradient | 0.227993 | 5.400577 | 23.687x | 0.229177 | 5.407722 | 23.596x |
| 1 | 512 | gradient | 5.722625 | 340.566240 | 59.512x | 5.732159 | 340.623452 | 59.423x |
| 64 | 8 | gradient | 0.077606 | 0.146536 | 1.888x | 0.079281 | 0.147815 | 1.864x |
| 64 | 32 | gradient | 0.164872 | 1.419705 | 8.611x | 0.178798 | 1.422637 | 7.957x |
| 64 | 128 | gradient | 0.466293 | 48.214070 | 103.399x | 0.469378 | 48.338850 | 102.985x |
| 64 | 128 | energy | 0.343965 | 31.347320 | 91.135x | 0.345419 | 31.362078 | 90.794x |

The crossover is already material at 8 atoms for both energy and energy-plus-gradient,
including a 64-system ragged fleet, while all measured p95 values improve rather than
regress. Production therefore admits `kD3CooperativeMinimumAtoms = 8` and publishes
scheduler identity `ragged-system-cooperative-pair-v1`.

## Generated-route boundary

The GeometryIR/PairIR ragged CUDA route remains a non-public retirement candidate.
Retiring the native D3 owner requires separate evidence for per-item failure isolation,
energy-only execution, resource bounds, changed-topology replay and matched endpoint
performance; its existence never widens the public D3 capability matrix.
