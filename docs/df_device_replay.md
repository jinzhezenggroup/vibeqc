# Bounded device DF replay (#205)

Positive-budget CUDA DF plans retain transformed B when its complete value/SCF
allowance fits, using bounded K scratch independently of B's full shape.
Otherwise J/K regenerate public-basis three-center tiles on device. The complete
two-electron HF force response uses separately bounded generation in both cases. The
mathematical provider remains `density_fitted`; execution placement does not
change its fitting basis, metric cutoff, energy normalization or force sign.
The one-electron and overlap/Pulay consumer remains owned by #141.

## Transfer and lifetime ledger

| Data | Setup / geometry rebuild | Fixed-geometry J/K | Final force |
| --- | --- | --- | --- |
| Raw metric M | Generated on device; returned to host once, then uploaded for cuSOLVER | No transfer | No transfer |
| Metric eigensystem Q, eigenvalues and X=M^(-1/2) | Produced by cuSOLVER; device ownership transfers from setup to the plan | X remains resident | Borrows the same Q, eigenvalues and X; no host metric reverse contraction |
| Raw A[mu,nu,P] | Source retains topology and public transforms; retained-B setup borrows K scratch and generates each raw value once | None for retained B; otherwise regenerated in bounded panels | Regenerated on device one AO matrix per auxiliary function |
| Transformed B | Optional full retained tensor, separate from bounded K scratch; otherwise bounded generated panels | Retained B is gathered by Q; generated panels are consumed on the same stream | Not needed by the response |
| RI-J charge and RI-K transforms | Plan-owned device buffers | No tensor transfer | Independent response scratch |
| HF densities | Device SCF state is retained; host entry points explicitly upload their inputs | Device SCF entry points have no density transfer; existing host numerical recovery transfers AO matrices | Current finalization uploads one RHF or three UHF density terms; separately counted |
| A/M response weights | Not retained | Not used | Computed and consumed on device; no weight upload or raw-value download |
| Basis metadata / positions | Source uploads on rebuild | Retained | Compact derivative metadata is uploaded per force call |
| Graph / solver state | Owned by the plan and invalidated on geometry/policy changes | Existing stream/graph rules and scalar convergence/status reads | Same owning stream; no new graph-owned storage |
| Final gradient | None | None | Only `3*Natom` doubles downloaded by the DF response |

The native host-value compatibility API still supports resident B and
host-backed A inputs. Its independent host force adapter and explicit weight
uploads are reported as such. A failed source response never retries through
that adapter. Geometry changes rebuild the corresponding source and metric;
fixed-topology batch slots retain their existing ordering and failure isolation.
The existing SCF numerical recovery may use CPU DIIS/eigensolvers and transfer
AO density/Fock matrices. Both control routes keep M/A/B tile generation and
DF contractions on device. Complete HF traces therefore include more traffic
than the separately captured DF force response.

## Bounded response and metric convention

For each symmetric density D the energy contribution is
`0.5*cJ*rho^T M+ rho - cK*Q:M+`. Device contractions use the same dense
multiplicity and signed coefficients as the independent CPU response adapter.
They construct `D^T A_P D` using two cubic AO matrix products, then accumulate
one bounded auxiliary block of A weights and the complete metric weight.
Every output element has a fixed summation order. The existing derivative
mapping controls final gradient reduction; `serial` replay is deterministic.

The reverse map uses the exact forward eigensystem:
`bar_M = Q (L .* (Q^T sym(bar_M+) Q)) Q^T`. L contains the divided differences
of the thresholded reciprocal. Its retained/discarded blocks include finite
discarded eigenvalues; using only `-M+ bar_M+ M+` would give wrong forces when
the retained subspace moves. Setup records an unresolved cutoff within
`128*epsilon*largest_eigenvalue`; force execution rejects that slot before
allocating response storage, while energy execution keeps its existing rank
semantics.

For n orbital functions, a auxiliary functions, t density terms and auxiliary
block size b, device scratch contains
`4*a*a + (3+2*b+t)*n*n + 2*t*a` doubles plus packed basis metadata, coordinates
and the final gradient. The four metric matrices and one-AO-matrix floor are
explicit. b is selected only after fixed allocations are charged and can be
one when a full transformed tensor does not fit. No allocation is hidden in
the response launch wrappers. The original eigensystem adds `a*a+a` retained
doubles per system, moving ownership out of setup without increasing its
allocation peak. The complete device ledger remains authoritative for all
simultaneously live plans.

The HF resource alternatives expose `response_minimum`, `response_capacity`,
and `response_host_capacity`, together with existing value tile dimensions,
resident/peak bytes and recomputation identity. A positive DF request still
splits equally between the value plan and response; UHF charges its host total
density against the response portion. Insufficient budgets fail transactionally.

## Reproduction and evidence

`DfGradientResources` separates tensor H2D/D2H, response-weight uploads,
density uploads, regenerated value bytes, and device-consumed response bytes.
These are semantic counters. Nsight Systems independently checks the observed
transfers in a capture containing five force calls after setup, plus a separate
complete HF capture containing five warm replays of three systems.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 env CUDA_ROOT=/path/to/cuda \
  VIBEQC_LIBRARY="$PWD/build/cuda/libvibeqc.so" PYTHON=/path/to/python \
  bash benchmarks/df_device_replay.sh
```

The script creates a new output directory and records source/binary identities,
raw timings, CPU-oracle errors, allocation and transfer counters, and hashes of
the raw trace databases. It rejects extra D2H traffic in the force capture.
Component timings include generation, response contractions, allocations and
final output. They are distinct from complete-SCF latency.

On the RTX 5090, Slurm job 9253 measures sp8 RHF/UHF at 16/64 KiB response
budgets and sdf18 RHF at 64/128 KiB. All source responses return only 48 bytes;
no tensor or response-weight bulk traffic occurs. Independent CUDA traces
observe five 48-byte D2H copies in each five-call force window. Total H2D is
7,040 bytes for RHF and 12,160 bytes for UHF, matching the metadata/density
counters exactly. The maximum component force error is `3.45e-13` Hartree/Bohr.

The sp8 source response owns 8,240/12,336 device bytes for RHF and
9,520/13,616 for UHF at auxiliary caps 3/7. It retains 1,168 host bytes,
compared with 7,568–11,792 bytes in the compatibility resident adapter.
Those resident inputs avoid value regeneration, so their shorter component
latencies are not a like-for-like comparison of this change's speed. The
complete profiled batch-3 HF median is 133.88 ms, including profiler overhead;
this is an execution record, not a speedup claim.

Slurm job 9255 additionally forces sdf18 to a 32 KiB response budget and a
single-auxiliary block. Its 27,912-byte device response allocation fits that
limit, while a full transformed tensor would require 46,656 bytes. It returns
only the final 48 bytes and agrees with the CPU force oracle to `3.45e-13`.
Regeneration increases to 933,120 value bytes per response, exposing the time
and memory tradeoff rather than hiding a full tensor allocation.

The [archived summary](../benchmarks/results/df-device-replay-rtx5090/summary.json)
retains all eleven component records and transfer-size histograms. Its adjacent
provenance records source identity, binary hash and Slurm allocation. Raw
trace hashes and the reproduction script allow an independent recapture.
The 38 focused CUDA endpoint/resource checks pass, including independent
PySCF forces, two finite-difference steps, RHF/UHF, Cartesian/spherical bases,
batch-3 changed/restored geometries, partial tiles, rank rejection, failed-item
neighbors, and measured global resource caps. All 16 native CPU tests and
106 selected Python CPU checks pass; the three native CUDA DF/provider tests
also pass.
