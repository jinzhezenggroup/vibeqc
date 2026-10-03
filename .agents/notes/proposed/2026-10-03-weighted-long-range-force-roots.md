# Proposal: share low-order long-range force roots across components and centers

Status: proposed; independent device and complete numerical gates pass;
controlled full-grid 24-atom endpoint passes; 96-atom comparison pending
Date: 2026-10-03

## Measured problem

The complete 96-atom WB97M-V diagnostic trace in n1 Slurm 5493 contains two
bounded force traversals: 28.802885 s and 188.013665 s. The frozen source's
full-then-LR launch order identifies these as full-range J/K and LR exchange;
Nsight records their order, not radial arguments. VV10 separately takes
114.341840 s in SCF and 158.525478 s in forces. All geometry kernels together
take 36.638392 s, so fusing geometry cannot remove the entire nonlocal drain.

The profiled source is `8bfde172cf9595e7cd82a9d2bcf856a9a07dcc37`, with library
`7345024fb005320f605027e5b2a1aaac555e51ba670c73dd17c27707cb8505f1`.
Its complete warm diagnostic endpoint is 630.426497 s. Cold and warm pass an
independent clean reference96 comparison (maximum energy error 1.13e-10 Eh and
force error 3.27e-10 Eh/Bohr). This is a profile, not a clean timing sample for
the new candidate. Raw trace, SQLite, hashes and the verifier are retained in
the storage experiment's `.artifacts/wb97m-force-storage/profile-review96/` and
`profile96-summary.json`.

## Candidate

Start independently at master `d35ae539f645cb5e8b9c0c2fb7a426c5e2ad08e8`,
which already shares low-order full-range J/K geometry. Reuse its compiler-owned
weighted force roots for LR exchange of total angular order zero through three.
The bounded shell scheduler builds one exchange weight vector and evaluates one
radial moment ladder per primitive quartet, then accumulates all independent
center derivatives. It no longer repeats the Cartesian Dual3 recurrence for
each component and unique atom in this closed low-order domain.

This does not introduce a new integral formula. At fixed primitive exponents
and omega, LR moments obey `dM_n/dT = -M_(n+1)`, exactly the derivative relation
consumed by the existing weighted roots. The scalar moment owner remains
`integrals/range_moments.hpp`; component/center algebra remains compiler-owned.
The dependency checker admits only that exact shared numerical header to the
native contraction adapters, without admitting the CPU tensor/oracle layer.

## Boundaries and gates

Full-range consumers retain their two independent source channels. Short-range,
fused RSH and higher angular orders retain their existing bounded/AOT paths.
The molecular omega=0.3 path still reconstructs SR as Full minus LR, with each
functional coefficient applied by its existing owner. Precision, screening,
basis normalization, source meanings and allocation bounds are unchanged.
Invalid moment inputs propagate nonfinite output to the existing failure gate.

AO contraction and atom accumulation order change for the selected LR tasks,
so bitwise LR equality is not claimed. Require independent CPU displaced
integrals at 3e-8 derivative error, complete RKS/UKS and changed-geometry gates
at 1e-8 Eh / 1e-7 Eh/Bohr, all repeats, and controlled full-grid 24/96 endpoints.
The native tests call the actual bounded molecular boundary as well as the
canonical provider, with a four-center d/p/s/s fixture and repeated atom binding.

The existing host control checks 13,026,816 full-range coordinates bitwise equal
to the retained scalar workers. Focused host checks total 110 passing tests.
The untouched latest-master library is
`c52dd96d9c35ea6f035e5d690a45fad70c7c2d2b6329f94c072c6be182f9275c`;
all 442 build compiler commands invoke ccache. Before/after host-wide cache
receipts, source archive, candidate patches and binaries remain under
`.artifacts/range-weighted-force/`. No endpoint improvement is established yet.

## Rejected diagnostic direction

n2 Slurm 2149 reconstructed VV10 predicates on an independently converged
reference96 density: 1,923,992 density-active rows, 1,920,439 nonzero-weight
partners, and only 3,553 exact-zero weight rows. Removing those rows in a
strictly weighted-only consumer could save only 0.1847% of reconstructed pair
work. This is not a native executed counter and does not justify changing the
native density policy or discarding observable weight derivatives. The known
GPU4PySCF additional absolute-weight filter is not adopted as an optimization.

## Revisit when

Promote only after controlled complete endpoint improvement, larger-system
qualification and independent numerical gates. Reconsider the scalar shell
schedule if its register pressure or serialization regresses the endpoint;
the generic higher-order recurrence remains the explicit bounded fallback.

## Final portable candidate qualification

The frozen production candidate is `ac8e14728e389d4a8791da9a603260455feccb81`.
Its clean-source content identity is
`d67decd856253688d79dd6e5b330678a10bb5d087a80771ea7cd85743e464c64`, independently
recomputed and matched against the rebuilt library. Library SHA-256 is
`e51239008c378cf61e4703abd5e287825beb131bb192f1aea7a444a7a449f5bd`; source archive
SHA-256 is `dfd3c6b4fd4dbad0ee153d402a1baf931802b992d22e83a8a13bdfb4ad1f269d`.

n1 Slurm 5510 (finite 45-minute RTX 5090 allocation, exit 0) passes native
s/p/d/f SR/LR and Cartesian order-two independent CPU finite-difference gates,
including the actual bounded route, four-center and repeated-center bindings.
The native executable passes compute-sanitizer memcheck with zero errors.
All seven complete independent WB97M-V RKS/UKS and force-rebuild/stale-state
tests pass in 199.40 seconds. CuMetal CI run 37101509535 also passes after the
invalid-input marker was changed to a namespace compile-time NaN constant.

The previous `e08112c` library (`dc835e13...`) passed the same numerical gates
but failed CuMetal compilation because its `nan()` call was host-only. That
prototype and all receipts are retained under `attempts/e08112c/`. Its
reduced-grid water12 diagnostic passes the independent CPU original/moved
oracles: single warm endpoint 22.580341 -> 20.892754 seconds, moved endpoint
90.994328 -> 89.742201 seconds, maximum energy/force errors 1.43e-12 Eh /
8.97e-10 Eh/Bohr. These single observations are neither full-grid speedup
claims nor timings of the final portable binary.

The final binary's same-allocation full-grid 24-atom comparison runs in n1
Slurm 5511; its moved12 requalification runs in 5512. The controlled full-grid
96-atom base/candidate comparison runs in n5 Slurm 1410. Its candidate phase
is gated by a successful numerical receipt carrying the actual library hash;
the final 5510 receipt has been verified and installed. Each full-grid engine
has a fresh full-density Fock reference and three engine-local warm repeats.
No controlled final-candidate speedup is established yet. Master advanced to
`59eee77f4` through Python loader work while these frozen measurements ran;
no old result is relabeled as that newer source.

## Final controlled full-grid 24-atom endpoint

n1 Slurm 5511 completed (exit 0) with both frozen builds sequentially on one
RTX 5090, water24/192 spherical def2-SVP AOs, 48 x 16 x 32 grid and three
engine-local warm repeats. Every cold/priming/warm native-reference pair and
all reference consistency gates pass. Across both builds, maximum errors are
2.73e-12 Eh and 4.96e-10 Eh/Bohr.

| Complete endpoint | Base / s | Candidate / s |
| --- | --- | --- |
| Native cold | 263.275725909 | 256.124819398 |
| Native priming | 38.985069916 | 30.759933736 |
| Native warm median | 39.042251013 | 30.441405468 |
| Fresh reference warm median | 27.692985114 | 27.682573192 |

The candidate warm samples are 30.458534, 30.441405 and 30.423864 seconds.
Native warm time improves by 22.030%, yet remains above the reference.
Both native cold solves take 18 iterations; all priming/warm solves take one.
The reference takes 14 cold iterations and one warm iteration/two J-K builds.
The first warm integral-derivative component falls from 14.872265 to 6.297789
seconds; grid/pair drain stays at 10.967430 versus 10.963736 seconds. Grid,
geometry, collocation and allocation work fields remain unchanged. Actual
executed bounded shell-quartet counts remain unavailable and are not inferred
from dense logical capacities.

Final moved12 qualification in n1 Slurm 5512 also completed (exit 0). All five
observations per build pass the retained independent CPU oracle. Base/candidate
cold, priming, single warm and moved endpoints are 168.526401/166.538064,
23.193344/21.721125, 23.220666/21.749440 and 92.803257/91.119179 seconds.
Iteration counts are 22/1/1/11. Maximum errors across both builds are 1.37e-12
Eh and 8.97e-10 Eh/Bohr. This reduced 24 x 8 x 16 grid remains a correctness
and rebuild diagnostic, not the full-grid performance comparison.

The full-grid 96-atom same-allocation comparison remains live in n5 Slurm 1410.
No large-system reference advantage is established by the completed 24-atom run.
