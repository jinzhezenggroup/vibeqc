# Cold CUDA DF DIIS and complete 768-AO forces (#308)

Device DIIS removes the cold solve's exhausted compact fixed-point loop and
subsequent host retry on the 96-atom / 768-spherical-AO fixture. Both energy
paths use the same frozen FP64 library, requested tolerances, initial core
policy and declared 8-GiB DF value allowance. The fixed-point control sets
`VIBEQC_DF_DISABLE_DEVICE_DIIS=1`; its ordinary host DIIS recovery is unchanged.

| Native endpoint | Native call | Compact iterations | Host retry iterations | Strict final corrections |
| --- | ---: | ---: | ---: | ---: |
| Cold energy, fixed-point control | 259.3482 s | 100 | 33 | 1 |
| Cold energy, device DIIS | 142.8711 s | 21 | 0 | 5 |
| Cold energy and full forces, device DIIS | 397.4849 s | 21 | 0 | 5 |

All three have zero CPU-reference eigensolves and pass independent physical
state checks. The cold-energy ratio is 1.8153x in this instrumented observation.
These are single diagnostic calls with journals, memory sampling and physical
export, not repeated clean timing or iteration-matched cross-engine parity.
The public retry iteration field describes the 33-iteration retry; the journal
also retains all 100 preceding compact iterations. It must not be read as the
complete control's work count. Final corrections are included in every time.

The cold full-force result compares every one of 288 components and the actual
native W consumed by Pulay assembly against the independently qualified PySCF
checkpoint and force array. No imported density seeds the native cold solve;
input C/S is qualified for reference checking only.

| Independent full-force check | Maximum absolute error |
| --- | ---: |
| Energy (Ha) | 3.729e-11 |
| Every force component (Ha/Bohr) | 2.085e-10 |
| Actual native W versus PySCF | 3.681e-10 |
| W versus exported C/epsilon | 7.106e-15 |
| F D − S W | 5.390e-10 |
| F C − S C epsilon | 1.883e-10 |
| Cᵀ S C − I | 3.031e-14 |
| Physical commutator | 6.717e-11 |
| Density versus PySCF | 5.567e-11 |

Electron trace, metric idempotency, canonical density and energy reconstruction
also pass the existing checks. Native energy is −2438.354141054259 Ha. Maximum
net force is 1.197e-11 Ha/Bohr; it supplements the component comparison.

The cold energy setup remains about 104 s. Compact SCF changes from 86.63 s
plus 62.16 s of host retry to 20.13 s. Strict finalization costs 17.05 s with
DIIS versus 5.07 s for the control; the complete endpoint includes this cost.
Cold full forces add about 60.40 s of one-electron preparation, including real
AO derivative exports, and 192.925 s of complete DF response within 210.34 s
of finalization. These are nested inclusive regions and cannot be summed again.

The sampled process GPU peaks are 6,040 / 6,130 / 18,564 MiB in table order.
Host high-water marks are 868,315,136 / 793,964,544 / 6,115,749,888 bytes.
The force request's 16-GiB DF allowance partitions 8 GiB for value/SCF and 8 GiB
for response. These subbudgets are not whole-process limits. The global #203
qualification remains at at most 16 orbital and 128 auxiliary AOs, with its
guards unchanged.

The implementation reuses the normalized device DIIS kernel and physical
commutators, with one joined-spin history per UHF item. Every solve resets its
history. Energy is evaluated before extrapolation; physical final-state
selection, complete response and failure limits remain intact. History storage
is reserved before K panels. Validation also exposed missing independent
metric/compact solver workspace floors: queried workspace must now fit each
owner's reservation, including a compact floor per batch item. Infeasible
allowances fail without a numerical retry; small-budget test thresholds reflect
these reservations and explicitly retain rejection coverage.

Validation on this library includes 127 host checks, 95 GPU Python checks,
17 further DIIS checks with actual reused warm graphs, native DF with both
one-electron response providers, capture recovery and occupied UHF batch-four
memcheck with zero errors. Jobs 9561–9564 used finite Slurm allocations on
`main` with `gpu:5090:1`, preserving assigned visibility. Earlier failing test
attempts, corrected resource expectations, logs, source reconstructions and
identities of retained development binaries are included. No numerical
threshold was relaxed. Provider smoke qualification separately passed with
GPU4PySCF 1.8.1, CuPy 14.2.0, cuTENSOR 2.3.1 and PySCF 2.14.0; it is not a
performance comparison. The initial cuTENSOR 2.8.0/CuPy fallback is retained.

The measured source commit is `6500a06760b84f426ee0bbc05877ec73c9f6737e`.
Native source identity is
`93e0edc6f59b0b71a2377e0715497138184a80443a14a7d2219048a2bb8b7589`;
library SHA-256 is
`44c140b4d46b69ea7846d8aa2d9aa8cc3bec8c198908fd9e830e9981f78d4bc7`.
The frozen library/probe and source patch remain in
`.artifacts/issue308-frozen-device-diis-final/`. The reconstruction patch applies
to `c859854002a00cd03679e34f06a138912b66e2e4`.

`summary.json` contains all force components, checks, source/input/library
identities, work counts, memory and member hashes. `evidence.zip` retains full
native W and forces, orbital energies, sampled D/S/H/F/C rows, complete journals,
runners, build metadata and failures. Binary state members are row-major native
little-endian float64. Full transient state exports were independently checked
and hashed and remain local. Every archive member was restored and compared
byte for byte. Archive SHA-256 is
`40f3096eb83b9f21065bc572ed12f2f042a73531ecab6c448fc758a7e65b391c`.
Reference inputs come from `../issue308-large-diagnostics/reference.zip`; its
archive hash and the separate checkpoint/force hashes are preserved by that
artifact and these result records.

Repeated clean 96/192/384-AO integration/ablation data, the full #309/#310/#311
matrix and matched GPU4PySCF acceptance remain open. This result does not close
#308/#310/#311/#206 or expand global resource qualification.
