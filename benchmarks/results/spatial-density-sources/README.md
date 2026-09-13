# Prepared spatial density sources (#235 C1 / #299)

Clean measured revision `4d23418a8af4d8363c41b10b300ecfc4208919e8`, RTX 5090,
CUDA 12.9.1 and driver 580.95.05. [evidence.json](evidence.json) retains exact
source/library/generated identities, assigned Slurm device, settings, raw samples
and the reproduction command. [publication.json](publication.json) pins all
selected bytes through the existing evidence schema.

Both routes pass 96 fixed-grid/mask E/V cases: H2, water, Cartesian/spherical f,
LDA/PBE, three spin/layout choices, two device budgets (128/256 MiB), and spatial
screening off/on. All 960 interleaved calls and 444 retained block gates pass
atol 1e-11, rtol 1e-10. Maximum E error is 4.44e-16; maximum V entry error is
8.88e-16. The six existing independent feature fixtures are also rechecked.

Screened acceptance compares against global-AO Python contractions with omitted
jets zeroed at the identical mask. Twelve endpoint cases contain nonempty strict
AO subsets at cutoff 1e-4. Screening differences against saved unmasked fixtures
are reported separately: at most 1.17e-11 energy and 8.88e-11 potential. These
are observed small-fixture differences, not a general error bound for this cutoff.
Additional through-f two-center tests exercise stronger local restriction,
fractional/empty spins, occupied counts larger than local AO counts, complete
cross terms, source failure/fallback, stale consumers, device leases/reset and
three-step density directional gates.

The largest observed CUDA arena is 4,244,992 bytes; the largest observed cuBLAS
allocation delta is 69,206,016 bytes. Composed capacities reach 104,908,288 device
bytes and 392,032 host bytes. The spatial CUDA owner is charged once alongside
CPU XC/work/output capacity. These quantities do not measure whole-process peak
memory, and a replacement must separately fit both old and new owners.

The endpoint executes AO/features on GPU, then downloads each tile for native
CPU XC/potential assembly and CPU global scatter. Raw samples include source
packing/upload, all device phases, downloads and CPU work; source creation,
test-only fixture Cholesky/validation and owner construction are separate.
Pooled D/C medians are 9.82/12.13 ms with screening off and 9.72/12.11 ms with
screening on. These small workloads make no performance promotion claim.
Cold generation/compilation took 5.45 seconds.

[verification.json](verification.json) records 108 targeted CPU tests, 152 GPU
tests, all 40 new nodes under memcheck and individual-process initcheck with zero
reported errors. The prior #297 long-lived instrumented cuBLAS allocation limit
remains explicit; no allowance was relaxed. Routine logs/builds stay untracked.

This completes the local/prepared fixed-density slice of #235 C. Native SCF
producer provenance, full forces and automatic selection remain #162/#163/#168.
The evidence commit adds only retention metadata to the measured code. A later
lease-order fix rejects borrowed spatial tasks before waiting for their CUDA lock;
it changes no numerical arithmetic. Its source hashes, 154-test GPU replay,
61 related CPU tests and both sanitizer checks on the two new concurrency nodes
are retained separately in verification.json. The original timing revision
remains explicit.
A subsequent review fix takes the entire constructor snapshot under the spatial
lock and invalidates a consumer when same-mask resource capacities change. Its
65 CPU and 154 GPU regression results and exact source hashes are also retained.
The compact 1.48 MB envelope retains all 960 full-cost samples, 96 resource plans
and 444 numerical block gates; its exact bytes have a scientific size-review
exception in `benchmarks/evidence-policy.json`.

CUDA ownership: existing generated AO and D/C feature definitions are reused.
Handwritten scientific CUDA LOC +0 / -0; runtime CUDA LOC +0 / -0. No legacy
production path removed; retained duplicate reason: none.
