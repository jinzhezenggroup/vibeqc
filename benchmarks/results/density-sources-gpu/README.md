# Bounded GPU density sources (#235 B / #297)

Numerical acceptance on clean revision
`db6f65bd96df7dc6f2eae6e7fc8eb766829d1027`, RTX 5090, CUDA 12.9.1,
driver 580.95.05. The source hash, native/JIT binary identities, actual Slurm
allocation, compiler versions and complete reproduction command are in
[evidence.json](evidence.json); [publication.json](publication.json) pins its
bytes through the existing validation/publication schemas.

Six independent saved grid fixtures cover H2, water, Cartesian/spherical f,
diffuse and tight bases. Both GPU routes pass every requested feature entry.
The complete available fixed-density endpoint uses GPU AO/density features
and the existing compiled **CPU** XC/potential consumer. Its 48 cases cover
four independent integration fixtures, LDA/PBE, total/unrestricted spin
layouts and 128/256 MiB device caps. All 480 interleaved D/C calls pass the
same-grid energy/potential gates (`atol=1e-11`, `rtol=1e-10`): maximum absolute
energy error 5.56e-16, potential-entry error 1.34e-15. All 252 retained
worst-block gates pass; their maximum scaled error is 8.71e-5.

The largest observed native arena is 4,225,792 bytes and the largest observed
cuBLAS allocation delta is 69,206,016 bytes. The largest composed capacity
is 104,889,088 device bytes and 349,376 host bytes. The arena contains global
D/B and bounded point/orbital panels; Psi does not scale with total occupied
count. These are numeric ownership observations/capacities with recorded
exclusions, not measured whole-process peak memory.

Raw wall-clock samples include source weighting/upload, GPU AO/GEMMs/reductions,
AO/feature downloads and CPU XC/AO-potential work. Source construction,
test-only fixture Cholesky and external factor validation are measured
separately. Cold code generation/compilation took 5.48 seconds. Across this
small diagnostic matrix, median D/C whole-call times are 6.40/8.28 ms;
these pooled numbers do not select an algorithm. The record makes no speedup,
SCF, force or device-resident XC promotion. Larger molecular and throughput
endpoints, native prepared/force integration and #168 selection remain open.

[verification.json](verification.json) retains the test/sanitizer results and
negative allocation-probe result. Routine logs and compiled products remain
untracked in `.artifacts/`. See [the source contract](../../../docs/developer/density_sources.md)
for public API, memory exclusions and reproduction instructions. The publishing
revision only adds evidence, documentation and the explicit empty `archives`
field required by the existing storage publisher; measured arithmetic is unchanged.

CUDA ownership relative to #296: compiler-owned AO polynomials and pruned
D/C bilinears/sigma share one generated definition. The conservative native
scientific traversal/reduction region grows 54 → 89 code lines (+41 / -6),
and runtime code grows 332 → 447 (+123 / -8). No legacy production path is
removed; D/C are distinct algorithms, so retained duplicate reason is `none`.
