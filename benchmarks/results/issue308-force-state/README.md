# Complete 768-AO native force and actual W comparison (#308)

The native RHF endpoint completes energy and all 288 analytic-force components
for the 96-atom / 768-spherical-AO case with the complete DF, Pulay, metric and
auxiliary-basis response. It starts from the independently qualified PySCF
density and converges in three compact iterations with no DIIS retry or
CPU-reference eigensolve. Final selection reuses the verified physical frame
after one physical Fock evaluation.

| Independent check | Maximum absolute error | Existing threshold |
| --- | ---: | ---: |
| Energy (Ha) | 3.684e-11 | 1e-9 |
| Every force component (Ha/Bohr) | 1.761e-10 | 1e-8 |
| Actual native force W versus PySCF | 1.004e-11 | 1e-8 |
| Actual W versus exported C/epsilon | 7.106e-15 | 1e-8 |
| F D − S W | 1.478e-12 | 1e-8 |
| F C − S C epsilon | 5.383e-13 | 1e-8 |
| Cᵀ S C − I | 3.019e-14 | 1e-8 |
| Physical commutator | 2.023e-13 | 1e-8 |
| Density versus PySCF | 2.189e-12 | 1e-8 |

Electron count, metric idempotency, canonical density, overlap, physical Fock
and reconstructed energy also pass. Native energy is −2438.3541410542593 Ha;
the retained PySCF energy is −2438.354141054296 Ha. Maximum net force is
9.413e-12 Ha/Bohr, reported as an auxiliary check rather than replacing the
comparison of every force component. Unlike the earlier energy diagnosis,
the exported W is the actual matrix consumed by native Pulay assembly.

The instrumented native call takes 365.7205 s. The journals separate about
60.25 s of one-electron work, 104 s of retained source materialization,
2.670 s of compact SCF, and 196.678 s of finalization. Finalization includes
193.175 s of complete force response and 1.125 s of diagnostic reference export.
Within DF response, the final three-center derivative contraction takes
20.285 s; substantial preceding work forms exchange response weights. Nested
inclusive times overlap and must not be added again. Raw journals and exclusive
host accounting are retained for the remaining #206 analysis.

This is a **seeded diagnostic full-force endpoint**, not a cold-start or clean
performance result. The 16-GiB DF subbudget partitions 8 GiB for values/SCF and
8 GiB for response; it is not a whole-HF resource acceptance. The sampled GPU
process peak is 18,474 MiB and host high-water mark is 6,115,356,672 bytes.
The default one-electron derivative export retains its real host copies.
The #203 qualified global-resource range and guards are unchanged. No
same-hardware GPU4PySCF parity is claimed.

Slurm job 9555 passed 26 Python GPU tests and the native DF suite before the
large probe. The small export tests cover both cold and independent seeded
starts, with energy and full forces. Native CPU reference export, structure,
ownership and journal checks also pass. All real GPU work and hardware queries
used a finite `srun` on `main` with `gpu:5090:1`, preserving assigned visibility.

The measured library SHA-256 is
`cce33dae40b774bc785dd2b7394dbaaa42741deca9d93c4d97eb7610ca4db603`;
native source identity is
`1fa09a8651ab1b15e85fedbfa1186ed59c1b15e60b414bbf6f4cfea2f03e166c`.
The frozen library/probe is in `.artifacts/issue308-frozen-force-state-initial/`.
The exact dirty reconstruction patch applies to
`9ebece6d31a619f679977237f85df8e54bca49ed`; new probe/validator sources are
archived separately. The independent checkpoint and force arrays were restored
byte for byte from `../issue308-large-diagnostics/reference.zip` before use.

`evidence.zip` retains full native W, all forces and orbital energies, sampled
rows of D/S/H/F/C, complete journals, memory samples, source/build/input
identities, runners and validation logs. Every component of the complete
transient state export was independently checked and hashed; the full
transient export remains local. Every archive member was restored and compared
byte for byte. Exact hashes and all force components are in `summary.json`.

Cold SCF orchestration, repeated clean matrix/ablations and matched GPU4PySCF
data remain outstanding. This endpoint does not close #308/#310/#311/#206.

## Archive storage correction

`evidence.zip` was removed from the current tree when restoring the hard
1 MiB file limit. Its exact bytes remain in commit `daa2da0867877c94c40f379ffe1f3db6e3036ef8`; the
[storage migration](../retention-size-limit/migration.json) pins its SHA-256
and size. Existing numerical conclusions and measured identities are unchanged.
Restore the historical archive to an ignored working directory with:

```bash
python tools/restore_retained_evidence.py benchmarks/results/issue308-force-state/evidence.zip
```

Archive restoration is only needed for historical raw-run inspection. New runs
keep full logs, profiles and retries outside Git.
