# Verified complete DF final-state qualification

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

Implementation `3ce4dabd508a0e25aee37b4e0e42b1dd137ae595` was qualified with finite Slurm
allocations on `main`, one RTX 5090: job 9496 for correctness and job 9497 for
memcheck. Source/tree, scientific identity, library and archive/member hashes
are pinned in the manifest. Original bytes remain in `qualification.zip`.

27 CPU native suites, 86 CPU Python resource/protocol/ownership checks (19 CUDA
skips), 22 new GPU selection/force checks, six GPU native suites and 145 existing
GPU Python regressions pass. Three actual endpoint cases (RHF, constrained
batch-four spherical UHF and empty-beta force differences) and the full ordinary
provider/export native suite pass memcheck with zero errors and zero leaked
bytes. All repository hooks pass.

RHF water/UHF OH tests cover Cartesian/spherical, batch one/four, resident/8 MiB
budgets, cold/warm/changed geometry, frozen seeds, force/energy transitions,
failed neighbors and independent CPU energies/complete forces at 1e-9 Eh and
1e-8 Eh/Bohr. Total CPU DF energy central differences with a 1e-4 Bohr step agree
with the new full GPU forces within 2e-6 Eh/Bohr, including H2+ with empty beta;
total forces obey translation invariance at 1e-9 Eh/Bohr. No force term is tested
only by comparing two consumers of the new W.

All 240 selection traces are retained and audited. Unchanged warm force reuse
performs zero final eigensolves and one current physical F evaluation per item.
Forced warm rebuilding performs one joint correction (two eigensolves for UHF)
and two physical F evaluations. Cold/changed cases require up to twelve joint
corrections within the existing sixteen-step budget; all numerical thresholds
remain unchanged. Full W is built once only for force requests.

Sixteen polarized-H2 export cases cover both AO layouts, force/energy output,
cold selection, retained frames and actual device/reference rebuilding. Each
completed reference passes its independent physical checks and agrees with CPU
D/canonical energies/full forces. Accepted export uses no extra eigensolve;
retained cases have zero final solves, while each forced warm case has one.

The archive retains 292 files in 688,727 bytes, including
270 molecular host traces and the complete native export trace. Every member
was restored and compared byte-for-byte before publication. These intrusive
records establish correctness and work counts; clean interleaved #206 timings,
larger/constrained acceptance and independent host-iterative Fock migration
remain separate work. No DF epic or GPU4PySCF parity claim is made.
