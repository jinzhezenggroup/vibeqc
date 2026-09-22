# Candidate: keep local screening distinct from global stream exhaustion

Status: proposed (emitted-scheduler and integrated GPU endpoint gates passed)
Date: 2026-09-23

## Root cause

The component-lane Fock worker claims a global flattened bra/ket ordinal,
then advances across systems. Its emitter incorrectly assigned the exhausted
state to any candidate below the coarse Schwarz gate. A ket tail is monotonic
only within one bra segment. The next bra can have surviving ket partners;
the next system is independent. Retiring a worker at the local tail can leave
valid integrals uncomputed, depending on worker count and topology.

Reserve state 2 for actual global exhaustion. A screened in-domain candidate
uses state 0 and advances the cursor. Keep bra-resident workers' correctly
scoped tail breaks. This changes only enumeration, retaining equations,
precision, screening threshold, density gates, and task ABI.

## Diagnosis

The integrated pure-J/ordinary-KS candidate failed the unchanged PBE24 energy
gate by 6.296e-8 Hartree. At the identical converged density and explicit grid,
XC agreed with GPU4PySCF within 6e-13 and one-electron energy within 1.1e-11.
The entire discrepancy was in Hartree. A direct production-device J probe
isolated ddpp (class 17): screened-minus-unscreened maximum J difference
5.42314e-5 and Hartree difference -6.29405e-8. Other classes stayed below
6e-12 in J. Every normalized Cartesian AO-pair and shell-pair Schwarz bound
independently agreed with libcint within 1e-11. Disabling screening restored
Hartree agreement, identifying lost work rather than scalar integral error.

The scalar fixed-density endpoint used 24 water-cluster atoms, 192 spherical
AOs (200 Cartesian source AOs), def2-SVP and screening 1e-12 on allocated
RTX 5090. Its density, full J matrices, per-class differences and component
comparisons remain in the integration checkout's ignored
`.artifacts/gpu-blocker-fixes/`. Tightening the user's threshold or weakening
the energy gate would conceal this defect and was rejected.

## Regression

The host test compiles the actual emitted ddpp worker control flow with a
serial CTA leader and a recording integral consumer. An independent nested
loop defines retained pairs. Two systems, correctly sorted segments,
screened gaps followed by retained bras, an additional exact-density
rejection, inactive systems, RHF/UHF instantiations, and different worker
counts verify complete enumeration and semantic work counts. The unpatched
emitter retains only 2 of 5 expected pairs in its first case; the fix passes.
This does not claim to simulate concurrent CUDA atomics or certify integral
math; allocated numerical tests and bounded complete endpoints are separate.

## Follow-up

Track #1095. Qualify both generated HF and pure-J consumers; keep #1077 and
PR #1086 open until complete endpoints satisfy the existing gates. Correct
enumeration can cost more than the defective path because the latter omits
work. Performance comparisons must include the retained quartet census.

## Allocated endpoint qualification

Slurm 11293 on RTX 5090 passed the independent native direct-J/K test and
complete PBE24 comparison at unchanged screening 1e-12. Maximum all-sample
energy error fell to 1.444e-11 Hartree. Native cold execution was 27.249 seconds,
warm 3.397/3.438 seconds; reference cold 5.635 seconds and warm 0.745/0.751.
The native and reference warm iteration branches differ, so this is numerical
qualification and complete scoped timing, not iteration-matched acceleration.
Direct-HF12 cold/priming/warm energy-plus-force comparison also passed: maximum
warm-pair errors 3.184e-12 Hartree and 1.589e-11 Hartree/Bohr.

These runs use the integration of #1073/#1076/#1086/#1089/#1091 plus this fix
on its retained older base, not the exact source of this standalone PR.
The exact PR source passed the emitted-worker and artifact/dependency checks.
Evidence files: `stream-fix-v6.log`, `pbe24-stream-fix-v6.json`, and
`hf12-stream-fix-v6.json` in the ignored integration artifact directory.
