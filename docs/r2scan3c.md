# r2SCAN-3c capability contract

VibeQC exposes canonical r2SCAN-3c through the Python selectors
"r2scan-3c", "r2scan-3c-rks", and "r2scan-3c-uks". These names resolve to the
same inspectable R2SCAN-3c MethodIR used by the compiler; they do not add a
second native scientific driver or a separate native method ID.

The canonical method binds four defining pieces: r2SCAN electronic structure,
the pinned spherical H-Ar def2-mTZVPP basis, the r2SCAN-3c D4(BJ)-EEQ-ATM
profile, and the matching gCP profile. Changing any defining component requires
a different method identity. The first supported element domain is Z=1..18.

| Path | Energy | Forces | Batch/replay | Notes |
| --- | --- | --- | --- | --- |
| CPU RKS/UKS | yes | not promoted | yes | exact canonical basis; D4 and gCP are explicit components |
| CUDA RKS/UKS | yes | gated | yes | electronic/D4 execution may use CUDA; gCP remains a disclosed CPU component |

method_capabilities() intentionally reports the conservative backend-neutral
surface (energy plus batch support). Force availability is resolved by
Calculator after the device, basis, and stationary-gradient domain are known.

Production total energy is assembled once as E_r2SCAN + E_D4 + E_gCP.
Where forces are admitted, the corresponding force is
F_r2SCAN - dE_D4/dR - dE_gCP/dR. Prepared replay keeps the correction owner
separate and supports changed coordinates without silently applying either
correction twice.

The method remains fail-closed outside H-Ar, for a changed defining basis, or
when a requested backend/property combination has not been qualified.
