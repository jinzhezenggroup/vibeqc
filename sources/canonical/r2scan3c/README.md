# r2SCAN-3c audited data

This directory pins the data defining the first VibeQC r2SCAN-3c support domain (H-Ar). The runtime orbital basis snapshot is shipped under `python/vibeqc/data/r2scan3c/`; this directory retains the reproducible BSE source export, gCP qualification parameters, source revisions, hashes, licenses, and the upstream MB16-43/06 reference fixture.

The canonical method is not an alias for plain r2SCAN: its MethodIR identity binds the exact def2-mTZVPP basis identity plus the r2SCAN-3c D4 and gCP profiles. Changing any defining component requires a different explicit method identifier.
