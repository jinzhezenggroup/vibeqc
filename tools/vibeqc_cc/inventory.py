"""Audited, expanded real RCCSD energy and singles contractions (CG11 A).

Adapted from PySCF 2.14.0 rccsd and rintermediates, Apache-2.0.
Copyright 2014-2021 The PySCF Developers. See NOTICE and LICENSE.pyscf.
Changes: expand intermediates, restore full Fock diagonals, exact integer
coefficients, named diagnostics, and lower to VibeQC TensorIR (no solver).
"""

# (stable term ID, diagnostic group, rational coefficient, contraction, inputs)
TERMS = (
    ("E01", "energy_t1", 2, "ia,ia->", ("fov", "t1")),
    ("E02", "energy_t2", 2, "iajb,ijab->", ("ovov", "t2")),
    ("E03", "energy_t2", -1, "ibja,ijab->", ("ovov", "t2")),
    ("E04", "energy_t1t1", 2, "iajb,ia,jb->", ("ovov", "t1", "t1")),
    ("E05", "energy_t1t1", -1, "ibja,ia,jb->", ("ovov", "t1", "t1")),
    ("S01", "singles_fock", 1, "ia->ia", ("fov",)),
    ("S02", "singles_fock", 1, "ac,ic->ia", ("fvv", "t1")),
    ("S03", "singles_fock", -1, "ki,ka->ia", ("foo", "t1")),
    ("S04", "singles_fock", -1, "kc,ka,ic->ia", ("fov", "t1", "t1")),
    ("S05", "singles_fock", 2, "kc,kica->ia", ("fov", "t2")),
    ("S06", "singles_fock", -1, "kc,ikca->ia", ("fov", "t2")),
    ("S07", "singles_g_t1", 2, "kcai,kc->ia", ("ovvo", "t1")),
    ("S08", "singles_g_t1", -1, "kiac,kc->ia", ("oovv", "t1")),
    ("S09", "singles_g_t2", 2, "kdac,ikcd->ia", ("ovvv", "t2")),
    ("S10", "singles_g_t2", -1, "kcad,ikcd->ia", ("ovvv", "t2")),
    ("S11", "singles_g_t2", -2, "lcki,klac->ia", ("ovoo", "t2")),
    ("S12", "singles_g_t2", 1, "kcli,klac->ia", ("ovoo", "t2")),
    ("S13", "singles_g_t1t1", 2, "kdac,kd,ic->ia", ("ovvv", "t1", "t1")),
    ("S14", "singles_g_t1t1", -1, "kcad,kd,ic->ia", ("ovvv", "t1", "t1")),
    ("S15", "singles_g_t1t1", -2, "lcki,lc,ka->ia", ("ovoo", "t1", "t1")),
    ("S16", "singles_g_t1t1", 1, "kcli,lc,ka->ia", ("ovoo", "t1", "t1")),
    ("S17", "singles_g_t1t2", -2, "kcld,klad,ic->ia", ("ovov", "t2", "t1")),
    ("S18", "singles_g_t1t2", 1, "kdlc,klad,ic->ia", ("ovov", "t2", "t1")),
    ("S19", "singles_g_t1t2", -2, "kcld,ilcd,ka->ia", ("ovov", "t2", "t1")),
    ("S20", "singles_g_t1t2", 1, "kdlc,ilcd,ka->ia", ("ovov", "t2", "t1")),
    ("S21", "singles_g_t1t2", 4, "kcld,ld,kica->ia", ("ovov", "t1", "t2")),
    ("S22", "singles_g_t1t2", -2, "kdlc,ld,kica->ia", ("ovov", "t1", "t2")),
    ("S23", "singles_g_t1t2", -2, "kcld,ld,ikca->ia", ("ovov", "t1", "t2")),
    ("S24", "singles_g_t1t2", 1, "kdlc,ld,ikca->ia", ("ovov", "t1", "t2")),
    ("S25", "singles_g_t1t1t1", -2, "kcld,ka,ld,ic->ia", ("ovov", "t1", "t1", "t1")),
    ("S26", "singles_g_t1t1t1", 1, "kdlc,ka,ld,ic->ia", ("ovov", "t1", "t1", "t1")),
    ("S27", "singles_g_t1t1t1", -2, "kcld,ic,ld,ka->ia", ("ovov", "t1", "t1", "t1")),
    ("S28", "singles_g_t1t1t1", 1, "kdlc,ic,ld,ka->ia", ("ovov", "t1", "t1", "t1")),
    ("S29", "singles_g_t1t1t1", 2, "kcld,ld,ic,ka->ia", ("ovov", "t1", "t1", "t1")),
    ("S30", "singles_g_t1t1t1", -1, "kdlc,ld,ic,ka->ia", ("ovov", "t1", "t1", "t1")),
)
