# DF cache replay budget expectation

The native two-item H2 sp cache fixture used to demand OOM for every positive
budget through 8 MiB. Source-executed host diagnosis on #1719's `3fcd8698`
shows that expectation is stale: the 8-MiB force envelope assigns 5,594,904
bytes to values, while the unchanged native planner admits the complete small
owner at 4,335,341 bytes, including the conservative 4,836-byte source metadata
bound and eight-slot DIIS. Budgets through
4 MiB still reject. This admission result is shared by the RHF and UHF rank
hints; no allocation or numerical calculation is needed to establish it.

The native test now sends 8 MiB through its existing successful-endpoint arm,
including both item statuses, a non-null plan, exact resolved-budget identity,
and unchanged force parity. Infeasible budgets still require both OOM statuses
and a null old plan. The detailed failure message is preserved. A host
regression executes the native expectation predicate against the real resolver
and planner and fails with the former 8-MiB predicate.

No production, mathematical, resource-budget or acceptance-tolerance policy
changes. The old recorded native failure remains historical evidence. This
host diagnosis does not reconstruct its unrecorded returned statuses, establish
GPU cache lifetime, or claim a newly passing complete native CUDA suite. The
separate 8-MiB response allowance is 2,793,704 bytes; value admission alone
does not establish successful force response. Both native assertion arms now
report result count, both statuses and cached-plan state. A focused rerun of
`generativeqc_density_fitting_tests` is the remaining device verification; repeating
the 96/99-atom performance campaigns is unnecessary for this test correction.

Validation: 15 focused host resolver/preparation/predicate tests passed after
current-master integration (`9c54107c`). Rebuilding the same probe with the old
8-MiB rejection predicate fails at exactly the 8-MiB row. Both rank hints
produce the same admission numbers. No native CUDA binary was executed.
