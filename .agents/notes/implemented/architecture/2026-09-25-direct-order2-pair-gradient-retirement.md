# Retire the orphaned Direct order-two pair-gradient owner

The Direct-HF order-two and order-three force consumers now use compiler-owned
Weighted IntegralIR roots through `direct_force_order2.cuh` and
`direct_force_order3.cuh`. The legacy
`direct_native_pair_order2_gradient.cuh` helper has no remaining production
include or call site, so keeping it as a scientific ownership shard overstates
the native Direct surface and leaves duplicate derivative algebra available for
accidental reuse.

This change removes that orphaned helper and its ownership shard, updates the
Direct-HF retirement ledger, and adds a structural test that prevents the native
pair-gradient owner from being reintroduced. Runtime queues, screening,
resident/page schedules, and numerical policy are unchanged.

Validation is structural: repository search finds no production consumer for
the deleted helper, while the existing order-two force tests require
`generated_weighted_eri::{psps,ppss,dsss}_force` and the generated generic
order-two gradient path.

Agent: ChatGPT
Model: GPT-5.6 Sol
