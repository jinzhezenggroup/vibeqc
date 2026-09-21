# Production registry serialization ownership

Issue: #487

The integral production refactor now gives generated registry serialization a dedicated leaf owner:

```text
production_cost.py       shell-class identity and stable shard policy
        |                         |
        v                         v
production_selection.py      production_profile.py
        \                         /
         \                       /
          v                     v
             production_registry.py
                      |
                      v
                 production.py
        source emission + bundle orchestration
```

`production_registry.py` owns the single-profile and multi-profile registry header/source serializers and their launch-signature helpers. It imports stable selection/profile/cost contracts, but never imports `production.py`. The orchestration module consumes and compatibility-re-exports those serializers, so existing repository imports preserve object identity while new code has an unambiguous registry owner.

Two shared helpers were moved to the lower-level owners that already define their semantics: compatibility selection normalization/canonical ordering now live in `production_selection.py`, and stable profile identifiers live in `production_profile.py`. This avoids a reverse dependency from registry serialization back into source emission.

The change is intentionally behavior-neutral. On the exact pre-change master, SHA-256 identities were captured for the single-profile registry header/source, multi-profile registry header/source, an 8-shard sm_120 bundle, and an 8-shard sm_80+sm_120 bundle. After extraction all six identities are byte-for-byte unchanged.

Remaining #487 work is source-emission and filesystem/bundle-orchestration ownership; this slice does not claim issue closure.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Review correction: retain the leaf boundary and compatibility surface

Canonical ordering consumes shell-class indexing from `production_cost`, so its
owner is `production_registry`, not the lower-level `production_selection`.
The latter retains compatibility normalization without acquiring a cost-policy
dependency. This supersedes the ordering edge in the initial diagram above.
All six moved launch-signature/argument helpers are identity-preserving
re-exports from `production`, including the existing streaming-Fock caller.
The pre-existing leaf-dependency and generated-signature tests stay enabled.

Agent: ChatGPT
Model: GPT-6 Astra Pro
