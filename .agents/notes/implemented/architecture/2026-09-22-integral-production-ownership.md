# Integral production ownership split

Issue #487 is closed structurally by moving the remaining source-emission and filesystem/bundle responsibilities out of `integral.production`.

The dependency direction is now:

```text
production (compatibility facade)
  -> production_bundle -> production_emission
  -> production_registry / production_profile / production_selection / production_cost
production_bundle
  -> production_emission
  -> production_registry / production_profile / production_cost / shell_spec
production_emission
  -> codegen/lowering + production_selection/profile/cost
```

Leaf policy owners do not import the compatibility facade. `production_emission` owns deterministic CUDA source text and has no filesystem writes; `production_bundle` owns directories, stable shard layout, file replacement, and registry artifact publication. Registry serialization, profile parsing, selection policy, and compile-cost partitioning remain in their previously split owners.

Compatibility imports from `integral.production` retain object identity so existing build tools and downstream tests do not need an eager migration. The compatibility facade contains no source-generation or filesystem implementation.
