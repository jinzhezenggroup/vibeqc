# Extending VibeQC

VibeQC does not currently promise a stable public plug-in ABI/API for third-party scientific extensions.

Future public extension contracts should define stable identity/versioning, inputs/outputs, scientific invariants and units, capability boundaries, state identity, validation requirements, failure behavior, and compatibility guarantees.

Reserved documentation slots:

- [Custom electronic methods](custom-method.md)
- [Custom XC functionals](custom-functional.md)
- [Custom backends](custom-backend.md)

Until these contracts are explicitly stabilized, internal Python/C++ classes are implementation details rather than public extension APIs.

```{toctree}
:hidden:
:maxdepth: 1

custom-method
custom-functional
custom-backend
```
