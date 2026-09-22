# Capability sources

Do not maintain handwritten copies of generated capability tables.

The canonical public method registry is `manifests/public_methods.json`; tooling generates [the public method table](../public_methods.md).

Compiler shell/code-generation capability evidence is tracked in `docs/codegen_capabilities.json`. CUDA semantic ownership is tracked in `docs/cuda_ownership/`.

A low-level capability does not by itself prove end-to-end scientific qualification; user-facing support still requires the relevant validation and fail-closed execution contracts.

The generated [bulk Libxc import report](../libxc_bulk_import.md) and [Maple importer coverage](../libxc_maple_coverage.md) retain their registered paths.
