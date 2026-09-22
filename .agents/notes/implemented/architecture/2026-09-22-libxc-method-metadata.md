# Decision: derive hybrid MethodIR metadata from pinned Libxc owners

Status: implemented

## Decision

Hybrid DFT method composition is generated from the pinned Libxc 7.0.0 C metadata owners instead of duplicating coefficients in `method/spec.py`. The extractor is fail-closed and evaluates only bounded arithmetic, `xc_mix_init` component lists, default external parameters, global/CAM exact-exchange metadata, and VV10 `b/C` constants. It never compiles or executes Libxc C.

`tools/generate_libxc_methods.py` reads the canonical scientific source registry, verifies each pinned hybrid owner digest, and emits `_generated_libxc_methods.py`. `MethodSpec` consumes every generated method whose semilocal components are already representable by the XC compiler. Representation admission remains separate from native/public runtime qualification.

This removes handwritten MethodSpec ownership for B3P86, B3LYP/B3LYP5, B3PW91, B5050LYP, CAM-B3LYP/CAMH-B3LYP and WB97M-V. Other representable pinned Libxc hybrids now enter MethodIR without a named catalog edit. Unsupported dynamic C composition remains explicit in `BLOCKED_LIBXC_METHODS` rather than being guessed.

## Invariants

- Updating a pinned Libxc hybrid owner makes the generated method product stale.
- Exact decimal source coefficients are evaluated as `Fraction`, avoiding binary-float drift.
- CAM uses Libxc semantics: short-range exact exchange is `alpha + beta`, long-range exact exchange is `alpha`.
- Missing semilocal primitives do not become MethodIR merely because hybrid metadata was parsed.
- MethodIR representation does not imply SCF, force, CUDA, performance, or public-Calculator qualification.
- Public method manifests and ABI IDs are unchanged by this slice.

## Evidence

- `python tools/generate_libxc_methods.py --check`
- `python tools/source_registry.py verify`
- `pytest tests/python/test_libxc_method_metadata.py`
- existing DFT MethodIR / KS execution-plan / Libxc bulk-metadata regressions
- existing WB97M-V / RSH / nonlocal-hybrid plan regressions

Refs #396 #167 #739 #745

Agent: ChatGPT
Model: GPT-5.6 Sol
