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


## Review qualification and fail-closed initialization

The 2026-09-23 review reproduced silently ignored initializer calls, CAM writes,
VV10 compound assignments and mix-array mutations. Initializers now admit only
complete statements in the modeled subset; unmodeled work blocks that record.
The setter already applied this rule. Generated component sequences are immutable,
and the product identity includes the shared C-metadata parser as well as the
hybrid extractor. Regeneration after the master integration refreshes the source
registry identity rather than retaining stale hashes.

All 28 generated records were independently initialized through the separately
compiled Libxc 7.0.0 C API in both spin modes (56 initializations). Component IDs
and nonzero coefficients, global/CAM exchange and VV10 parameters agree within
an absolute 2e-15 gate; the largest difference is 1.1102230246251565e-16.
The reference library came from PySCF 2.14.0, SHA-256
`d120371b37f4729dfc21f3b0575e563ef9b0254839926c41c0c02f5082143548`.
Permanent tests also compare all 28 records with Libxc's exchange/VV10 APIs,
using numeric IDs to avoid PySCF's configurable B3LYP alias.

All 33 pre-existing MethodSpec payloads remain identical to master 3a64fe25,
including B3LYP's VWN-RPA convention and WB97M-V's nonlocal contract. The selected
metadata, MethodIR, fixed-density exchange/VV10, bulk inventory, registry and
production-policy suite passes 624 cases with no skips. The CPU native library
is reused only for unchanged basis/integral consumers; generated Python metadata
and compiler paths execute from this source. These are representation and
fixed-density qualifications, without new SCF/public-capability admission.

The same 624 cases pass after integrating master e9fa40b1, including the reviewed
packed-XC donation and molecular-quadrature changes.
