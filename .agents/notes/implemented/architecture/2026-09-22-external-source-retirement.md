# Retire the mixed external scientific asset tree

## Decision

Remove the top-level `external/` tree and make ownership explicit:

- immutable third-party scientific bytes live under `upstream/<provider>/<revision>/`;
- derived audit/compatibility metadata lives under `manifests/`;
- repository generator inputs owned by VibeQC live beside their generators under `tools/parameters/`;
- retained third-party legal texts live under `LICENSES/`.

Libxc adapters now require the canonical `upstream/libxc/7.0.0` source closure instead
of falling back to a legacy source tree. Its compatibility manifests move to
`manifests/libxc/7.0.0/` without changing their bytes.

Pinned xTBloom GFN1/D3 bytes move into the existing revisioned upstream snapshot.
The redundant covalent-radii JSON is retired: consumers derive the same ordered
86-element table from `gfn1.json` and verify its historical SHA-256 identity.

The r2SCAN-3c audit manifest/BSE export move to `manifests/r2scan3c/`; the compact
gCP generator input moves to `tools/parameters/r2scan3c_gcp.json`.

## Contracts

Normal build, runtime, and tests remain offline. `upstream/manifest.json` remains
the source of truth for source revision/digest/license identity and generated products.
Compiler asset layout version advances to 3 so pre-migration caches cannot alias the new
wheel/repository paths.

Validation: source-registry verify; CPU `vibeqc` build; 430 Python tests passed with
9 expected skips; native D3 ATM/zero, r2SCAN-3c gCP, and XC point tests passed.

Agent: ChatGPT
Model: GPT-5.6 Sol
