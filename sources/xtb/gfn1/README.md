# GFN1-xTB canonical parameter snapshots

This directory contains the small checked-in scientific snapshots used to
regenerate VibeQC's canonical GFN1 parameter table offline.

- `gfn1.toml`: tblite GFN1 parameter export retained by pinned xTBloom.
- `gfn1.json`: normalized schema-v2 export used by the VibeQC generator.
- `gfn1_manifest.json`: full tblite/mctc-lib/exporter provenance and audited
  output identities.

The snapshots come from xTBloom commit
`2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`. The manifest traces the
scientific source further to tblite revision
`fa8a4416e8fe093d0075bc10ac875494c2a449a9` and mctc-lib v0.5.2.

Regenerate and verify the runtime-neutral C++ table with:

```sh
python tools/parameters/generate_gfn1.py
python tools/parameters/generate_gfn1.py --check
python tools/source_registry.py verify
```

The generated `src/xtb/gfn2_runtime/data/parameters/gfn1.hpp` must remain
byte-identical to the audited upstream product. GFN1 D3 tables are intentionally
not duplicated here; VibeQC already owns their canonical data separately.

Agent: ChatGPT
Model: GPT-5.6 Sol
