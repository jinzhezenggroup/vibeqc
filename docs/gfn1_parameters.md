# GFN1-xTB canonical parameter snapshots

VibeQC keeps the checked-in GFN1-xTB scientific source snapshots with the
rest of the pinned upstream inputs:

`upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/`

The snapshot is pinned to xTBloom commit `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3` and contains:

- `gfn1.toml`: the tblite GFN1 parameter export retained by pinned xTBloom;
- `gfn1.json`: the normalized schema-v2 export consumed by VibeQC generators;
- `gfn1_manifest.json`: tblite/mctc-lib/exporter provenance and audited output identities.

The manifest traces the scientific source further to tblite revision
`fa8a4416e8fe093d0075bc10ac875494c2a449a9` and mctc-lib v0.5.2.

Regenerate and verify the runtime-neutral products with:

```sh
python tools/parameters/generate_gfn1.py
python tools/parameters/generate_gfn1.py --check
python tools/parameters/generate_gfn1_geometry.py
python tools/source_registry.py verify
```

The generated `src/xtb/gfn2_runtime/data/parameters/gfn1.hpp` and
`python/vibeqc_compiler/geometry/_gfn1_data.py` remain deterministic products
of the pinned source bytes. GFN1 D3 tables remain owned separately and are not
duplicated in this snapshot.
