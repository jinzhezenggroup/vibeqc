# Scientific source registry

VibeQC keeps externally maintained scientific definitions and numerical data behind one pinned source registry: `sources/manifest.json`. The registry is the source of truth for upstream repository, immutable revision, upstream path, content digest, license, local canonical destination, and the checked-in products that depend on those inputs.

Normal configure, build, runtime, and test paths do not fetch the network. Network access is a maintainer action only.

## Repository layout

```text
sources/
├── manifest.json
├── upstream/   # byte-pinned upstream inputs, grouped by provider/revision
├── canonical/  # VibeQC-owned normalized/composite scientific inputs
└── derived/    # deterministic compatibility manifests generated from the registry
```

Generated runtime tables remain beside the code that consumes them (for example `src/dft/dispersion/*_data.hpp` and `python/vibeqc_compiler/integral/rys*_data.py`). They are products, not source-of-truth inputs.

## Source classes

- **Checked-in file sets** keep the audited upstream closure required to reproduce a product. The local root lives under `sources/upstream/<provider>/<revision>/`.
- **Checked-in snapshots** are compact upstream catalogs such as DFT-D4 and simple-DFTD3 parameter TOML files; they use the same provider/revision layout.
- **Canonical inputs** under `sources/canonical/` are VibeQC-owned normalized or composite scientific data whose identity is separately verified by the product registry.
- **Derived manifests** under `sources/derived/` are compatibility views rendered from `sources/manifest.json`; they must never become an independent source of truth.
- **Remote file sets** remain schema-supported for exceptional future inputs that are too large to vendor. They materialize only under `.cache/vibeqc-sources/` and must not be required for a clean-checkout regeneration path.

The current checked-in source closure covers Libxc, DFT-D4 reference inputs, EEQ/mctc-lib inputs, dispersion parameter snapshots, and simple-DFTD3 gCP inputs. The upstream GPU4PySCF Rys table remains the one remote-only source because its 1.32 MiB file exceeds VibeQC's non-waivable 1 MiB tracked-file retention limit; its deterministic generated products remain checked in. r2SCAN-3c canonical qualification data lives under `sources/canonical/r2scan3c/`.

## Commands

Run the complete offline integrity/freshness check:

```bash
python tools/source_registry.py verify
```

Regenerate registry-derived compatibility manifests:

```bash
python tools/source_registry.py regenerate
```

Restore one already-pinned checked-in source from upstream:

```bash
python tools/source_registry.py sync libxc-7.0.0
python tools/source_registry.py sync dftd4-reference
python tools/source_registry.py sync gpu4pyscf-rys
```

Move an existing allowlisted source closure to one explicitly selected commit or release tag:

```bash
python tools/source_registry.py update libxc-7.0.0 --revision 7.1.0
python tools/source_registry.py update gpu4pyscf-rys --revision <commit-sha>
```

`sync` refuses bytes whose digest differs from the registry. `update` reconstructs URLs from the registered repository and upstream paths, rejects obvious floating refs such as `main`, `master`, and `HEAD`, and records new raw and normalized digests. Neither command is used by a normal build.

Each generated product also records its source identity, generator identity, canonical inputs where applicable, and checked-in output hashes. Changing an upstream revision or file digest therefore makes `verify` fail until the affected product has been deliberately regenerated and reviewed.

## Libxc ownership

`libxc-7.0.0` owns one union inventory under `sources/upstream/libxc/7.0.0/` plus named `core`, `rsh`, and `wb97mv` collections. The compatibility `manifest.json`, `rsh-manifest.json`, and `wb97mv-manifest.json` files live under `sources/derived/libxc/7.0.0/` and are regenerated from the common registry.

Issue #739 continues to own Maple syntax admission, Graph lowering, functional qualification, and retirement of handwritten XC mathematics. The source registry owns acquisition, local source placement, and provenance identity.

## Updating an upstream source

An update is deliberately review-driven rather than automatic:

1. Choose an immutable upstream tag or commit; never use a floating branch.
2. Run `update <source> --revision <revision>` and inspect the source + registry diff.
3. If the admitted upstream file closure itself changed, edit that allowlist explicitly rather than silently widening it.
4. Regenerate the affected scientific products with their domain generator.
5. Refresh the reviewed product input identity and generator/output hashes in the registry.
6. Run `python tools/source_registry.py regenerate` and `verify`.
7. Run the domain numerical tests before review.

`update` intentionally does not refresh product identities for you: a source change must leave downstream products stale until its scientific regeneration has actually been reviewed. Build-time network access is not an accepted shortcut.
