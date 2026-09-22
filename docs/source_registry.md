# Scientific source registry

VibeQC keeps externally maintained scientific definitions and numerical data behind one pinned source registry: `upstream/manifest.json`. The registry is the source of truth for upstream repository, immutable revision, file path, content digest, license, and the checked-in products that depend on those inputs.

Normal configure, build, runtime, and test paths do not fetch the network. Network access is a maintainer action only.

## Source classes

- **Checked-in file sets** keep the audited upstream bytes needed for deterministic regeneration under `upstream/<provider>/<revision>/`. The registry's `local_root` is the only canonical repository location for those bytes.
- **Checked-in snapshots** keep compact upstream catalogs needed by deterministic generators, such as the simple-DFTD3 and DFT-D4 parameter TOML files and the pinned xTBloom GFN1-xTB parameter export.
- **Remote file sets** pin upstream implementation/qualification sources that are not needed by normal compiler/catalog work. `sync` materializes these under `.cache/vibeqc-sources/` only for maintainer regeneration. Large DFT-D4 reference data, EEQ implementation sources, simple-DFTD3 gCP implementation sources, and the GPU4PySCF Rys source table stay remote-only.
- **Products** record deterministic generators, their hashes, canonical inputs where applicable, and checked-in output hashes. Generated tables are products, not source-of-truth definitions.

The current registry covers Libxc, xTBloom GFN1-xTB parameter snapshots, DFT-D4 reference inputs, EEQ/mctc-lib inputs, dispersion parameter snapshots, GPU4PySCF Rys tables, r2SCAN-3c gCP data, and VibeQC-generated high-accuracy Rys coefficients.

## Commands

Run the complete offline integrity/freshness check:

```bash
python tools/source_registry.py verify
```

Regenerate compatibility manifests that older domain code still consumes:

```bash
python tools/source_registry.py regenerate
```

Materialize one already-pinned upstream source explicitly:

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

Checked-in sources are restored or updated at their registered `upstream/...` `local_root`. Future remote-only source sets are written below `.cache/vibeqc-sources/<source-id>/`. `sync` refuses bytes whose digest differs from the registry. `update` reconstructs URLs from the registered repository and upstream paths, rejects obvious floating refs such as `main`, `master`, and `HEAD`, and records new raw and normalized digests. Neither command is used by a normal build.

Each generated product also records an input-source identity. Changing an upstream revision or file digest therefore makes `verify` fail until the affected product is deliberately regenerated and its reviewed identity/output hashes are refreshed.

## Libxc ownership

`libxc-7.0.0` owns one union inventory under `upstream/libxc/7.0.0/` plus named `core`, `rsh`, and `wb97mv` collections. The historical `manifest.json`, `rsh-manifest.json`, and `wb97mv-manifest.json` files are derived compatibility views. Their bytes are regenerated from the common registry so existing compiler artifact identities do not change merely because provenance ownership moved.

Issue #739 continues to own Maple syntax admission, Graph lowering, functional qualification, and retirement of handwritten XC mathematics. The source registry only owns acquisition and provenance identity.

## Updating an upstream source

An update is deliberately review-driven rather than automatic:

1. Choose an immutable upstream tag or commit; never use a floating branch.
2. Run `update <source> --revision <revision>` and inspect the source + registry diff.
3. If the admitted upstream file closure itself changed, edit that allowlist explicitly rather than silently widening it.
4. Regenerate the affected scientific products with their domain generator.
5. Refresh the reviewed product input identity and generator/output hashes in the registry.
6. Run `python tools/source_registry.py regenerate` and `verify`.
7. Run the domain numerical tests before review.

`update` intentionally does not refresh product identities for you: a source change must leave downstream products stale until their scientific regeneration has actually been reviewed. Build-time network access is not an accepted shortcut.
