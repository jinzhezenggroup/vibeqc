"""Repository-wide upstream scientific source registry contracts."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import source_registry


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_repository_text_digest_is_checkout_eol_stable(tmp_path: Path) -> None:
    path = tmp_path / "generated.py"
    path.write_bytes(b"first\r\nsecond\r\n")
    assert source_registry._repository_text_sha256(path) == _digest(b"first\nsecond\n")


def test_checked_in_registry_is_offline_verifiable() -> None:
    summary = source_registry.verify()
    assert summary["sources"] >= 7
    assert summary["local_files"] >= 43
    assert summary["products"] >= 5
    assert summary["derived_manifests"] == 4


REMOTE_REGENERATION_SOURCES = {
    "dftd4-reference",
    "gpu4pyscf-rys",
    "mctc-lib-eeq",
    "multicharge-eeq2019",
    "simple-dftd3-gcp",
}


def test_checked_in_sources_are_vendored_under_upstream_tree() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    for source_id, source in registry["sources"].items():
        if "local_root" not in source:
            continue
        assert source["kind"] != "remote-file-set", source_id
        assert source["local_root"].startswith("upstream/"), source_id
        root = source_registry.ROOT / source["local_root"]
        for name in source["files"]:
            assert (root / name).is_file()


def test_large_or_qualification_only_sources_stay_remote() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    for source_id in REMOTE_REGENERATION_SOURCES:
        source = registry["sources"][source_id]
        assert source["kind"] == "remote-file-set"
        assert "local_root" not in source


def test_libxc_registry_owns_every_pinned_source_file() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    source = registry["sources"]["libxc-7.0.0"]
    root = source_registry.ROOT / source["local_root"]
    derived = {
        Path(path).name
        for path, spec in registry["derived_manifests"].items()
        if spec.get("source") == "libxc-7.0.0"
    }
    actual = {path.name for path in root.iterdir() if path.is_file()} - derived
    assert actual == set(source["files"])
    assert set(source["collections"]) == {"core", "rsh", "wb97mv"}


def test_libxc_importer_semantics_are_pinned_separately() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    admission = registry["sources"]["libxc-7.0.0"]["admission"]
    importer = source_registry.ROOT / admission["importer"]
    from vibeqc_compiler.xc.libxc_maple import IMPORTER_SEMANTICS

    assert admission["semantics"] == IMPORTER_SEMANTICS
    assert admission["importer_sha256"] == source_registry._repository_text_sha256(
        importer
    )
    assert registry["products"]["libxc-xc-admission"]["inputs"] == ["libxc-7.0.0"]


def test_gcp_canonical_input_matches_registered_upstream_provenance() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    source = registry["sources"]["simple-dftd3-gcp"]
    data = json.loads(
        (source_registry.ROOT / "tools/parameters/r2scan3c_gcp.json").read_text()
    )
    upstream = data["upstream"]
    assert upstream["commit"] == source["revision"]
    assert upstream["license"] == source["license"]
    hashes = {
        item["upstream_path"]: item["sha256"] for item in source["files"].values()
    }
    assert hashes["src/dftd3/gcp/param.f90"] == upstream["param_sha256"]
    assert hashes["src/dftd3/gcp.f90"] == upstream["implementation_sha256"]
    assert hashes["src/dftd3/data/vdwrad.f90"] == upstream["vdwrad_sha256"]


def test_dispersion_generators_consume_common_registry_source_identity() -> None:
    contracts = {
        "tools/parameters/generate_d4.py": (
            "DFTD4_REPOSITORY",
            "DFTD4_LICENSE",
            "MCTC_REVISION",
            "MCTC_TREE",
            "SOURCE_PATHS",
            "--source-git-dir",
            '"--revision"',
        ),
        "tools/parameters/generate_d4_eeq.py": (
            "DFTD4_REVISION",
            "MULTICHARGE_REVISION",
            "MCTC_REVISION",
            "MULTICHARGE_SOURCE",
            "--dftd4-git-dir",
            "--multicharge-git-dir",
            "--mctc-git-dir",
        ),
        "tools/parameters/generate_gcp_r2scan3c.py": (
            "41d5a07b98ce15e97bec7a1815869725f6c7b0c2",
            '"--source"',
        ),
    }
    for relative, forbidden in contracts.items():
        text = (source_registry.ROOT / relative).read_text(encoding="utf-8")
        assert "load_product_sources(" in text, relative
        assert not any(token in text for token in forbidden), relative


def test_gcp_generator_regenerates_checked_in_header_byte_for_byte() -> None:
    from tools.parameters import generate_gcp_r2scan3c

    sources = source_registry.load_product_sources(
        "r2scan3c-gcp",
        generator=Path(generate_gcp_r2scan3c.__file__),
        expected_inputs=("simple-dftd3-gcp",),
        expected_canonical_inputs=("tools/parameters/r2scan3c_gcp.json",),
    )
    data = json.loads(generate_gcp_r2scan3c.SOURCE.read_text(encoding="utf-8"))
    regenerated = generate_gcp_r2scan3c.render(data, sources["simple-dftd3-gcp"])
    assert regenerated == generate_gcp_r2scan3c.OUTPUT.read_text(encoding="utf-8")


def test_registry_derived_manifests_are_byte_stable() -> None:
    registry = source_registry._load()
    for relative, spec in registry["derived_manifests"].items():
        path = source_registry.ROOT / relative
        assert path.read_text(
            encoding="utf-8"
        ) == source_registry.render_derived_manifest(registry, spec)


def test_sync_is_pinned_and_normalizes_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = b"alpha  \n beta\t\n"
    normalized = b"alpha\n beta\n"
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "local_root": "upstream/sample",
                "files": {
                    "data.txt": {
                        "upstream_path": "data.txt",
                        "url": "https://example.invalid/sample/rev-123/data.txt",
                        "upstream_sha256": _digest(upstream),
                        "sha256": _digest(normalized),
                        "normalization": "trailing-whitespace-only",
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.urllib.request,
        "urlopen",
        lambda request, timeout: contextlib.closing(io.BytesIO(upstream)),
    )
    written = source_registry.sync_source("sample", registry_path)
    assert written == [tmp_path / "upstream/sample/data.txt"]
    assert written[0].read_bytes() == normalized
    assert source_registry.verify(registry_path)["local_files"] == 1


def test_sync_refuses_unexpected_upstream_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"expected\n"
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "remote-file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "files": {
                    "data.txt": {
                        "upstream_path": "data.txt",
                        "url": "https://example.invalid/sample/rev-123/data.txt",
                        "sha256": _digest(expected),
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.urllib.request,
        "urlopen",
        lambda request, timeout: contextlib.closing(io.BytesIO(b"changed upstream\n")),
    )
    with pytest.raises(source_registry.SourceRegistryError, match="upstream digest"):
        source_registry.sync_source(
            "sample", registry_path, cache_root=tmp_path / ".cache/vibeqc-sources"
        )
    assert not (tmp_path / ".cache/vibeqc-sources/sample/data.txt").exists()


def test_registry_rejects_floating_or_unsafe_source_paths(tmp_path: Path) -> None:
    payload = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "local_root": "../escape",
                "files": {
                    "data.txt": {
                        "url": "https://example.invalid/sample/main/data.txt",
                        "sha256": "0" * 64,
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(source_registry.SourceRegistryError):
        source_registry.verify(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_blob", 123, "invalid Git blob identity"),
        ("size", True, "invalid byte size"),
    ],
)
def test_registry_rejects_invalid_optional_file_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    item = {
        "url": "https://example.invalid/sample/rev-123/data.txt",
        "sha256": "0" * 64,
        field: value,
    }
    payload = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "remote-file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "files": {"data.txt": item},
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(source_registry.SourceRegistryError, match=message):
        source_registry.verify(path)


def test_update_requires_explicit_revision_and_invalidates_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"old\n"
    candidate = b"new\n"
    source = {
        "kind": "file-set",
        "repository": "https://github.com/example/sample",
        "revision": "rev-123",
        "license": "MIT",
        "git_tree": "a" * 40,
        "local_root": "upstream/sample",
        "files": {
            "data.txt": {
                "upstream_path": "data.txt",
                "url": "https://raw.githubusercontent.com/example/sample/rev-123/data.txt",
                "sha256": _digest(original),
                "git_blob": source_registry._git_blob_sha1(original),
                "size": len(original),
            }
        },
    }
    generator_bytes = b"generator\n"
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {"sample": source},
        "products": {
            "derived": {
                "inputs": ["sample"],
                "input_identity_sha256": source_registry._product_input_identity(
                    {"sample": source}, ["sample"]
                ),
                "generator": "tools/generator.py",
                "generator_sha256": _digest(generator_bytes),
                "canonical_inputs": {},
                "outputs": {},
            }
        },
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    local = tmp_path / "upstream/sample/data.txt"
    local.parent.mkdir(parents=True)
    local.write_bytes(original)
    generator = tmp_path / "tools/generator.py"
    generator.parent.mkdir(parents=True)
    generator.write_bytes(generator_bytes)
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.urllib.request,
        "urlopen",
        lambda request, timeout: contextlib.closing(io.BytesIO(candidate)),
    )

    with pytest.raises(source_registry.SourceRegistryError, match="immutable"):
        source_registry.update_source("sample", "master", registry_path)

    with pytest.raises(source_registry.SourceRegistryError, match="Git tree identity"):
        source_registry.update_source("sample", "rev-456", registry_path)

    written = source_registry.update_source(
        "sample", "rev-456", registry_path, git_tree="b" * 40
    )
    assert written == [local]
    assert local.read_bytes() == candidate
    updated = json.loads(registry_path.read_text())
    item = updated["sources"]["sample"]["files"]["data.txt"]
    assert updated["sources"]["sample"]["revision"] == "rev-456"
    assert updated["sources"]["sample"]["git_tree"] == "b" * 40
    assert item["sha256"] == _digest(candidate)
    assert item["git_blob"] == source_registry._git_blob_sha1(candidate)
    assert item["size"] == len(candidate)
    assert "/rev-456/data.txt" in item["url"]
    with pytest.raises(
        source_registry.SourceRegistryError, match="source inputs are stale"
    ):
        source_registry.verify(registry_path)

    assert (
        source_registry.stage_product_inputs("derived", registry_path) == registry_path
    )
    sources = source_registry.load_product_sources(
        "derived",
        generator=generator,
        expected_inputs=("sample",),
        expected_canonical_inputs=(),
        registry_path=registry_path,
    )
    assert sources["sample"]["revision"] == "rev-456"
    assert source_registry.verify(registry_path)["products"] == 1


@pytest.mark.parametrize(
    "mutation",
    ("missing", "revision", "hash", "git_blob", "git_tree", "collection"),
)
def test_product_source_binding_fails_closed_on_registry_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    payload = b"registered scientific source\n"
    generator_bytes = b"generator\n"
    source = {
        "kind": "remote-file-set",
        "repository": "https://github.com/example/source",
        "revision": "rev-123",
        "license": "MIT",
        "git_tree": "a" * 40,
        "collections": {"generator": ["data.txt"]},
        "files": {
            "data.txt": {
                "upstream_path": "data.txt",
                "url": "https://raw.githubusercontent.com/example/source/rev-123/data.txt",
                "sha256": _digest(payload),
                "git_blob": source_registry._git_blob_sha1(payload),
            }
        },
    }
    product = {
        "inputs": ["sample"],
        "input_identity_sha256": source_registry._product_input_identity(
            {"sample": source}, ["sample"]
        ),
        "generator": "tools/generator.py",
        "generator_sha256": _digest(generator_bytes),
        "canonical_inputs": {},
        "outputs": {},
    }
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {"sample": source},
        "products": {"derived": product},
        "derived_manifests": {},
    }
    if mutation == "missing":
        del registry["sources"]["sample"]
    elif mutation == "revision":
        registry["sources"]["sample"]["revision"] = "rev-456"
    elif mutation == "hash":
        registry["sources"]["sample"]["files"]["data.txt"]["sha256"] = "f" * 64
    elif mutation == "git_blob":
        registry["sources"]["sample"]["files"]["data.txt"]["git_blob"] = "f" * 40
    elif mutation == "git_tree":
        registry["sources"]["sample"]["git_tree"] = "f" * 40
    else:
        registry["sources"]["sample"]["collections"] = {"generator": []}
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    generator = tmp_path / "tools/generator.py"
    generator.parent.mkdir(parents=True)
    generator.write_bytes(generator_bytes)
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)

    with pytest.raises(source_registry.SourceRegistryError):
        source_registry.load_product_sources(
            "derived",
            generator=generator,
            expected_inputs=("sample",),
            expected_canonical_inputs=(),
            registry_path=registry_path,
        )


def test_product_source_reader_requires_registered_cached_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"registered scientific source\n"
    generator_bytes = b"generator\n"
    source = {
        "kind": "remote-file-set",
        "repository": "https://github.com/example/source",
        "revision": "rev-123",
        "license": "MIT",
        "collections": {"generator": ["data.txt"]},
        "files": {
            "data.txt": {
                "upstream_path": "data.txt",
                "url": "https://raw.githubusercontent.com/example/source/rev-123/data.txt",
                "sha256": _digest(payload),
                "git_blob": source_registry._git_blob_sha1(payload),
            }
        },
    }
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {"sample": source},
        "products": {
            "derived": {
                "inputs": ["sample"],
                "input_identity_sha256": source_registry._product_input_identity(
                    {"sample": source}, ["sample"]
                ),
                "generator": "tools/generator.py",
                "generator_sha256": _digest(generator_bytes),
                "canonical_inputs": {},
                "outputs": {},
            }
        },
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    cache = tmp_path / "cache"
    cached = cache / "sample/data.txt"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payload)
    generator = tmp_path / "tools/generator.py"
    generator.parent.mkdir(parents=True)
    generator.write_bytes(generator_bytes)
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)

    sources = source_registry.load_product_sources(
        "derived",
        generator=generator,
        expected_inputs=("sample",),
        expected_canonical_inputs=(),
        registry_path=registry_path,
    )
    assert source_registry.read_source_texts(
        "sample", sources["sample"], cache_root=cache, collection="generator"
    ) == {"data.txt": payload.decode()}

    sources["sample"]["files"]["data.txt"]["git_blob"] = "f" * 40
    with pytest.raises(source_registry.SourceRegistryError, match="Git blob mismatch"):
        source_registry.read_source_texts(
            "sample", sources["sample"], cache_root=cache, collection="generator"
        )
    sources["sample"]["files"]["data.txt"]["git_blob"] = source_registry._git_blob_sha1(
        payload
    )
    cached.write_bytes(b"unregistered replacement\n")
    with pytest.raises(source_registry.SourceRegistryError, match="digest mismatch"):
        source_registry.read_source_texts(
            "sample", sources["sample"], cache_root=cache, collection="generator"
        )


def test_verify_rejects_stale_generator_and_product_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bytes = b"source\n"
    generator_bytes = b"generator\n"
    canonical_bytes = b"canonical input\n"
    output_bytes = b"output\n"
    source_path = tmp_path / "upstream/sample/data.txt"
    generator_path = tmp_path / "tools/generator.py"
    canonical_path = tmp_path / "inputs/canonical.json"
    output_path = tmp_path / "generated/output.txt"
    for path, data in (
        (source_path, source_bytes),
        (generator_path, generator_bytes),
        (canonical_path, canonical_bytes),
        (output_path, output_bytes),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source = {
        "kind": "snapshot",
        "repository": "https://github.com/example/source",
        "revision": "rev-123",
        "license": "MIT",
        "local_root": "upstream/sample",
        "files": {
            "data.txt": {
                "upstream_path": "data.txt",
                "url": "https://raw.githubusercontent.com/example/source/rev-123/data.txt",
                "sha256": _digest(source_bytes),
            }
        },
    }
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {"sample": source},
        "products": {
            "derived": {
                "inputs": ["sample"],
                "input_identity_sha256": source_registry._product_input_identity(
                    {"sample": source}, ["sample"]
                ),
                "generator": "tools/generator.py",
                "generator_sha256": _digest(generator_bytes),
                "canonical_inputs": {"inputs/canonical.json": _digest(canonical_bytes)},
                "outputs": {"generated/output.txt": _digest(output_bytes)},
            }
        },
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    assert source_registry.verify(registry_path)["product_files"] == 2
    assert source_registry.load_product_sources(
        "derived",
        generator=generator_path,
        expected_inputs=("sample",),
        expected_canonical_inputs=("inputs/canonical.json",),
        registry_path=registry_path,
    ) == {"sample": source}

    generator_path.write_bytes(b"stale generator\n")
    with pytest.raises(source_registry.SourceRegistryError, match="generator digest"):
        source_registry.load_product_sources(
            "derived",
            generator=generator_path,
            expected_inputs=("sample",),
            expected_canonical_inputs=("inputs/canonical.json",),
            registry_path=registry_path,
        )
    with pytest.raises(source_registry.SourceRegistryError, match="generator digest"):
        source_registry.verify(registry_path)
    generator_path.write_bytes(generator_bytes)
    canonical_path.write_bytes(b"altered canonical input\n")
    with pytest.raises(
        source_registry.SourceRegistryError, match="canonical input digest"
    ):
        source_registry.load_product_sources(
            "derived",
            generator=generator_path,
            expected_inputs=("sample",),
            expected_canonical_inputs=("inputs/canonical.json",),
            registry_path=registry_path,
        )
    canonical_path.write_bytes(canonical_bytes)
    output_path.write_bytes(b"stale product\n")
    with pytest.raises(source_registry.SourceRegistryError, match="product digest"):
        source_registry.verify(registry_path)
