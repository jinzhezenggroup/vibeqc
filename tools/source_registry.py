#!/usr/bin/env python3
"""Verify, regenerate, and explicitly synchronize pinned scientific sources.

Ordinary VibeQC builds never use the network.  This maintainer tool makes the
repository-wide source registry the single ownership point for upstream
revision/path/hash/license metadata while keeping domain artifacts checked in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "sources/manifest.json"
DEFAULT_CACHE = ROOT / ".cache/vibeqc-sources"
_SCHEMA = "vibeqc.scientific-source-registry"


class SourceRegistryError(ValueError):
    """The source registry or one of its pinned artifacts is inconsistent."""


def _load(path: Path = REGISTRY) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("source registry must contain a JSON object")
    if value.get("schema") != _SCHEMA or value.get("schema_version") != 1:
        raise SourceRegistryError("unsupported scientific source registry schema")
    for key in ("sources", "products", "derived_manifests"):
        if not isinstance(value.get(key), dict):
            raise TypeError(f"source registry {key!r} must be an object")
    return value


def _relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SourceRegistryError(f"{label} must stay inside the repository/cache")
    return path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _check_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SourceRegistryError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SourceRegistryError(f"{label} must be hexadecimal") from error
    return value.lower()


def _inside(root: Path, relative: Path, *, label: str) -> Path:
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise SourceRegistryError(
            f"{label} must stay inside its selected root"
        ) from error
    return candidate


def _repository_path(value: Any, *, label: str) -> Path:
    return _inside(ROOT, _relative_path(value, label=label), label=label)


def _source_file_destination(
    source_id: str, source: dict[str, Any], name: str, cache_root: Path
) -> Path:
    identifier = _relative_path(source_id, label="source id")
    if len(identifier.parts) != 1:
        raise SourceRegistryError("source id must be one relative path component")
    local_root = source.get("local_root")
    if local_root is not None:
        relative = _relative_path(
            local_root, label=f"{source_id}.local_root"
        ) / _relative_path(name, label=f"{source_id}.files[{name!r}]")
        return _inside(ROOT, relative, label="source destination")
    item = source["files"][name]
    upstream = _relative_path(
        item.get("upstream_path", name), label=f"{source_id}.{name}.upstream_path"
    )
    return _inside(cache_root, identifier / upstream, label="source cache destination")


def _normalize(data: bytes, rule: Any) -> bytes:
    if rule is None:
        return data
    if rule != "trailing-whitespace-only":
        raise SourceRegistryError(f"unsupported source normalization {rule!r}")
    text = data.decode("utf-8")
    had_final_newline = text.endswith("\n")
    lines = text.splitlines()
    normalized = "\n".join(line.rstrip(" \t\r") for line in lines)
    if had_final_newline:
        normalized += "\n"
    return normalized.encode("utf-8")


def _source_identity(source: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that define the scientific upstream identity."""
    files: dict[str, Any] = {}
    for name, item in source["files"].items():
        record = {
            "upstream_path": item.get("upstream_path", name),
            "sha256": item["sha256"],
        }
        for optional in ("upstream_sha256", "normalization"):
            if optional in item:
                record[optional] = item[optional]
        files[name] = record
    return {
        "repository": source["repository"],
        "revision": source["revision"],
        "license": source["license"],
        "files": files,
    }


def _product_input_identity(sources: dict[str, Any], source_ids: list[str]) -> str:
    payload = {
        source_id: _source_identity(sources[source_id]) for source_id in source_ids
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _validate_source_metadata(source_id: str, source: Any) -> None:
    if not isinstance(source, dict):
        raise TypeError(f"source {source_id!r} must be an object")
    for field in ("kind", "repository", "revision", "license", "files"):
        if field not in source:
            raise SourceRegistryError(f"source {source_id!r} is missing {field!r}")
    if source["kind"] not in {"file-set", "snapshot", "remote-file-set"}:
        raise SourceRegistryError(f"source {source_id!r} has unsupported kind")
    if not all(
        isinstance(source[field], str) and source[field]
        for field in ("repository", "revision", "license")
    ):
        raise TypeError(f"source {source_id!r} identity fields must be strings")
    if _checked_revision(source["revision"]) != source["revision"]:
        raise SourceRegistryError("registered upstream revision must be canonical")
    if not isinstance(source["files"], dict) or not source["files"]:
        raise SourceRegistryError(f"source {source_id!r} must pin at least one file")
    for name, item in source["files"].items():
        _relative_path(name, label=f"{source_id}.file")
        if not isinstance(item, dict):
            raise TypeError(f"source {source_id!r} file {name!r} must be an object")
        _check_digest(item.get("sha256"), label=f"{source_id}.{name}.sha256")
        url = item.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise SourceRegistryError(
                f"source {source_id!r} file {name!r} needs an HTTPS URL"
            )
        if source["revision"] not in url:
            raise SourceRegistryError(
                f"source {source_id!r} file {name!r} URL does not pin revision {source['revision']!r}"
            )
        if "upstream_sha256" in item:
            _check_digest(
                item["upstream_sha256"], label=f"{source_id}.{name}.upstream_sha256"
            )
        _normalize(b"", item.get("normalization"))

    admission = source.get("admission")
    if admission is not None:
        if not isinstance(admission, dict):
            raise TypeError(f"source {source_id!r} admission must be an object")
        importer = _repository_path(
            admission.get("importer"), label=f"{source_id}.admission.importer"
        )
        semantics = admission.get("semantics")
        if not isinstance(semantics, str) or not semantics:
            raise SourceRegistryError(
                f"source {source_id!r} admission requires importer semantics"
            )
        if not importer.is_file():
            raise FileNotFoundError(importer)
        expected = _check_digest(
            admission.get("importer_sha256"),
            label=f"{source_id}.admission.importer_sha256",
        )
        if _sha256(importer) != expected:
            raise SourceRegistryError(f"source importer digest mismatch: {importer}")


def _render_libxc_collection(
    registry: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    source = registry["sources"][spec["source"]]
    collection = source.get("collections", {}).get(spec["collection"])
    if not isinstance(collection, list) or not collection:
        raise SourceRegistryError("Libxc derived manifest names an empty collection")
    files: dict[str, Any] = {}
    for name in collection:
        item = source["files"].get(name)
        if not isinstance(item, dict):
            raise SourceRegistryError(
                f"Libxc collection references unknown file {name!r}"
            )
        rendered = {"url": item["url"], "sha256": item["sha256"]}
        for optional in ("upstream_sha256", "normalization"):
            if optional in item:
                rendered[optional] = item[optional]
        files[name] = rendered
    return {"version": source["revision"], "license": source["license"], "files": files}


def _render_dispersion_sources(
    registry: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    rendered: dict[str, Any] = {"schema_version": 1}
    for public_name in ("simple_dftd3", "dftd4"):
        source_id = spec["sources"][public_name]
        source = registry["sources"][source_id]
        if len(source["files"]) != 1 or "local_root" not in source:
            raise SourceRegistryError(
                f"{source_id!r} is not a single checked-in snapshot"
            )
        filename, item = next(iter(source["files"].items()))
        repository = source["repository"].removeprefix("https://github.com/")
        rendered[public_name] = {
            "repository": repository,
            "revision": source["revision"],
            "path": item["upstream_path"],
            "snapshot": (Path(source["local_root"]) / filename).as_posix(),
            "sha256": item["sha256"],
            "license": source["license"],
        }
    return rendered


def render_derived_manifest(registry: dict[str, Any], spec: dict[str, Any]) -> str:
    renderer = spec.get("renderer")
    if renderer == "libxc-collection":
        value = _render_libxc_collection(registry, spec)
    elif renderer == "dispersion-parameter-sources":
        value = _render_dispersion_sources(registry, spec)
    else:
        raise SourceRegistryError(f"unknown derived-manifest renderer {renderer!r}")
    text = json.dumps(value, indent=2, ensure_ascii=False)
    return text + ("\n" if spec.get("final_newline", True) else "")


def regenerate(registry_path: Path = REGISTRY) -> list[Path]:
    registry = _load(registry_path)
    written: list[Path] = []
    for relative, spec in registry["derived_manifests"].items():
        path = _repository_path(relative, label="derived manifest path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_derived_manifest(registry, spec), encoding="utf-8", newline="\n"
        )
        written.append(path)
    return written


def verify(
    registry_path: Path = REGISTRY, *, cache_root: Path = DEFAULT_CACHE
) -> dict[str, int]:
    registry = _load(registry_path)
    sources = registry["sources"]
    local_files = 0
    cached_files = 0
    for source_id, source in sources.items():
        _validate_source_metadata(source_id, source)
        for name, item in source["files"].items():
            expected = _check_digest(item["sha256"], label=f"{source_id}.{name}.sha256")
            destination = _source_file_destination(source_id, source, name, cache_root)
            if "local_root" in source:
                if not destination.is_file():
                    raise FileNotFoundError(destination)
                if _sha256(destination) != expected:
                    raise SourceRegistryError(
                        f"pinned source digest mismatch: {destination}"
                    )
                local_files += 1
            elif destination.exists():
                if _sha256(destination) != expected:
                    raise SourceRegistryError(
                        f"cached source digest mismatch: {destination}"
                    )
                cached_files += 1
        collections = source.get("collections", {})
        if collections:
            if not isinstance(collections, dict):
                raise TypeError(f"source {source_id!r} collections must be an object")
            for collection, names in collections.items():
                if (
                    not isinstance(names, list)
                    or not names
                    or len(names) != len(set(names))
                ):
                    raise SourceRegistryError(
                        f"invalid source collection {source_id}:{collection}"
                    )
                missing = set(names) - set(source["files"])
                if missing:
                    raise SourceRegistryError(
                        f"source collection {source_id}:{collection} has unknown files {sorted(missing)}"
                    )

    products = registry["products"]
    product_files = 0
    for product_id, product in products.items():
        if not isinstance(product, dict):
            raise TypeError(f"product {product_id!r} must be an object")
        inputs = product.get("inputs")
        if not isinstance(inputs, list) or any(item not in sources for item in inputs):
            raise SourceRegistryError(
                f"product {product_id!r} has unknown source inputs"
            )
        expected_inputs = _check_digest(
            product.get("input_identity_sha256"),
            label=f"{product_id}.input_identity_sha256",
        )
        if _product_input_identity(sources, inputs) != expected_inputs:
            raise SourceRegistryError(f"product source inputs are stale: {product_id}")
        generator = _repository_path(
            product.get("generator"), label=f"{product_id}.generator"
        )
        if not generator.is_file():
            raise FileNotFoundError(generator)
        expected_generator = _check_digest(
            product.get("generator_sha256"), label=f"{product_id}.generator_sha256"
        )
        if _sha256(generator) != expected_generator:
            raise SourceRegistryError(f"generator digest mismatch: {generator}")
        for group in ("canonical_inputs", "outputs"):
            values = product.get(group, {})
            if not isinstance(values, dict):
                raise TypeError(f"product {product_id!r} {group} must be an object")
            for relative, digest in values.items():
                path = _repository_path(relative, label=f"{product_id}.{group}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if _sha256(path) != _check_digest(
                    digest, label=f"{product_id}.{relative}"
                ):
                    raise SourceRegistryError(f"product digest mismatch: {path}")
                product_files += 1

    derived_count = 0
    for relative, spec in registry["derived_manifests"].items():
        path = _repository_path(relative, label="derived manifest path")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.read_text(encoding="utf-8") != render_derived_manifest(registry, spec):
            raise SourceRegistryError(f"derived source manifest is stale: {path}")
        derived_count += 1
    return {
        "sources": len(sources),
        "local_files": local_files,
        "cached_files": cached_files,
        "products": len(products),
        "product_files": product_files,
        "derived_manifests": derived_count,
    }


def _source_url(repository: str, revision: str, upstream_path: str) -> str:
    path = _relative_path(upstream_path, label="upstream path").as_posix()
    repository = repository.removesuffix(".git").rstrip("/")
    if repository.startswith("https://github.com/"):
        slug = repository.removeprefix("https://github.com/")
        return f"https://raw.githubusercontent.com/{slug}/{revision}/{path}"
    if repository.startswith("https://gitlab.com/"):
        return f"{repository}/-/raw/{revision}/{path}"
    raise SourceRegistryError(
        f"automatic source URL construction is unsupported for {repository!r}"
    )


def _fetch(url: str, *, label: str) -> bytes:
    if not url.startswith("https://"):
        raise SourceRegistryError(f"{label} requires an HTTPS URL")
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "vibeqc-source-sync/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.read()
    except (OSError, TimeoutError) as error:
        raise SourceRegistryError(f"failed to fetch {label}: {error}") from error


def _checked_revision(revision: Any) -> str:
    if not isinstance(revision, str) or not revision.strip():
        raise TypeError("upstream revision must be a non-empty string")
    revision = revision.strip()
    if revision.casefold() in {"head", "main", "master", "latest", "stable", "develop"}:
        raise SourceRegistryError(
            "source updates require an explicit immutable commit or release tag"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", revision):
        raise SourceRegistryError("unsafe upstream revision")
    return revision


def sync_source(
    source_id: str,
    registry_path: Path = REGISTRY,
    *,
    cache_root: Path = DEFAULT_CACHE,
) -> list[Path]:
    registry = _load(registry_path)
    source = registry["sources"].get(source_id)
    if source is None:
        raise SourceRegistryError(f"unknown scientific source {source_id!r}")
    _validate_source_metadata(source_id, source)
    written: list[Path] = []
    for name, item in source["files"].items():
        upstream = _fetch(item["url"], label=f"pinned source {source_id}:{name}")
        upstream_expected = item.get("upstream_sha256", item["sha256"])
        if _sha256_bytes(upstream) != upstream_expected:
            raise SourceRegistryError(
                f"upstream digest mismatch for {source_id}:{name}; refusing to write"
            )
        data = _normalize(upstream, item.get("normalization"))
        if _sha256_bytes(data) != item["sha256"]:
            raise SourceRegistryError(
                f"normalized digest mismatch for {source_id}:{name}; refusing to write"
            )
        destination = _source_file_destination(source_id, source, name, cache_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        written.append(destination)
    return written


def update_source(
    source_id: str,
    revision: str,
    registry_path: Path = REGISTRY,
    *,
    cache_root: Path = DEFAULT_CACHE,
) -> list[Path]:
    """Move one existing allowlisted source set to an explicit new revision."""
    revision = _checked_revision(revision)
    registry = _load(registry_path)
    source = registry["sources"].get(source_id)
    if source is None:
        raise SourceRegistryError(f"unknown scientific source {source_id!r}")
    _validate_source_metadata(source_id, source)
    if revision == source["revision"]:
        raise SourceRegistryError(
            f"source {source_id!r} is already pinned to {revision}"
        )

    fetched: list[tuple[str, dict[str, Any], str, bytes, bytes]] = []
    for name, item in source["files"].items():
        upstream_path = item.get("upstream_path", name)
        url = _source_url(source["repository"], revision, upstream_path)
        upstream = _fetch(url, label=f"candidate source {source_id}:{name}@{revision}")
        data = _normalize(upstream, item.get("normalization"))
        fetched.append((name, item, url, upstream, data))

    written: list[Path] = []
    for name, item, url, upstream, data in fetched:
        item["url"] = url
        item["sha256"] = _sha256_bytes(data)
        if item.get("normalization") is not None:
            item["upstream_sha256"] = _sha256_bytes(upstream)
        else:
            item.pop("upstream_sha256", None)
        if "size" in item:
            item["size"] = len(upstream)
        if "git_blob" in item:
            header = f"blob {len(upstream)}\0".encode()
            item["git_blob"] = hashlib.sha1(
                header + upstream, usedforsecurity=False
            ).hexdigest()
        destination = _source_file_destination(source_id, source, name, cache_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        written.append(destination)
    source["revision"] = revision
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="offline integrity/freshness check"
    )
    verify_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    subparsers.add_parser(
        "regenerate", help="rewrite registry-derived compatibility manifests"
    )
    sync_parser = subparsers.add_parser(
        "sync", help="download one already-pinned source"
    )
    sync_parser.add_argument("source")
    sync_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    update_parser = subparsers.add_parser(
        "update", help="move one allowlisted source set to an explicit revision"
    )
    update_parser.add_argument("source")
    update_parser.add_argument("--revision", required=True)
    update_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if args.command == "verify":
        summary = verify(args.registry, cache_root=args.cache_root)
        print(json.dumps(summary, sort_keys=True))
    elif args.command == "regenerate":
        for path in regenerate(args.registry):
            print(path.relative_to(ROOT))
    elif args.command == "sync":
        for path in sync_source(args.source, args.registry, cache_root=args.cache_root):
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    else:
        for path in update_source(
            args.source, args.revision, args.registry, cache_root=args.cache_root
        ):
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
