#!/usr/bin/env python3
"""Generate the build-identity header from the repository-owned source manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path


def _bool_int(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "on", "true", "yes"}:
        return 1
    if normalized in {"0", "off", "false", "no", ""}:
        return 0
    raise ValueError(f"unsupported boolean value: {value}")


def _inventory(source_root: Path, manifest_path: Path) -> list[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(
            f"unsupported VibeQC source identity manifest schema: "
            f"{data.get('schema_version')}"
        )

    root = source_root.resolve()
    manifest = manifest_path.resolve()
    inputs: set[Path] = {manifest.relative_to(root)}

    for group in data.get("recursive_groups", []):
        relative_root = Path(group["root"])
        if relative_root.is_absolute() or ".." in relative_root.parts:
            raise ValueError(f"unsafe source identity root: {relative_root}")
        group_root = root / relative_root
        if not group_root.is_dir():
            raise ValueError(f"source identity root is not a directory: {relative_root}")
        patterns = group.get("patterns", [])
        if not patterns:
            raise ValueError(f"source identity root has no patterns: {relative_root}")
        for path in group_root.rglob("*"):
            if not path.is_file():
                continue
            relative_to_group = path.relative_to(group_root).as_posix()
            if any(fnmatch.fnmatch(relative_to_group, pattern) for pattern in patterns):
                inputs.add(path.resolve().relative_to(root))

    for relative_text in data.get("files", []):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source identity file: {relative}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"source identity file is missing: {relative}")
        inputs.add(relative)

    return sorted(inputs, key=lambda path: path.as_posix())


def _source_identity(source_root: Path, inputs: list[Path]) -> str:
    digest_input = []
    for relative in inputs:
        payload = (source_root / relative).read_bytes()
        digest_input.append(
            f"{relative.as_posix()}:{hashlib.sha256(payload).hexdigest()}\n"
        )
    return hashlib.sha256("".join(digest_input).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-fast-compile", required=True)
    parser.add_argument("--release-build", required=True)
    args = parser.parse_args()

    identity = _source_identity(
        args.source_root.resolve(),
        _inventory(args.source_root.resolve(), args.manifest.resolve()),
    )
    rendered = args.template.read_text(encoding="utf-8")
    rendered = rendered.replace("@VIBEQC_SOURCE_IDENTITY@", identity)
    rendered = rendered.replace(
        "#cmakedefine01 VIBEQC_CUDA_FAST_COMPILE",
        f"#define VIBEQC_CUDA_FAST_COMPILE {_bool_int(args.cuda_fast_compile)}",
    )
    rendered = rendered.replace(
        "#cmakedefine01 VIBEQC_TUNING_RELEASE_BUILD",
        f"#define VIBEQC_TUNING_RELEASE_BUILD {_bool_int(args.release_build)}",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding="utf-8") == rendered:
        return
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
