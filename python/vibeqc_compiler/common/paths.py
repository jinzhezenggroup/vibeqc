"""Locate compiler inputs consistently in a checkout and an installed wheel.

Native templates and audited expression provenance remain owned by src/ and
external/. Packaging copies those inputs into wheel assets; there is no second
editable source tree. Reference fixtures and reproduction scripts intentionally
require a checkout and are not installation-time dependencies.
"""

from pathlib import Path

from .provenance import file_hash

PACKAGE = Path(__file__).resolve().parents[1]
LAYOUT_VERSION = 2


def source_root() -> Path:
    """Return this compiler's checkout, rejecting unrelated working directories."""
    root = PACKAGE.parents[1]
    if (
        not (root / "CMakeLists.txt").is_file()
        or (root / "python/vibeqc_compiler").resolve() != PACKAGE
    ):
        raise ValueError("this operation needs a VibeQC source checkout")
    return root


def asset_path(relative: str) -> Path:
    """Find an owned native/provenance input using its repository-relative name."""
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("compiler assets require a relative path without '..'")
    try:
        path = source_root() / name
    except ValueError:
        path = PACKAGE / "assets" / name
    if not path.exists():
        raise FileNotFoundError(f"missing compiler input: {relative}")
    return path


def source_hashes(*families: str, assets: tuple[str, ...] = ()) -> dict[str, str]:
    """Hash actual inputs under stable logical names in both installation modes.

    Layout version 2 deliberately invalidates pre-migration compiler caches.
    No absolute paths, bytecode, timestamps or compatibility shims enter this
    inventory. Source edits still invalidate it, including nested leaf modules.
    """
    paths = {PACKAGE / "__init__.py"}
    for family in families:
        if family not in {"common", "integral", "tensor", "xc", "dft"}:
            raise ValueError(f"unknown compiler subsystem: {family}")
        paths.update((PACKAGE / family).rglob("*.py"))
        paths.update((PACKAGE / family).rglob("*.json"))
    return {
        **{
            "python/vibeqc_compiler/" + path.relative_to(PACKAGE).as_posix(): file_hash(
                path
            )
            for path in sorted(paths)
        },
        **{name: file_hash(asset_path(name)) for name in sorted(assets)},
    }
