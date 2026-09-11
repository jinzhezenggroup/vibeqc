"""Canonical production-manifest location shared by tuning policy and CLI defaults."""

from __future__ import annotations

from pathlib import Path

_PRODUCTION_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "production_shell_classes.json"
)

_PROVENANCE_FIELDS = (
    "runtime_seconds",
    "compile_seconds",
    "source_bytes",
    "object_bytes",
)
