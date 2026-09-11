"""Resolve CLI shell and schedule selections without executing or compiling candidates."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from ..cuda_schedule import (
    ScheduleKind,
)
from ..shell_spec import FUSED_SHELL_SPEC_BY_NAME, ShellClassSpec


def _resolve_specifications(names: Iterable[str]) -> tuple[ShellClassSpec, ...]:
    """Resolve CLI shell names while preserving the requested order."""

    specifications = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        try:
            specification = FUSED_SHELL_SPEC_BY_NAME[name]
        except KeyError as error:
            choices = ", ".join(FUSED_SHELL_SPEC_BY_NAME)
            raise ValueError(
                f"unknown shell class {name!r}; choose from {choices}"
            ) from error
        specifications.append(specification)
        seen.add(name)
    if not specifications:
        raise ValueError("at least one shell class is required")
    return tuple(specifications)


def _read_shell_class_file(path: Path) -> tuple[str, ...]:
    """Read shell-class names for a batch run.

    Accept one name per line and comma-separated names so a hotspot list stays
    easy to edit.  ``#`` starts a comment.  Parsing in the tuner keeps the
    expanded order visible in the report and avoids shell-wrapper differences.
    """

    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read shell-class file {path}") from error
    names: list[str] = []
    for raw_line in contents.splitlines():
        line = raw_line.split("#", 1)[0]
        names.extend(line.replace(",", " ").split())
    if not names:
        raise ValueError(f"shell-class file {path} does not contain any names")
    return tuple(names)


def _requested_shell_class_names(arguments: argparse.Namespace) -> tuple[str, ...]:
    """Combine repeated CLI names and list files into one ordered batch."""

    raw_names = getattr(arguments, "shell_class", None)
    names = [raw_names] if isinstance(raw_names, str) else list(raw_names or ())
    raw_files = getattr(arguments, "shell_class_file", None)
    files = [raw_files] if isinstance(raw_files, (str, Path)) else list(raw_files or ())
    for path in files:
        names.extend(_read_shell_class_file(Path(path)))
    if not names:
        raise ValueError(
            "batch autotune requires at least one --shell-class or --shell-class-file"
        )
    # Keep the expanded list available to ``main`` for the all-winners exit
    # gate, including names that came from a file.
    arguments.shell_class = names
    return tuple(names)


def _requested_schedule_kinds(
    arguments: argparse.Namespace,
) -> tuple[ScheduleKind, ...]:
    """Return the optional ordered schedule-family filter for a batch run."""

    raw_kinds = getattr(arguments, "schedule_kind", None)
    values = [raw_kinds] if isinstance(raw_kinds, str) else list(raw_kinds or ())
    kinds: list[ScheduleKind] = []
    seen: set[ScheduleKind] = set()
    for value in values:
        kind = ScheduleKind(value)
        if kind not in seen:
            kinds.append(kind)
            seen.add(kind)
    return tuple(kinds)
