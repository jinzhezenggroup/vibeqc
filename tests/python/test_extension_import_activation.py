"""Probe each lazy extension in an independent, bounded interpreter."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_import_probe(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_extension_package_root_keeps_public_surfaces_lazy() -> None:
    _run_import_probe(
        """
        import sys
        import vibeqc.extensions as extensions
        assert extensions.API_VERSION == 1
        assert extensions.__all__ == ["API_VERSION", "method", "tensor", "xc"]
        assert {"method", "tensor", "xc"}.issubset(dir(extensions))
        assert all("vibeqc.extensions." + name not in sys.modules
                   for name in ("method", "tensor", "xc"))
        assert not hasattr(extensions, "unknown_extension")
        assert all("vibeqc.extensions." + name not in sys.modules
                   for name in ("method", "tensor", "xc"))
        """
    )


@pytest.mark.parametrize("surface", ("method", "tensor", "xc"))
def test_selecting_one_extension_surface_does_not_activate_siblings(surface: str) -> None:
    _run_import_probe(
        f"""
        import importlib
        import sys
        from vibeqc.extensions import {surface} as selected
        import vibeqc.extensions as extensions
        assert selected.__name__ == "vibeqc.extensions.{surface}"
        assert getattr(extensions, "{surface}") is selected
        assert importlib.import_module("vibeqc.extensions.{surface}") is selected
        assert all("vibeqc.extensions." + name not in sys.modules
                   for name in {{"method", "tensor", "xc"}} - {{"{surface}"}})
        """
    )


def test_method_composition_reuses_canonical_xc_helpers() -> None:
    _run_import_probe(
        """
        from fractions import Fraction
        from collections.abc import Iterable, Mapping
        from typing import get_type_hints
        from vibeqc.extensions import method
        assert get_type_hints(method.compose, localns={"Iterable": Iterable, "Mapping": Mapping})["exact_exchange"] == (
            int | str | Fraction | None
        )
        result = method.compose("custom-hf", exact_exchange="1")
        from vibeqc_compiler.method import MethodSpec, resolve_method
        expected = resolve_method(MethodSpec(identifier="custom-hf", semilocal_components=(), exact_exchange=Fraction(1)))
        assert result.identity == expected.identity
        assert result.to_payload() == expected.to_payload()
        try:
            method.compose("bad", exact_exchange=0.5)
        except TypeError:
            pass
        else:
            raise AssertionError("inexact coefficient was accepted")
        """
    )
