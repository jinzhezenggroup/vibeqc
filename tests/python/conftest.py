"""Pytest-only helpers for keeping expensive independent references live but reusable."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

_RANGE_REFERENCE_NODE = (
    "tests/python/test_range_separation.py::"
    "test_generated_range_values_and_all_center_derivatives_against_libcint"
)
_RANGE_INTOR_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def _molecule_integral_identity(mol) -> bytes:
    """Hash every PySCF array that determines the requested integral."""
    digest = hashlib.sha256()
    digest.update(mol._atm.tobytes())
    digest.update(mol._bas.tobytes())
    digest.update(mol._env.tobytes())
    digest.update(b"cart=1" if mol.cart else b"cart=0")
    return digest.digest()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Memoize only repeated libcint oracle blocks in the range-separation gate."""
    if not item.nodeid.startswith(_RANGE_REFERENCE_NODE):
        yield
        return

    mole = pytest.importorskip("pyscf.gto.mole")
    mole_type = mole.Mole
    original = mole_type.intor_by_shell

    def cached_intor_by_shell(self, intor, shls, *args, **kwargs):
        # The targeted gates use the simple allocating form. Preserve exact
        # PySCF semantics for any call with extra options or an output buffer.
        if args or kwargs:
            return original(self, intor, shls, *args, **kwargs)
        key = (
            intor,
            tuple(shls),
            _molecule_integral_identity(self),
        )
        cached = _RANGE_INTOR_CACHE.get(key)
        if cached is None:
            cached = np.asarray(original(self, intor, shls)).copy()
            cached.setflags(write=False)
            _RANGE_INTOR_CACHE[key] = cached
        return cached.copy()

    mole_type.intor_by_shell = cached_intor_by_shell
    try:
        yield
    finally:
        mole_type.intor_by_shell = original
