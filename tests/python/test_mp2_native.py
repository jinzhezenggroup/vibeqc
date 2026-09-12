"""Explicit native CPU source/export tier for the internal A1 consumer."""

import ctypes as ct
import os

import numpy as np
import pytest

from tools.vibeqc_mp2 import PreparedMP2Energy
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_NATIVE_TEST") != "1",
    reason="requires explicit native CPU library selection; fixture tests are separate",
)


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_native_source_to_component_energies(name):
    meta, arrays = load_fixture(name)
    s = fixture_snapshot(meta, arrays)
    mo = arrays["conventional_mo"]
    ovov = mo[
        np.ix_(range(s.nocc), range(s.nocc, s.nmo), range(s.nocc), range(s.nocc, s.nmo))
    ]
    g, t = ovov.transpose(0, 2, 1, 3), arrays["conventional_t2"]
    expected = [np.sum(t * g), np.sum(t * (g - g.swapaxes(2, 3)))]
    with (
        NativeSource(**source_arguments(meta)) as source,
        PreparedMP2Energy(s, source, occupied_tile=2, virtual_tile=3) as p,
    ):
        r = p.execute()
        np.testing.assert_allclose(
            [r.opposite_spin, r.same_spin], expected, atol=1e-11, rtol=1e-10
        )
        assert (
            abs(
                r.correlation_energy
                - meta["records"]["conventional"]["correlation_energy"]
            )
            <= 1e-9
        )


def test_native_hf_export_and_scf_failure_remain_visible():
    meta, _ = load_fixture("h2")
    with NativeSource(**source_arguments(meta)) as source:
        s, diagnostics = export_rhf(source)
        with PreparedMP2Energy(s, source) as p:
            r = p.execute()
        assert s.hf_backend == "native-cpu"
        assert diagnostics["canonicalization_backend"] == "cpu-numpy"
        expected = meta["records"]["conventional"]
        assert (
            abs(r.energy - expected["hf_energy"] - expected["correlation_energy"])
            <= 1e-9
        )
        with pytest.raises(RuntimeError, match="HF failed|converge"):
            export_rhf(source, max_iterations=1)


def bounded_reference(source, *, budget=256 << 20, iterations=100):
    lib = source._library
    fn = lib.vibeqc_posthf_reference_v1
    ptr = ct.POINTER(ct.c_double)
    fn.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_uint,
        ct.c_double,
        ct.c_size_t,
        ptr,
        ct.c_size_t,
        ptr,
        ct.c_char_p,
        ct.c_size_t,
    ]
    n = source.nbf
    arrays, scalars = np.full(5 * n * n + n, np.nan), np.full(6, np.nan)
    error = ct.create_string_buffer(2048)
    status = fn(
        source._handle,
        0,
        0,
        iterations,
        1e-11,
        budget,
        arrays.ctypes.data_as(ptr),
        arrays.size,
        scalars.ctypes.data_as(ptr),
        error,
        len(error),
    )
    if status:
        assert np.isnan(arrays).all() and np.isnan(scalars).all()
        raise RuntimeError(error.value.decode())
    return (
        [arrays[k * n * n : (k + 1) * n * n].reshape(n, n) for k in range(5)],
        arrays[5 * n * n :],
        scalars,
    )


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
def test_bounded_reference_uses_native_physical_fock(name):
    meta, a = load_fixture(name)
    with NativeSource(**source_arguments(meta)) as source:
        (s, h, f, c, d), eps, diag = bounded_reference(source)
        np.testing.assert_allclose(s, a["conventional_S"], atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(h, a["conventional_h"], atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(c.T @ s @ c, np.eye(source.nbf), atol=1e-11)
        np.testing.assert_allclose(f @ c, (s @ c) * eps, atol=1e-10)
        np.testing.assert_allclose(c.T @ f @ c, np.diag(eps), atol=1e-10)
        np.testing.assert_allclose(f @ d @ s, s @ d @ f, atol=1e-9)
        assert abs(diag[0] - meta["records"]["conventional"]["hf_energy"]) <= 1e-9
        assert max(diag[2:5]) <= 1e-8
        with pytest.raises(RuntimeError, match="memory budget"):
            bounded_reference(source, budget=int(diag[5]) - 1)
        with pytest.raises(RuntimeError, match="HF failed"):
            bounded_reference(source, iterations=1)
