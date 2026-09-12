"""Identical-C native adapters, including direct CUDA transforms and equations."""

import ctypes as ct
import os

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments
from tools.vibeqc_posthf.sources import NativeSource


def check_native(
    source, arrays, hf_energy, mo, os_ref, ss_ref, backend, tiles=(1, 2, 4, 8)
):
    lib = source._library
    ptr = ct.POINTER(ct.c_double)
    sizeptr = ct.POINTER(ct.c_size_t)
    n = source.nbf
    packed = np.ascontiguousarray(
        np.concatenate([arrays[k].ravel() for k in ("S", "h", "F", "C", "D", "eps")])
    )
    fn = lib.vibeqc_posthf_mp2_energy_v1
    fn.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ptr,
        ct.c_size_t,
        ct.c_double,
        ct.c_size_t,
        ct.c_double,
        ct.c_uint,
        ptr,
        ct.c_char_p,
        ct.c_size_t,
    ]
    error = ct.create_string_buffer(2048)
    for tile in tiles:
        output = np.full(5, np.nan)
        status = fn(
            source._handle,
            backend,
            0,
            packed.ctypes.data_as(ptr),
            packed.size,
            hf_energy,
            256 << 20,
            1e-10,
            tile,
            output.ctypes.data_as(ptr),
            error,
            len(error),
        )
        assert status == 0, error.value.decode()
        np.testing.assert_allclose(output[:2], [os_ref, ss_ref], atol=1e-11, rtol=1e-10)
    block = lib.vibeqc_posthf_mo_block_v1
    block.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ptr,
        ct.c_size_t,
        ct.c_double,
        sizeptr,
        sizeptr,
        ct.c_size_t,
        ptr,
        ct.c_size_t,
        ct.c_char_p,
        ct.c_size_t,
    ]
    # Asymmetric ordered MO selections exercise all four axes, pair exchange
    # and within-pair permutations independently of total-energy agreement.
    base = ((n - 1, 0), (1,), (0, n - 1), (1, 0))
    for order in ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (0, 3, 2, 1)):
        slots = tuple(base[k] for k in order)
        shape = (ct.c_size_t * 4)(*map(len, slots))
        indices = (ct.c_size_t * sum(map(len, slots)))(
            *(i for slot in slots for i in slot)
        )
        result = np.full(tuple(map(len, slots)), np.nan)
        status = block(
            source._handle,
            backend,
            0,
            packed.ctypes.data_as(ptr),
            packed.size,
            hf_energy,
            shape,
            indices,
            256 << 20,
            result.ctypes.data_as(ptr),
            result.size,
            error,
            len(error),
        )
        assert status == 0, error.value.decode()
        np.testing.assert_allclose(result, mo[np.ix_(*slots)], atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
@pytest.mark.parametrize("backend", [0, 1])
def test_same_orbitals_native_components_and_permutations(name, backend):
    if backend and os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1":
        pytest.skip("requires allocated CUDA device")
    meta, a = load_fixture(name)
    arrays = {k: a["conventional_" + k] for k in ("S", "h", "F", "C", "eps")}
    no = meta["records"]["conventional"]["electron_count"] // 2
    arrays["D"] = 2 * arrays["C"][:, :no] @ arrays["C"][:, :no].T
    n = len(arrays["eps"])
    g = a["conventional_mo"][
        np.ix_(range(no), range(no, n), range(no), range(no, n))
    ].transpose(0, 2, 1, 3)
    t = a["conventional_t2"]
    os_ref, ss_ref = np.sum(t * g), np.sum(t * (g - g.swapaxes(2, 3)))
    with NativeSource(**source_arguments(meta)) as source:
        check_native(
            source,
            arrays,
            meta["records"]["conventional"]["hf_energy"],
            a["conventional_mo"],
            os_ref,
            ss_ref,
            backend,
        )
