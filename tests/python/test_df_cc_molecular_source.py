"""Real-CUDA molecular DF source against committed independent AO/metric data."""

from __future__ import annotations

import ctypes as ct
import json
import os
import shutil
import subprocess
import time
import typing
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_CC_SOURCE_CUDA_TEST") != "1",
    reason="requires finite Slurm CUDA allocation and built native library",
)


@pytest.fixture(scope="module")
def source_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    from generativeqc import _native

    directory = tmp_path_factory.mktemp("df-source-native")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    obj = directory / "source.o"
    output = directory / "source-probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(library.parent / "generated"),
            "-c",
            str(ROOT / "tests/native/df_cc_source_probe.cpp"),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            str(obj),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    _native.load_library()
    dll = ct.CDLL(str(output))
    call: typing.Any = dll.df_cc_source_probe
    call.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
        ct.POINTER(ct.c_double),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    molecular = dll.df_cc_molecular_probe
    molecular.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.c_void_p,
        ct.c_size_t,
    ]
    molecular.restype = ct.c_int
    call.molecular = molecular
    return call


def auxiliary_g_fixture(representation: str) -> tuple[dict, dict]:
    """Independent signed f-orbital/g-auxiliary blocks and a nontrivial frame.

    This source test supplies an orthogonal frame, not a converged HF state.
    Complete molecular tests below separately cover native RHF composition.
    """
    pytest.importorskip("pyscf")
    from tools.generativeqc_validation.df_gradient import reference_df_matrices

    common = {
        "atomic_numbers": [2, 1, 1],
        "coordinates": [[0.1, -0.2, -0.7], [0.3, 0.1, 0.8], [-0.5, 0.4, 0.2]],
        "charge": 0,
        "multiplicity": 1,
        "basis_representation": representation,
    }

    def shell(atom: int, angular: int) -> dict:
        return {
            "atom_index": atom,
            "angular_momentum": angular,
            "primitives": [[0.6 + 0.2 * angular, 0.8], [1.7 + 0.1 * angular, -0.1]],
        }

    orbital = {**common, "shells": [shell(0, 0), shell(0, 3), shell(1, 0), shell(1, 1)]}
    auxiliary = {
        **common,
        "shells": [shell(0, 0), shell(0, 2), shell(1, 4), shell(2, 3)],
    }
    raw, metric, _, _ = reference_df_matrices(orbital, auxiliary)
    c, _ = np.linalg.qr(np.random.default_rng(1777).normal(size=(len(raw), len(raw))))
    metadata = {
        "inputs": {
            **orbital,
            "basis_representation": "real_spherical"
            if representation == "spherical"
            else "cartesian",
        },
        "auxiliary_shells": auxiliary["shells"],
        "records": {"conventional": {"electron_count": 4}},
    }
    return metadata, {"metric": metric, "raw_three_center": raw, "conventional_C": c}


@pytest.mark.parametrize(
    "name", ["h2", "water", "lih", "f_heh", "g_cartesian", "g_spherical"]
)
@pytest.mark.parametrize("duplicate_auxiliary", [False, True])
def test_native_molecular_factors_and_blocks(
    source_probe: typing.Any, name: str, duplicate_auxiliary: bool, tmp_path: Path
) -> None:
    from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
    from tools.generativeqc_posthf.sources import NativeSource

    meta, arrays = (
        auxiliary_g_fixture(name.removeprefix("g_"))
        if name.startswith("g_")
        else load_fixture(name)
    )
    args = source_arguments(meta)
    auxiliary_indices = list(range(len(arrays["metric"])))
    if duplicate_auxiliary:
        # Duplicate an existing s shell. Independent saved integrals extend by
        # repeated rows/columns, giving Q != N and a rank-deficient metric
        # without calling another integral implementation in this test.
        assert args["auxiliary_basis"][0].angular_momentum == 0
        args["auxiliary_basis"] += (args["auxiliary_basis"][0],)
        auxiliary_indices.append(0)
    c = np.ascontiguousarray(arrays["conventional_C"])
    o = meta["records"]["conventional"]["electron_count"] // 2
    n = len(c)
    v = n - o
    with NativeSource(**args) as source:
        q = source.naux
        shapes = [
            (q, o, v),
            (q, v, v),
            (o, v, o, v),
            (o, v, v, o),
            (o, o, v, v),
            (o, v, o, o),
            (o, o, o, o),
        ]
        elements = sum(int(np.prod(shape)) for shape in shapes)
        output = np.full(elements, np.nan)
        counts = np.zeros(12, dtype=np.uintp)
        values = np.zeros(7)
        error = ct.create_string_buffer(1024)

        def run(budget: int) -> int:
            return source_probe(
                source._handle,
                c.ctypes.data_as(ct.POINTER(ct.c_double)),
                o,
                budget,
                output.ctypes.data_as(ct.POINTER(ct.c_double)),
                elements,
                counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
                values.ctypes.data_as(ct.POINTER(ct.c_double)),
                error,
                len(error),
            )

        assert run(1) != 0
        assert np.isnan(output).all(), "failure published partial factors"
        assert run(1 << 30) == 0, error.value.decode()
        capacity = int(counts[0])
        published = output.copy()
        assert run(capacity - 1) != 0
        np.testing.assert_array_equal(output, published)
        assert run(capacity) == 0, error.value.decode()
        # Reference integrals are committed PySCF values, never a production
        # source or a native CPU-oracle retry. Reconstruct the same metric root.
        metric = arrays["metric"][np.ix_(auxiliary_indices, auxiliary_indices)]
        eig, u = np.linalg.eigh(metric)
        root = (
            u
            * np.where(
                eig > eig.max() * 1e-10,
                1 / np.sqrt(np.maximum(eig, np.finfo(float).tiny)),
                0,
            )
        ) @ u.T
        b = np.einsum(
            "mp,nq,mnP,PQ->pqQ",
            c,
            c,
            arrays["raw_three_center"][:, :, auxiliary_indices],
            root,
            optimize=True,
        )
        # The physical owner applies one pair projection before selecting any
        # sector. The independent reference must construct that same fitted
        # Hamiltonian; per-block or B_vv-only repair would be inconsistent.
        b = (b + b.transpose(1, 0, 2)) / 2
        eri = np.einsum("pqQ,rsQ->pqrs", b, b, optimize=True)
        expected = [
            b[:o, o:, :].transpose(2, 0, 1),
            b[o:, o:, :].transpose(2, 0, 1),
            eri[:o, o:, :o, o:],
            eri[:o, o:, o:, :o],
            eri[:o, :o, o:, o:],
            eri[:o, o:, :o, :o],
            eri[:o, :o, :o, :o],
        ]
        offset = 0
        errors = []
        for shape, want in zip(shapes, expected, strict=True):
            size = int(np.prod(shape))
            actual = output[offset : offset + size].reshape(shape)
            offset += size
            np.testing.assert_allclose(actual, want, atol=3e-10, rtol=3e-10)
            if shape == (q, v, v):
                np.testing.assert_array_equal(actual, actual.transpose(0, 2, 1))
            errors.append(float(np.max(np.abs(actual - want))))
        assert (
            counts[2] == n
            and counts[3] == n * n * q
            and counts[4] == n + 2
            and counts[5] == 5
        )
        assert counts[6] == 2 * n**3 * q + n * n * q * q
        assert counts[7] == q * sum(int(np.prod(shape)) for shape in shapes[2:])
        assert (
            counts[8] == n * n * 8
            and counts[9] == elements * 8
            and counts[10] == 2 * q * q * 8
        )
        assert counts[0] <= 1 << 30 and counts[11] == np.count_nonzero(
            eig > eig.max() * 1e-10
        )
        assert np.isfinite(values).all() and values[6] > 0
        (tmp_path / "source.json").write_text(
            json.dumps(
                {
                    "case": name,
                    "duplicate_auxiliary": duplicate_auxiliary,
                    "n": n,
                    "o": o,
                    "q": q,
                    "counts": counts.tolist(),
                    "values": values.tolist(),
                    "maximum_block_errors": errors,
                },
                indent=2,
            )
            + "\n"
        )


@pytest.mark.parametrize("duplicate_auxiliary", [False, True])
@pytest.mark.parametrize("auxiliary_g", [False, True])
def test_complete_native_h2_df_ccsd_against_determinant_energy(
    source_probe: typing.Any,
    duplicate_auxiliary: bool,
    auxiliary_g: bool,
    tmp_path: Path,
) -> None:
    from tools.generativeqc_cc.oracle import DeterminantOracle
    from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
    from tools.generativeqc_posthf.sources import NativeSource

    meta, arrays = load_fixture("h2")
    args = source_arguments(meta)
    if auxiliary_g:
        pytest.importorskip("pyscf")
        from generativeqc import Primitive, Shell

        from tools.generativeqc_validation.df_gradient import reference_df_matrices

        extra = {"atom_index": 0, "angular_momentum": 4, "primitives": [[0.7, 1.0]]}
        args["auxiliary_basis"] += (Shell(0, 4, (Primitive(0.7, 1.0),)),)
        orbital = {**meta["inputs"], "basis_representation": args["representation"]}
        auxiliary = {**orbital, "shells": [*meta["auxiliary_shells"], extra]}
        raw, metric, _, _ = reference_df_matrices(orbital, auxiliary)
        arrays = {**arrays, "metric": metric, "raw_three_center": raw}
    indices = list(range(len(arrays["metric"])))
    if duplicate_auxiliary:
        args["auxiliary_basis"] += (args["auxiliary_basis"][0],)
        indices.append(0)
    metric = arrays["metric"][np.ix_(indices, indices)]
    eig, u = np.linalg.eigh(metric)
    scale = np.zeros_like(eig)
    keep = eig > eig.max() * 1e-10
    scale[keep] = 1 / np.sqrt(eig[keep])
    root = (u * scale) @ u.T
    c = arrays["conventional_C"]
    b = np.einsum(
        "mp,nq,mnP,PQ->pqQ",
        c,
        c,
        arrays["raw_three_center"][:, :, indices],
        root,
        optimize=True,
    )
    eri = np.einsum("pqQ,rsQ->pqrs", b, b)
    fock = c.T @ arrays["conventional_F"] @ c
    # For two electrons, CCSD spans the full determinant space. This reference
    # is independent of both native CC equations and its convergence replay.
    expected = np.linalg.eigvalsh(DeterminantOracle(fock, eri, 1).hnormal)[0]
    values = np.full(20, np.nan)
    error = ct.create_string_buffer(1024)
    with NativeSource(**args) as source:
        started = time.perf_counter()
        status = source_probe.molecular(
            source._handle,
            1 << 30,
            values.ctypes.data_as(ct.POINTER(ct.c_double)),
            error,
            len(error),
        )
        elapsed = time.perf_counter() - started
    assert status == 0, error.value.decode()
    assert np.isfinite(values).all()
    assert abs(values[0] - meta["records"]["conventional"]["hf_energy"]) < 1e-9
    assert abs(values[1] - expected) < 3e-9
    assert abs(values[2] - values[0] - values[1]) < 2e-14
    assert max(values[4:6]) <= 1e-10 and values[6] <= 1 << 30
    assert values[10] == len(c) ** 2 * len(indices) and values[11] == len(indices)
    assert values[12] > 0 and values[13] >= values[3] and values[14] == 1
    assert values[15] == len(indices) * (values[13] + values[14])
    assert values[16] > 0 and values[17] == len(c) and min(values[18:]) > 0
    (tmp_path / "molecular.json").write_text(
        json.dumps(
            {
                "case": "h2",
                "auxiliary_g": auxiliary_g,
                "duplicate_auxiliary": duplicate_auxiliary,
                "values": values.tolist(),
                "expected_correlation_energy": float(expected),
                "absolute_energy_error": float(abs(values[1] - expected)),
                # Complete internal native method call; input basis normalization
                # and this test's independent oracle construction are excluded.
                "native_method_call_seconds": elapsed,
            },
            indent=2,
        )
        + "\n"
    )
