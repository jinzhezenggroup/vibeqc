"""Internal native DF solver against dense and determinant-space endpoints."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools import generate_df_ccsd_core as core
from tools import generate_df_ccsd_hoisted as hoisted
from tools import generate_df_ccsd_native as actions
from tools import generate_rccsd_native as conventional
from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "foo",
    "fov",
    "fvv",
    "ovov",
    "ovvo",
    "oovv",
    "ovvv",
    "ovoo",
    "oooo",
    "vvvv",
    "d1",
    "d2",
    "t1",
    "t2",
    "bov",
    "bvv",
)
COLUMNS = (
    "status",
    "energy",
    "iterations",
    "r1",
    "r2",
    "capacity",
    "device_bytes",
    "h2d",
    "scalar_d2h",
    "amplitude_d2h",
    "iterations_called",
    "replays_called",
    "q_calls",
    "q_operations",
    "accumulations",
    "seconds",
    "hoisted_evaluations",
    "preparation_calls",
    "contraction_terms",
)


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def solver_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, bool]:
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("requires explicit Slurm real-device qualification")
    cache = shutil.which("ccache")
    cxx = shutil.which("c++")
    nvcc = shutil.which("nvcc")
    if not cache or not cxx or (cuda and not nvcc):
        pytest.skip("requires ccache and selected compilers")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("df-solver-" + request.param)
    for name, producer in (
        ("generated_rccsd_cpu.hpp", conventional.cpu_header),
        ("generated_rccsd_cuda.cu", conventional.cuda_source),
        ("generated_df_ccsd_cpu.hpp", actions.cpu_header),
        ("generated_df_ccsd_cuda.cuh", actions.cuda_header),
        ("generated_df_ccsd_cuda.cu", actions.cuda_source),
        ("generated_df_ccsd_core_cpu.hpp", core.cpu_header),
        ("generated_df_ccsd_core_cuda.cuh", core.cuda_header),
        ("generated_df_ccsd_core_cuda.cu", core.cuda_source),
        ("generated_df_ccsd_hoisted_cpu.hpp", hoisted.cpu_header),
        ("generated_df_ccsd_hoisted_cuda.cuh", hoisted.cuda_header),
        ("generated_df_ccsd_hoisted_cuda.cu", hoisted.cuda_source),
    ):
        (directory / name).write_text(producer())
    sources = [ROOT / "src/cc/solver.cpp", ROOT / "tests/native/df_cc_solver_probe.cpp"]
    if cuda:
        sources += [
            ROOT / "src/cc/cuda_solver.cu",
            *[
                directory / name
                for name in (
                    "generated_rccsd_cuda.cu",
                    "generated_df_ccsd_cuda.cu",
                    "generated_df_ccsd_core_cuda.cu",
                    "generated_df_ccsd_hoisted_cuda.cu",
                )
            ],
        ]
    objects = []
    for source in sources:
        device = source.suffix == ".cu"
        compiler = nvcc if device else cxx
        assert compiler is not None
        obj = directory / (source.name + ".o")
        command = [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            f"-DGENERATIVEQC_HAS_CUDA={int(cuda)}",
            "-I" + str(directory),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ]
        if device:
            command += ["-arch=sm_120"]
        subprocess.run(
            [*command, "-c", str(source), "-o", str(obj)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        objects.append(str(obj))
    executable = directory / "solver-probe"
    compiler = nvcc if cuda else cxx
    assert compiler is not None
    subprocess.run(
        [compiler, *objects, "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return executable, cuda


def _case(
    o: int = 2, v: int = 3, q: int = 4
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(1759 + o + v)
    factors = rng.normal(scale=0.025, size=(q, o + v, o + v))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    g = np.einsum("Qpq,Qrs->pqrs", factors, factors)
    eps = np.r_[np.linspace(-1.3, -0.7, o), np.linspace(0.4, 1.1, v)]
    fock = np.diag(eps)
    fock[:o, o:] = rng.normal(scale=0.002, size=(o, v))
    fock[o:, :o] = fock[:o, o:].T
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    arrays = dense_feeds(
        fock, g, fock[:o, o:] / d1, g[:o, o:, :o, o:].transpose(0, 2, 1, 3) / d2
    )
    arrays.update(d1=d1, d2=d2, bov=factors[:, :o, o:], bvv=factors[:, o:, o:])
    return fock, g, arrays


def _stream(
    arrays: dict[str, np.ndarray],
    cuda: bool,
    *,
    df: bool = True,
    budget: int = 1 << 30,
    diis: int = 6,
    hoist: bool = True,
) -> bytes:
    o, v = arrays["t1"].shape
    q = len(arrays["bov"]) if df else 0
    header = np.array(
        [o, v, q, budget, 100, diis, int(cuda) | (0 if hoist else 4)], dtype=np.uint64
    )
    omitted = ("ovvv", "vvvv") if df else ("bov", "bvv")
    return header.tobytes() + b"".join(
        np.asarray(arrays[name], dtype=np.float64).tobytes()
        for name in FIELDS
        if name not in omitted
    )


def _run(
    probe: tuple[Path, bool], arrays: dict[str, np.ndarray], **kwargs: object
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    executable, cuda = probe
    process = subprocess.run(
        [str(executable)],
        input=_stream(arrays, cuda, **kwargs),
        capture_output=True,
        check=True,
        timeout=120,
    )
    lines = process.stdout.decode().splitlines()
    status = {
        key: float(value) if key in ("energy", "r1", "r2", "seconds") else int(value)
        for key, value in zip(COLUMNS, lines[0].split(), strict=True)
    }
    values = np.fromstring(lines[1], sep=" ")
    o, v = arrays["t1"].shape
    return status, values[: o * v].reshape(o, v), values[o * v :].reshape(o, o, v, v)


@pytest.mark.parametrize("o,v,q,diis", [(1, 1, 1, 0), (2, 3, 4, 6), (3, 2, 3, 6)])
def test_solver_matches_dense_and_independent_determinants(
    solver_probe: tuple[Path, bool], o: int, v: int, q: int, diis: int
) -> None:
    fock, g, arrays = _case(o, v, q)
    actual, t1, t2 = _run(solver_probe, arrays, diis=diis)
    dense, dense_t1, dense_t2 = _run(solver_probe, arrays, df=False, diis=diis)
    assert actual["status"] == dense["status"] == 0
    np.testing.assert_allclose(actual["energy"], dense["energy"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(t1, dense_t1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(t2, dense_t2, atol=2e-11, rtol=0)
    energy, r1, r2 = DeterminantOracle(fock, g, o).evaluate_full(t1, t2)
    np.testing.assert_allclose(actual["energy"], energy, atol=2e-12, rtol=0)
    assert (
        max(np.max(np.abs(r1)), np.max(np.abs(r2)), actual["r1"], actual["r2"]) <= 1e-10
    )
    calls = actual["iterations_called"] + actual["replays_called"]
    assert actual["q_calls"] == q * calls
    assert (
        actual["accumulations"] == 2 * q * calls + 4 * q * actual["hoisted_evaluations"]
    )
    assert actual["preparation_calls"] == actual["hoisted_evaluations"]
    assert actual["hoisted_evaluations"] <= actual["iterations_called"]
    assert actual["q_operations"] > actual["q_calls"]
    assert dense["q_calls"] == dense["q_operations"] == dense["accumulations"] == 0
    if solver_probe[1]:
        expected = sum(
            arrays[name].nbytes for name in FIELDS if name not in ("ovvv", "vvvv")
        )
        assert actual["h2d"] == expected
        assert actual["amplitude_d2h"] == t1.nbytes + t2.nbytes


def test_exact_memory_admission_and_symmetric_factor_gate(
    solver_probe: tuple[Path, bool],
) -> None:
    _, _, arrays = _case()
    actual, _, _ = _run(solver_probe, arrays)
    fallback, _, _ = _run(solver_probe, arrays, hoist=False)
    conventional_admission = bytearray(_stream(arrays, solver_probe[1]))
    conventional_admission[6 * 8 : 7 * 8] = np.uint64(2).tobytes()
    process = subprocess.run(
        [str(solver_probe[0])],
        input=conventional_admission,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode and b"factorized execution owner" in process.stderr
    _run(solver_probe, arrays, budget=int(actual["capacity"]))
    bounded, _, _ = _run(solver_probe, arrays, budget=int(fallback["capacity"]))
    assert actual["capacity"] > fallback["capacity"]
    assert actual["hoisted_evaluations"] > 0 and bounded["hoisted_evaluations"] == 0
    process = subprocess.run(
        [str(solver_probe[0])],
        input=_stream(
            arrays,
            solver_probe[1],
            budget=int(min(actual["capacity"], fallback["capacity"])) - 1,
        ),
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode and b"budget" in process.stderr
    arrays["bvv"][0, 0, 1] += 0.01
    process = subprocess.run(
        [str(solver_probe[0])],
        input=_stream(arrays, solver_probe[1]),
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode and b"symmetric" in process.stderr


@pytest.mark.parametrize("o,v,q", [(2, 3, 4), (4, 1, 1), (2, 6, 1)])
def test_hoisted_solver_matches_forced_bounded_fallback(
    solver_probe: tuple[Path, bool], o: int, v: int, q: int
) -> None:
    """Charge complete solves, including the unchanged expanded replay.

    Low-Q and occupied-rich cases protect scheduling outside the large
    virtual-rich workload, without interpreting work counts as wall time.
    """
    _, _, arrays = _case(o, v, q)
    fast, t1, t2 = _run(solver_probe, arrays)
    old, old_t1, old_t2 = _run(solver_probe, arrays, hoist=False)
    assert fast["status"] == old["status"] == 0
    assert fast["replays_called"] == old["replays_called"] == 1
    assert fast["hoisted_evaluations"] == fast["iterations_called"] > 0
    assert old["hoisted_evaluations"] == old["preparation_calls"] == 0
    assert 0 < fast["contraction_terms"] < old["contraction_terms"]
    np.testing.assert_allclose(fast["energy"], old["energy"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(t1, old_t1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(t2, old_t2, atol=2e-11, rtol=0)


@pytest.mark.parametrize("field", ["t1", "oooo"])
def test_prepare_and_core_overflow_remain_sticky(
    solver_probe: tuple[Path, bool], field: str
) -> None:
    """Finite inputs overflow in prepare/core, with ordinary later Q slices."""
    _, _, arrays = _case()
    arrays[field].fill(1e200 if field == "t1" else 1e308)
    if field == "oooo":
        arrays["t2"].fill(8.0)
    status, _, _ = _run(solver_probe, arrays)
    assert status["status"] == 2


def test_early_auxiliary_overflow_survives_later_slices(
    solver_probe: tuple[Path, bool],
) -> None:
    _, _, arrays = _case()
    arrays["bvv"][0] *= 1e200
    # Later Q slices remain well behaved. A per-slice reset must not erase the
    # first error before convergence checks (including optimized DIIS paths).
    status, _, _ = _run(solver_probe, arrays)
    assert status["status"] == 2


@pytest.mark.parametrize("name", ["h2", "h2o", "ch4"])
def test_molecular_exact_factorization_recovers_pinned_ccsd(
    solver_probe: tuple[Path, bool], name: str
) -> None:
    """Exact positive-pair factorization is a solver oracle, not a DF-fit claim."""
    from test_df_ccsd_factorized import _factor_eri

    from tools.cc_endpoint_fixtures import load

    meta, source = load(name)
    o = int(np.count_nonzero(source["occ"]))
    eps = source["eps"]
    fock = source["C"].T @ source["F"] @ source["C"]
    factors = _factor_eri(source["g"])
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    arrays = dense_feeds(
        fock,
        source["g"],
        fock[:o, o:] / d1,
        source["g"][:o, o:, :o, o:].transpose(0, 2, 1, 3) / d2,
    )
    arrays.update(d1=d1, d2=d2, bov=factors[:, :o, o:], bvv=factors[:, o:, o:])
    status, t1, t2 = _run(solver_probe, arrays)
    assert status["status"] == 0
    np.testing.assert_allclose(
        status["energy"], meta["correlation_energy"], atol=3e-10, rtol=0
    )
    np.testing.assert_allclose(t1, source["t1"], atol=2e-9, rtol=0)
    np.testing.assert_allclose(t2, source["t2"], atol=2e-9, rtol=0)
