"""CPU-safe derivative work admission and diagnostic environment isolation."""

import os
import shutil
import subprocess
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.df_admission_probe import CONTROLS, controls, errors


@pytest.fixture(scope="module")
def policy(tmp_path_factory: typing.Any) -> typing.Any:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("df-derivative-policy")
    source = directory / "query.cpp"
    source.write_text(r"""
#include <iostream>
#include "scf/df_derivative_policy.hpp"
int main() {
  std::size_t n, a, op, ap, rank;
  unsigned arch, varied;
  while (std::cin >> n >> a >> op >> ap >> rank >> varied >> arch)
    std::cout << vibeqc::scf::df_shell_execution_preferred(n,a,arch) << " "
              << vibeqc::scf::df_signature_packets_preferred(op,ap,varied,arch) << " "
              << vibeqc::scf::df_packed_response_preferred(n,a,rank,arch) << "\n";
}
""")
    executable = directory / "query"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )

    def query(rows: typing.Any) -> typing.Any:
        result = subprocess.run(
            [str(executable)],
            input="".join(" ".join(map(str, r)) + "\n" for r in rows),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]

    return query


def test_work_model_boundaries_and_unknown_architecture(
    policy: typing.Any,
) -> None:
    rows = [
        (n, a, op, ap, rank, v, arch)
        for n, a in [
            (0, 96),
            (29, 29),
            (58, 58),
            (64, 63),
            (64, 64),
            (87, 87),
            (96, 464),
            (192, 928),
            (384, 385),
            (513, 767),
        ]
        for op, ap in [
            (0, 1),
            (64, 64),
            (128, 255),
            (128, 256),
            (256, 1000),
            (2**64 - 1, 2**64 - 1),
        ]
        for rank in (0, max(1, n // 5), max(1, n // 4), n // 4 + 1)
        for v in (0, 1)
        for arch in (80, 89, 100, 120, 121)
    ]
    for (n, a, op, ap, rank, v, arch), selected in zip(rows, policy(rows), strict=True):
        assert selected == (
            int(arch == 120 and n * n * a >= 2**18),
            int(arch == 120 and v and op * op * ap >= 2**22),
            int(
                arch == 120
                and rank > 0
                and n >= 4
                and rank <= n // 4
                and n * n * a >= 2**28
            ),
        )


def test_packed_response_profile_generalizes_beyond_benchmark_tuples(
    policy: typing.Any,
) -> None:
    """The profile keeps small defaults while accepting nearby heavy workloads."""
    rows = [
        # n, naux, orbital primitives, auxiliary primitives, rank, varied, arch
        (384, 384, 128, 256, 80, 1, 120),
        (640, 640, 128, 256, 128, 1, 120),
        (646, 646, 128, 256, 129, 1, 120),
        (700, 600, 128, 256, 140, 1, 120),
        (768, 768, 128, 256, 160, 1, 120),
        (769, 900, 128, 256, 191, 1, 120),
        (769, 900, 128, 256, 193, 1, 120),
        (768, 768, 128, 256, 160, 1, 121),
    ]
    packed = [result[2] for result in policy(rows)]
    assert packed == [0, 0, 1, 1, 1, 1, 0, 0]


def test_controls_restore_absent_and_present_variables_on_failure(
    monkeypatch: typing.Any,
) -> None:
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "generic")
    monkeypatch.delenv("VIBEQC_DF_PRIMITIVE_BUCKETS", raising=False)
    monkeypatch.setenv("VIBEQC_DF_TRACE", "caller-trace")
    before = {
        k: os.environ.get(k)
        for k in [*("VIBEQC_DF_" + n for n in CONTROLS), "VIBEQC_DF_TRACE"]
    }
    with pytest.raises(RuntimeError, match="probe failure"), controls("packet"):
        assert os.environ["VIBEQC_DF_PRIMITIVE_BUCKETS"] == "packet"
        assert "VIBEQC_DF_TRACE" not in os.environ
        with controls("auto"):
            assert all("VIBEQC_DF_" + n not in os.environ for n in CONTROLS)
        assert os.environ["VIBEQC_DF_PRIMITIVE_BUCKETS"] == "packet"
        raise RuntimeError("probe failure")
    assert {k: os.environ.get(k) for k in before} == before


@pytest.mark.parametrize("invalid", ["shape", "energy_nan", "force_inf", "oracle_nan"])
def test_probe_rejects_invalid_numerical_evidence(invalid: typing.Any) -> None:
    result = SimpleNamespace(
        energies=np.array([1.0]), items=[SimpleNamespace(forces=np.ones((2, 3)))]
    )
    energy, force = np.array([1.0]), np.ones((1, 2, 3))
    if invalid == "shape":
        force = np.ones((1, 1, 3))
    elif invalid == "energy_nan":
        result.energies[:] = np.nan
    elif invalid == "force_inf":
        result.items[0].forces[:] = np.inf
    else:
        energy[:] = np.nan
    with pytest.raises(RuntimeError):
        errors(result, energy, force)


def test_architecture_query_only_reads_required_attributes(
    tmp_path: typing.Any,
) -> None:
    """Mock the runtime to test failure propagation and forbid full queries."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "cuda_runtime.h").write_text(r"""
#pragma once
using cudaError_t = int;
constexpr int cudaSuccess=0;
enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor };
int calls=0, fail=0;
inline int cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  ++calls;
  if(device!=7 || calls==fail) return 42;
  *value=attr==cudaDevAttrComputeCapabilityMajor ? 12 : 0;
  return 0;
}
""")
    source = tmp_path / "check.cpp"
    source.write_text(
        r"""
#include <cassert>
#include "runtime/cuda_architecture.hpp"
int main() {
  unsigned architecture=999;
  assert(vibeqc::runtime::cuda_architecture(7,architecture)==0);
  assert(architecture==120 && calls==2);
  for (int rejected : {1,2}) {
    calls=0; fail=rejected; architecture=999;
    assert(vibeqc::runtime::cuda_architecture(7,architecture)==42);
    assert(architecture==0 && calls==rejected);
  }
}
""".replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>")
    )
    executable = tmp_path / "check"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(tmp_path),
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
