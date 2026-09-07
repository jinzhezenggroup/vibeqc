"""Fast all-class source gates and fail-closed evidence/cache boundaries."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_codegen.cuda_adapter import CudaCompileResult
from tools.vibeqc_validation import f_shell
from tools.vibeqc_validation.f_shell_cuda import emit_numerical_driver
from tools.vibeqc_validation.f_shell_numerics import contract_reference, eri_orbit

EXPECTED_CLASSES = {
    "fsss",
    "fsps",
    "fspp",
    "fsds",
    "fsdp",
    "fsdd",
    "fsfs",
    "fpss",
    "fpps",
    "fppp",
    "fpds",
    "fpdp",
    "fpdd",
    "fpfs",
    "fpfp",
    "fdss",
    "fdps",
    "fdpp",
    "fdds",
    "fddp",
    "fddd",
    "fdfs",
    "fdfp",
    "fdfd",
    "ffss",
    "ffps",
    "ffpp",
    "ffds",
    "ffdp",
    "ffdd",
    "fffs",
    "fffp",
    "fffd",
    "ffff",
}


@pytest.mark.parametrize("name", f_shell.F_SHELL_CLASSES)
def test_all_34_classes_emit_complete_deterministic_first_derivative_sources(name):
    audit, source = f_shell.source_audit(name)
    repeated, repeated_source = f_shell.source_audit(name)
    assert audit == repeated
    assert source == repeated_source
    assert audit["deterministic_generations"] == 2
    assert audit["independent_derivative_centers"] == [0, 1, 2]
    assert audit["recovered_derivative_centers"] == [3]
    assert len(audit["symbols"]) == 8
    driver = emit_numerical_driver(name)
    assert driver == emit_numerical_driver(name)
    assert "@" not in driver
    assert "component_gradient" not in driver
    for symbol in audit["symbols"]:
        assert driver.count(symbol) == 2  # Declaration plus host kernel table.


def test_catalog_separates_manifest_selection_from_unmeasured_acceptance():
    assert set(f_shell.F_SHELL_CLASSES) == EXPECTED_CLASSES
    report = f_shell.catalog()
    assert len(report["rows"]) == 34
    assert len({row["source"]["registry_class_index"] for row in report["rows"]}) == 34
    for row in report["rows"]:
        assert row["source"]["status"] == "pass"
        assert row["consumers"] == ["fock", "force"]
        for stage in ("compilation", "resources", "numerical", "endpoint", "promotion"):
            assert row[stage]["status"] == "not-run"
    fpps = next(row for row in report["rows"] if row["shell_class"] == "fpps")
    assert fpps["manifest"]["force"]
    assert fpps["manifest"]["acceptance"] == "provisional"
    assert set(f_shell.SMOKE_CLASSES) < EXPECTED_CLASSES
    assert len(f_shell.SMOKE_CLASSES) == 5
    for names in ([], ["ssss"], ["fsss", "fsss"]):
        with pytest.raises(ValueError):
            f_shell.catalog(names=names)


def test_compile_cache_checks_toolchain_objects_and_resource_records(
    tmp_path, monkeypatch
):
    calls = []
    version = ["test CUDA 12.9"]
    monkeypatch.setattr(f_shell, "_tool_version", lambda tool: version[0])

    def compile_fake(self, source, output):
        calls.append(source)
        output.write_bytes(b"test-only object")
        log = "\n".join(
            f"ptxas info    : Function properties for {symbol}\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info    : Used 32 registers, used 0 barriers, 128 bytes smem\n"
            for symbol in f_shell.generated_symbols("fsss")
        )
        return CudaCompileResult(0, False, 0.25, "", log)

    monkeypatch.setattr(f_shell.CudaCompilerAdapter, "compile", compile_fake)

    def inspect_fake(command, **kwargs):
        if "--extract-elf" in command:
            (Path(kwargs["cwd"]) / "kernel.sm_120.cubin").write_bytes(b"test cubin")
        resources = "\n".join(
            f" Function {symbol}:\n  REG:32 STACK:0 SHARED:128 LOCAL:0"
            for symbol in f_shell.generated_symbols("fsss")
        )
        return subprocess.CompletedProcess(command, 0, resources, "")

    monkeypatch.setattr(f_shell.subprocess, "run", inspect_fake)
    kwargs = {"nvcc": tmp_path / "nvcc", "cache": tmp_path / "cache", "jobs": 1}
    initial = f_shell.catalog(names=["fsss"])
    first = f_shell.compile_matrix(initial, **kwargs)
    assert len(calls) == 1
    assert first["rows"][0]["resources"]["status"] == "pass"
    assert first["rows"][0]["resources"]["spill_free"]
    assert not first["rows"][0]["compilation"]["cache_hit"]
    second = f_shell.compile_matrix(initial, **kwargs)
    assert len(calls) == 1
    assert second["rows"][0]["compilation"]["cache_hit"]
    key = second["rows"][0]["compilation"]["cache_key"]
    (kwargs["cache"] / key / "kernel.o").write_bytes(b"corrupt")
    f_shell.compile_matrix(initial, **kwargs)
    assert len(calls) == 2
    version[0] = "different compiler"
    changed = f_shell.compile_matrix(initial, **kwargs)
    assert len(calls) == 3
    assert changed["rows"][0]["compilation"]["cache_key"] != key
    assert initial["rows"][0]["compilation"]["status"] == "not-run"


def test_missing_resource_rows_fail_even_when_compilation_returns_success(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(f_shell, "_tool_version", lambda tool: "test only")

    def compile_fake(self, source, output):
        output.write_bytes(b"test object")
        return CudaCompileResult(0, False, 0.01, "", "no resource records")

    monkeypatch.setattr(f_shell.CudaCompilerAdapter, "compile", compile_fake)
    monkeypatch.setattr(
        f_shell.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "", ""),
    )
    report = f_shell.compile_matrix(
        f_shell.catalog(names=["fsss"]),
        nvcc=tmp_path / "nvcc",
        cache=tmp_path / "cache",
    )
    assert report["rows"][0]["compilation"]["status"] == "pass"
    assert report["rows"][0]["resources"]["status"] == "fail"


def test_loop_contractions_match_full_tensor_jk_and_energy_derivatives():
    rng = np.random.default_rng(135)
    eri = np.array([[[[0.7]]]])
    derivative = rng.normal(size=(4, 3, 1, 1, 1, 1))
    derivative[-1] = -derivative[:-1].sum(axis=0)
    matrices = rng.normal(size=(2, 4, 4))
    spin = (matrices + matrices.transpose(0, 2, 1)) / 2
    density = spin.sum(axis=0)
    result = contract_reference(
        eri, derivative, density, spin, (0, 1, 2, 3), (0, 1, 2, 3)
    )
    full = np.zeros((4, 4, 4, 4))
    assert len(eri_orbit((0, 1, 2, 3))) == 8
    assert len(eri_orbit((0, 0, 0, 0))) == 1
    for indices in eri_orbit((0, 1, 2, 3)):
        full[indices] = 0.7
    J = np.einsum("abcd,cd->ab", full, density)
    K = np.einsum("abcd,bd->ac", full, density)
    np.testing.assert_allclose(result["rhf_fock"], J - 0.5 * K, atol=1e-14)
    for index in range(2):
        np.testing.assert_allclose(
            result["uhf_fock"][index],
            J - np.einsum("abcd,bd->ac", full, spin[index]),
            atol=1e-14,
        )
    energy = 0.5 * np.sum(density * result["rhf_fock"])
    np.testing.assert_allclose(
        result["rhf_force"], -energy / 0.7 * derivative.reshape(4, 3), atol=1e-14
    )
    np.testing.assert_allclose(result["rhf_force"].sum(axis=0), 0, atol=1e-14)


def test_numerical_driver_checks_fixture_sizes_and_uses_no_gpu_visibility_override():
    source = emit_numerical_driver("ffff")
    assert "invalid bounded fixture dimensions" in source
    assert "active_blocks_per_sm" in source
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source
    assert "cudaSetDevice" not in source
    assert "setenv" not in source
    assert "unrestricted" in source and "persistent_args" in source


def test_source_cli_help_and_small_report(tmp_path):
    import sys

    script = Path(__file__).resolve().parents[2] / "tools/validate_f_shells.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--smoke" in help_result.stdout and "--slurm-time" in help_result.stdout
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tier",
            "source",
            "--shell-class",
            "fsss",
            "--output",
            str(output),
        ],
        check=True,
        cwd=tmp_path,
    )
    report = json.loads(output.read_text())
    assert report["rows"][0]["source"]["status"] == "pass"
    assert report["rows"][0]["numerical"]["status"] == "not-run"


def test_generated_ff_value_pair_terms_include_all_wick_matchings(tmp_path):
    """Execute the emitted coefficient routine on CPU against Gaussian moments.

    This directly guards the f/f omission: six equal axes at zero shift have
    15 full pairings. It also checks every derivative subset, mixed axes, and
    asymmetric shifts without NVCC or an installed reference chemistry package.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    _, source = f_shell.source_audit("ffss")
    start = source.index("struct GeneratedFfssValueTerm")
    stop = source.index(
        "/** Evaluate one AO component without constructing force-only derivatives. */",
        start,
    )
    routine = source[start:stop]
    axes_cases = ((0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 2, 2), (0, 1, 0, 2, 1, 0))
    shifts = np.array([0.17, -0.31, 0.23, 0.41, -0.11, 0.07])
    variance = 0.63

    def moment(remaining, axes):
        # Condition on the first factor: its mean leaves the smaller moment;
        # pairing it with every equal-axis factor contributes the covariance.
        if not remaining:
            return 1.0
        first, *rest = remaining
        result = shifts[first] * moment(rest, axes)
        for index, other in enumerate(rest):
            if axes[first] == axes[other]:
                result += variance * moment(rest[:index] + rest[index + 1 :], axes)
        return result

    expected = [
        variance ** subset.bit_count()
        * moment([i for i in range(6) if not subset & (1 << i)], axes)
        for axes in axes_cases
        for subset in range(64)
    ]
    table = ",".join("{" + ",".join(map(str, axes)) + "}" for axes in axes_cases)
    program = "\n".join(
        (
            "#include <iostream>",
            "#include <iomanip>",
            "#define __device__",
            "#define __forceinline__ inline",
            "#define __popc __builtin_popcount",
            routine,
            "int main() {",
            f"const unsigned axes[][6] = {{{table}}};",
            "const double shifts[6] = {0.17,-0.31,0.23,0.41,-0.11,0.07};",
            "std::cout << std::setprecision(17);",
            "for (const auto& one : axes) for(unsigned subset=0; subset<64; ++subset)",
            "std::cout << generated_ffss_pair_value_term<6>(one, shifts, 0.63, subset).coefficient << '\\n';",
            "}",
        )
    )
    path, executable = tmp_path / "moment.cpp", tmp_path / "moment"
    path.write_text(program)
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    run = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
    np.testing.assert_allclose(
        np.fromstring(run.stdout, sep="\n"), expected, rtol=2e-14, atol=2e-14
    )


def test_measured_device_time_preserves_mixed_class_uncertainty(tmp_path):
    import sqlite3

    path = Path(__file__).resolve().parents[2] / "benchmarks/f_shell_device_time.py"
    spec = importlib.util.spec_from_file_location("f_shell_device_time_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [
        "generated_sm120_fpps_shell_class_force_rhf_kernel",
        "void two_electron_force_quartet_persistent_kernel<(bool)0, (unsigned int)9>()",
        "void two_electron_force_quartet_persistent_kernel<false, 5u>()",
        "void build_fock_direct_quartet_persistent_kernel<false, 12u, double>()",
        "void build_fock_direct_quartet_persistent_kernel<false, 8u, double>()",
        "void build_fock_direct_psss_persistent_kernel<false>()",
    ]
    trace = tmp_path / "trace.sqlite"
    with sqlite3.connect(trace) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER)"
        )
        db.executemany("INSERT INTO StringIds VALUES (?,?)", enumerate(names))
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,100,?)",
            ((i,) for i in range(6)),
        )
    result = module.read_trace(trace)
    assert result["consumers"]["force"]["f_fraction_lower"] == pytest.approx(2 / 3)
    assert result["consumers"]["force"]["f_fraction_upper"] == 1
    assert result["consumers"]["fock"]["f_fraction_lower"] == pytest.approx(1 / 3)
    assert result["consumers"]["fock"]["f_fraction_upper"] == pytest.approx(2 / 3)
    assert result["exact_class_nanoseconds"] == {"fpps/force": 100}
    assert result["all_kernel_nanoseconds"] == 600
    with pytest.raises(ValueError, match="no fock"):
        module.summarize([])


def test_ffff_force_wick_coefficients_do_not_overflow_before_division(tmp_path):
    """Check emitted integer arithmetic through the highest first-force order."""
    import math

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    _, source = f_shell.source_audit("ffff")
    start = source.index(
        "__device__ __forceinline__ unsigned generated_ffff_wick_multiplicity("
    )
    stop = source.index(
        "__device__ __forceinline__ double generated_ffff_coulomb(", start
    )
    expected = [
        math.factorial(order)
        // (2**pairs * math.factorial(pairs) * math.factorial(order - 2 * pairs))
        for order in range(14)
        for pairs in range(order // 2 + 1)
    ]
    path, executable = tmp_path / "wick.cpp", tmp_path / "wick"
    path.write_text(
        "\n".join(
            (
                "#include <iostream>",
                "#include <cstdint>",
                "#define __device__",
                "#define __forceinline__ inline",
                source[start:stop],
                "int main() { for(unsigned n=0;n<=13;++n) for(unsigned p=0;p<=n/2;++p)",
                "std::cout << generated_ffff_wick_multiplicity(n,p) << '\\n'; }",
            )
        )
    )
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    run = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
    assert [int(value) for value in run.stdout.split()] == expected
