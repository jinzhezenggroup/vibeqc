"""Only explicit restart-baseline markers are nullable KS history values."""

import ctypes as ct
import json
import math
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import _native
from generativeqc.ks_diagnostics import KsDiagnostic, read_ks_diagnostic


def read_history(
    changes: list[float],
    *,
    index: int | None = None,
    physical: tuple[str, float] | None = None,
) -> KsDiagnostic:
    def query(*args: typing.Any) -> int:
        summary_pointer, rows, capacity = args[-3:]
        summary = ct.cast(
            summary_pointer, ct.POINTER(_native.KsDiagnosticDescriptor)
        ).contents
        summary.scf_domain_version = 1
        summary.history_count = len(changes)
        summary.fock_builds = len(changes) + 4
        if rows is not None:
            assert capacity == len(changes)
            for iteration, (row, change) in enumerate(zip(rows, changes), 1):
                row.iteration = iteration
                row.energy_change = change
                row.nuclear_energy = 1.0
                row.one_electron_energy = -2.0
                row.hartree_energy = 0.4
                row.xc_energy = -0.3
                row.density_change_max = 1e-12
                row.physical_residual_max = 2e-12
            if physical is not None:
                setattr(rows[-1], physical[0], physical[1])
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_calculation_get_ks_diagnostic=query,
        generativeqc_batch_get_ks_diagnostic=query,
    )
    return read_ks_diagnostic(library, None, index=index)


@pytest.mark.parametrize("index", (None, 0))
def test_later_explicit_stage_baseline_is_null_without_changing_history(
    index: int | None,
) -> None:
    changes = [math.inf] + [1e-13] * 16 + [-1.0, 2e-14]
    diagnostic = read_history(changes, index=index)
    assert len(diagnostic.history) == 19
    assert [row.iteration for row in diagnostic.history] == list(range(1, 20))
    assert diagnostic.history[0].energy_change is None
    assert diagnostic.history[17].energy_change is None
    assert diagnostic.history[18].energy_change == 2e-14
    assert diagnostic.history[16].energy_change == 1e-13
    assert diagnostic.history[17].density_change_max == 1e-12
    assert diagnostic.history[17].physical_residual_max == 2e-12
    json.dumps(diagnostic.to_payload(), allow_nan=False)


@pytest.mark.parametrize("value", (math.inf, -math.inf, math.nan))
def test_unexpected_later_nonfinite_is_not_a_stage_marker(value: float) -> None:
    diagnostic = read_history([math.inf, value])
    actual = diagnostic.history[1].energy_change
    assert actual is not None
    assert math.isnan(actual) if math.isnan(value) else actual == value
    with pytest.raises(ValueError):
        json.dumps(diagnostic.to_payload(), allow_nan=False)


@pytest.mark.parametrize(
    "iteration,value", ((1, -1.0), (1, -2.0), (2, -2.0), (2, 0.0), (2, 3e-14))
)
def test_only_the_reserved_later_marker_becomes_null(
    iteration: int, value: float
) -> None:
    changes = [math.inf] * iteration
    changes[-1] = value
    assert read_history(changes).history[-1].energy_change == value


@pytest.mark.parametrize(
    "field", ("nuclear_energy", "density_change_max", "physical_residual_max")
)
@pytest.mark.parametrize("value", (math.inf, -math.inf, math.nan))
def test_stage_marker_does_not_hide_nonfinite_physical_data(
    field: str, value: float
) -> None:
    diagnostic = read_history([math.inf, -1.0], physical=(field, value))
    assert diagnostic.history[1].energy_change is None
    with pytest.raises(ValueError):
        json.dumps(diagnostic.to_payload(), allow_nan=False)


@pytest.mark.parametrize("value", (-math.inf, math.nan))
def test_initial_unexpected_nonfinite_is_not_the_positive_infinity_baseline(
    value: float,
) -> None:
    diagnostic = read_history([value])
    assert diagnostic.history[0].energy_change is not None
    with pytest.raises(ValueError):
        json.dumps(diagnostic.to_payload(), allow_nan=False)


@pytest.fixture(scope="module")
def recorder_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import shutil
    import subprocess

    root = Path(__file__).resolve().parents[2]
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a host C++ compiler")
    source = (root / "src/dft/rks.cpp").read_text()
    begin = source.index("          const unsigned reported_iteration =")
    end = source.index("\n        });", begin)
    body = source[begin:end]
    directory = tmp_path_factory.mktemp("ks-stage-recorder")
    program = (
        r"""
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
struct Components { double nuclear=1, one_electron=-2, hartree=.4, xc=-.3; };
struct Row {
 unsigned iteration; Components components;
 double energy_change, density_rms, residual_rms;
 std::array<double,2> electrons;
};
struct Result {
 unsigned iterations{}; double energy{},energy_change{},density_rms{},physical_residual_rms{};
};
struct Ks {
 double physical_residual{},density_change{}; Components components;
 std::array<double,2> electrons; std::vector<Row> history;
};
struct Progress { unsigned iteration; double energy=-.9,energy_change,state_rms=1e-12,residual_rms=2e-12; };
struct Evaluation { Components components; double spin_electrons=1; };
struct Incremental { void observe_progress(double) {} };
int main(int argc,char**argv) {
 if(argc!=5) return 2;
 const unsigned local=std::stoul(argv[1]),iteration_offset=std::stoul(argv[2]);
 const double change=std::stod(argv[3]), expected=std::stod(argv[4]);
 Result result; Ks ks; const Progress progress{local,-.9,change}; const Evaluation evaluation;
 const bool strict_full=false; std::optional<Incremental> incremental_state;
"""
        + body
        + r"""
 const double actual=ks.history.back().energy_change;
 const auto same=[](double a,double b){return a==b || (std::isnan(a)&&std::isnan(b));};
 if(!same(actual,expected) || !same(result.energy_change,change) || !same(progress.energy_change,change))
  throw std::runtime_error("history marker or solver/result arithmetic changed");
 if(ks.history.back().iteration!=local+iteration_offset || ks.history.back().density_rms!=1e-12 ||
    ks.history.back().residual_rms!=2e-12 || result.energy!=-.9)
  throw std::runtime_error("unrelated history or physical telemetry changed");
}
"""
    )
    path, executable = directory / "recorder.cpp", directory / "recorder"
    path.write_text(program)
    subprocess.run(
        [cache, compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize(
    "local,offset,change,expected",
    [
        (1, 17, "inf", "-1"),
        (1, 0, "inf", "inf"),
        (2, 17, "inf", "inf"),
        (1, 17, "-inf", "-inf"),
        (1, 17, "nan", "nan"),
        (1, 17, "0.125", "0.125"),
        (2, 17, "0.125", "0.125"),
        (1, 17, "-2", "-2"),
    ],
)
def test_actual_recorder_marks_only_known_positive_infinite_restart(
    recorder_probe: Path,
    local: int,
    offset: int,
    change: str,
    expected: str,
) -> None:
    import subprocess

    result = subprocess.run(
        [str(recorder_probe), str(local), str(offset), change, expected],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
