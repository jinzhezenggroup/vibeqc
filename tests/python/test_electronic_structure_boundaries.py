"""Cross-method architecture ownership regression tests."""

from __future__ import annotations

import typing

import pytest

from tools.check_electronic_structure_boundaries import (
    audit_electronic_structure_boundaries,
)


def test_current_cross_method_boundaries_are_valid() -> None:
    report = audit_electronic_structure_boundaries()
    assert report["errors"] == []
    assert report["modules"]
    for area in ("core", "runtime", "tensor", "response", "scf", "dft", "posthf", "cc"):
        assert area in report["areas"]


@pytest.mark.parametrize("owner", ["core", "runtime", "tensor", "response"])
@pytest.mark.parametrize("method", ["scf", "dft", "posthf", "cc"])
def test_shared_layer_cannot_depend_on_concrete_method(
    tmp_path: typing.Any, owner: str, method: str
) -> None:
    source = tmp_path / "src"
    (source / owner).mkdir(parents=True)
    (source / method).mkdir(parents=True)
    (source / method / "method.hpp").write_text("// method implementation\n")
    (source / owner / "shared.cpp").write_text(f'#include "{method}/method.hpp"\n')
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert f"forbidden {owner} dependency on {method}/method.hpp" in errors[0]


def test_relative_include_cannot_hide_method_dependency(tmp_path: typing.Any) -> None:
    source = tmp_path / "src"
    (source / "runtime/internal").mkdir(parents=True)
    (source / "scf").mkdir()
    (source / "scf/rhf.hpp").write_text("// method implementation\n")
    (source / "runtime/internal/context.cpp").write_text(
        '#include "../../scf/rhf.hpp"\n'
    )
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden runtime dependency on scf/rhf.hpp" in errors[0]


def test_comments_do_not_create_dependency_edges(tmp_path: typing.Any) -> None:
    source = tmp_path / "src"
    (source / "tensor").mkdir(parents=True)
    (source / "cc").mkdir()
    (source / "cc/solver.hpp").write_text("// method implementation\n")
    (source / "tensor/runtime.hpp").write_text(
        '/* forbidden example:\n#include "cc/solver.hpp"\n*/\n'
        '// #include "cc/solver.hpp"\n'
        "#include <vector>\n"
    )
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []


def test_known_reverse_edge_is_a_ceiling_not_an_exemption(
    tmp_path: typing.Any,
) -> None:
    source = tmp_path / "src"
    (source / "runtime").mkdir(parents=True)
    (source / "scf").mkdir()
    (source / "scf/aot_shell_registry.hpp").write_text("// existing debt\n")
    (source / "scf/rhf.hpp").write_text("// new method owner\n")
    runtime = source / "runtime/cuda_runtime.cu"
    runtime.write_text('#include "scf/aot_shell_registry.hpp"\n')
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []

    runtime.write_text(
        '#include "scf/aot_shell_registry.hpp"\n#include "scf/rhf.hpp"\n'
    )
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden runtime dependency on scf/rhf.hpp" in errors[0]


def test_shared_layers_may_depend_on_other_shared_layers(tmp_path: typing.Any) -> None:
    source = tmp_path / "src"
    (source / "runtime").mkdir(parents=True)
    (source / "tensor").mkdir()
    (source / "runtime/context.hpp").write_text("// shared runtime\n")
    (source / "tensor/runtime.cpp").write_text('#include "runtime/context.hpp"\n')
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []


@pytest.mark.parametrize(
    ("snippet", "debt_name"),
    [
        ("struct Diis {};\n", "cc_cpu_diis_owner"),
        (
            (
                "bool solve_linear(std::vector<double> matrix) "
                "{ return !matrix.empty(); }\n"
            ),
            "cc_cpu_local_linear_solver",
        ),
        ("void run_diis() {}\n", "cc_cuda_diis_owner"),
    ],
)
def test_known_cc_solver_debt_may_shrink_but_not_grow(
    tmp_path: typing.Any, snippet: str, debt_name: str
) -> None:
    cc = tmp_path / "src/cc"
    cc.mkdir(parents=True)
    (cc / "solver.cpp").write_text(snippet)
    report = audit_electronic_structure_boundaries(tmp_path)
    assert report["errors"] == []
    assert report["duplicate_infrastructure"][debt_name]["count"] == 1

    (cc / "second.cpp").write_text(snippet)
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert any(
        f"duplicate infrastructure debt grew for {debt_name}" in error
        for error in errors
    )


def test_metrics_separate_generated_lines(tmp_path: typing.Any) -> None:
    runtime = tmp_path / "src/runtime"
    runtime.mkdir(parents=True)
    (runtime / "context.cpp").write_text("line1\nline2\n")
    (runtime / "policy_generated.hpp").write_text("generated\n")
    metrics = audit_electronic_structure_boundaries(tmp_path)["areas"]["runtime"]
    assert metrics["files"] == 2
    assert metrics["lines"] == 3
    assert metrics["generated_lines"] == 1


@pytest.mark.parametrize(
    "include", ["tensor/../cc/solver.hpp", "tensor/./../cc/solver.hpp"]
)
def test_noncanonical_source_root_include_cannot_hide_dependency(
    tmp_path: typing.Any, include: str
) -> None:
    source = tmp_path / "src"
    (source / "tensor").mkdir(parents=True)
    (source / "cc").mkdir()
    (source / "cc/solver.hpp").write_text("// concrete method\n")
    (source / "tensor/helper.cpp").write_text(f'#include "{include}"\n')
    assert any(
        "forbidden tensor dependency on cc/solver.hpp" in error
        for error in audit_electronic_structure_boundaries(tmp_path)["errors"]
    )


def test_comments_separate_tokens_in_duplicate_guard(tmp_path: typing.Any) -> None:
    cc = tmp_path / "src/cc"
    cc.mkdir(parents=True)
    (cc / "first.cpp").write_text("struct/**/Diis {};\n")
    (cc / "second.cpp").write_text("struct/* ownership */Diis {};\n")
    report = audit_electronic_structure_boundaries(tmp_path)
    assert report["duplicate_infrastructure"]["cc_cpu_diis_owner"]["count"] == 2
    assert any(
        "duplicate infrastructure debt grew" in error for error in report["errors"]
    )
