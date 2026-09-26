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
    for area in (
        "core",
        "runtime",
        "tensor",
        "solver",
        "response",
        "scf",
        "dft",
        "posthf",
        "cc",
    ):
        assert area in report["areas"]
    assert {
        "solver/dense_linear.hpp",
        "solver/diis.hpp",
        "solver/diis_coefficients.hpp",
        "solver/diis_history.hpp",
        "solver/iteration_control.hpp",
    } <= {module["path"] for module in report["modules"] if module["owner"] == "solver"}
    assert {"cc", "scf"} <= set(
        report["infrastructure_inventory"]["diis_history"]["consumer_areas"]
    )
    assert {"cc", "scf"} <= set(
        report["infrastructure_inventory"]["dense_linear"]["consumer_areas"]
    )
    assert {
        "runtime/nvidia_host_api/cublas_v2.h",
        "runtime/nvidia_host_api/cusolverDn.h",
        "runtime/nvidia_host_api/vibeqc_nvidia_host_api.h",
    } <= {module["path"] for module in report["modules"]}
    assert report["posthf_scf_edges"]
    assert all(edge["known_debt"] for edge in report["posthf_scf_edges"])
    for name, item in report["infrastructure_inventory"].items():
        assert item["owner_exists"], name
        assert set(item["required_consumer_areas"]) <= set(item["consumer_areas"]), name
    assert all(
        test["exists"]
        for gate in report["regression_inventory"].values()
        for test in gate["tests"]
    )
    assert {"shared", "hf_only", "dft_only", "cc_only"} <= set(
        report["ownership_groups"]
    )


@pytest.mark.parametrize("owner", ["core", "runtime", "tensor", "solver", "response"])
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
    (source / "scf/fleet.hpp").write_text("// method implementation\n")
    (source / "runtime/internal/context.cpp").write_text(
        '#include "../../scf/fleet.hpp"\n'
    )
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden runtime dependency on scf/fleet.hpp" in errors[0]


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


def test_shared_c_header_cannot_hide_method_dependency(tmp_path: typing.Any) -> None:
    source = tmp_path / "src"
    (source / "runtime").mkdir(parents=True)
    (source / "cc").mkdir()
    (source / "cc/solver.hpp").write_text("// concrete method\n")
    (source / "runtime/host_api.h").write_text('#include "cc/solver.hpp"\n')
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "runtime/host_api.h:1: forbidden runtime dependency" in errors[0]


def test_known_reverse_edge_is_a_ceiling_not_an_exemption(
    tmp_path: typing.Any,
) -> None:
    source = tmp_path / "src"
    (source / "runtime").mkdir(parents=True)
    (source / "scf").mkdir()
    (source / "scf/aot_shell_registry.hpp").write_text("// existing debt\n")
    (source / "scf/fleet.hpp").write_text("// new method owner\n")
    runtime = source / "runtime/cuda_runtime.cu"
    runtime.write_text('#include "scf/aot_shell_registry.hpp"\n')
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []

    runtime.write_text(
        '#include "scf/aot_shell_registry.hpp"\n#include "scf/fleet.hpp"\n'
    )
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden runtime dependency on scf/fleet.hpp" in errors[0]


def test_shared_layers_may_depend_on_other_shared_layers(tmp_path: typing.Any) -> None:
    source = tmp_path / "src"
    (source / "runtime").mkdir(parents=True)
    (source / "tensor").mkdir()
    (source / "runtime/context.hpp").write_text("// shared runtime\n")
    (source / "tensor/runtime.cpp").write_text('#include "runtime/context.hpp"\n')
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []


@pytest.mark.parametrize("method", ["scf", "dft", "posthf", "cc"])
def test_provider_contract_cannot_depend_on_concrete_method(
    tmp_path: typing.Any, method: str
) -> None:
    source = tmp_path / "src"
    (source / "integrals").mkdir(parents=True)
    (source / method).mkdir()
    (source / method / "method.hpp").write_text("// method implementation\n")
    contract = source / "integrals/electron_interaction_source.hpp"
    contract.write_text(f'#include "{method}/method.hpp"\n')
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert (
        "integrals/electron_interaction_source.hpp:1: forbidden provider contract "
        f"dependency on {method}/method.hpp"
    ) in errors[0]


def test_posthf_cannot_add_new_scf_dependency(tmp_path: typing.Any) -> None:
    source = tmp_path / "src"
    (source / "posthf").mkdir(parents=True)
    (source / "scf").mkdir()
    (source / "scf/new_driver.hpp").write_text("// SCF-owned driver\n")
    (source / "posthf/consumer.cpp").write_text('#include "scf/new_driver.hpp"\n')
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert (
        "posthf/consumer.cpp:1: forbidden post-HF dependency on SCF-owned "
        "scf/new_driver.hpp"
    ) in errors[0]


def test_known_posthf_scf_edge_is_a_ceiling_not_an_exemption(
    tmp_path: typing.Any,
) -> None:
    source = tmp_path / "src"
    (source / "posthf").mkdir(parents=True)
    (source / "scf").mkdir()
    (source / "scf/mean_field.hpp").write_text("// existing debt\n")
    (source / "scf/new_driver.hpp").write_text("// new driver\n")
    consumer = source / "posthf/bridge.cpp"
    consumer.write_text('#include "scf/mean_field.hpp"\n')
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []

    consumer.write_text(
        '#include "scf/mean_field.hpp"\n#include "scf/new_driver.hpp"\n'
    )
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden post-HF dependency" in errors[0]


def test_known_cc_cuda_solver_debt_may_shrink_but_not_grow(
    tmp_path: typing.Any,
) -> None:
    cc = tmp_path / "src/cc"
    cc.mkdir(parents=True)
    (cc / "cuda_solver.cu").write_text("void run_diis() {}\n")
    report = audit_electronic_structure_boundaries(tmp_path)
    assert report["errors"] == []
    assert report["duplicate_infrastructure"]["cc_cuda_diis_owner"]["count"] == 1

    (cc / "second.cu").write_text("void run_diis() {}\n")
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert any(
        "duplicate infrastructure debt grew for cc_cuda_diis_owner" in error
        and "cc/second.cu:1" in error
        for error in errors
    )


def test_cc_cuda_solver_debt_cannot_move_to_an_unapproved_owner(
    tmp_path: typing.Any,
) -> None:
    cc = tmp_path / "src/cc"
    cc.mkdir(parents=True)
    (cc / "replacement.cu").write_text("void run_diis() {}\n")
    errors = audit_electronic_structure_boundaries(tmp_path)["errors"]
    assert any(
        "cc_cuda_diis_owner has an unapproved owner at cc/replacement.cu:1" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("snippet", "debt_name"),
    [
        ("void run_bounded_iterations() {}\n", "cc_host_bounded_iteration_owner"),
        ("struct Diis {};\n", "cc_cpu_diis_owner"),
        (
            (
                "bool solve_linear(std::vector<double> matrix) "
                "{ return !matrix.empty(); }\n"
            ),
            "cc_cpu_local_linear_solver",
        ),
    ],
)
def test_retired_cc_host_solver_debt_cannot_return(
    tmp_path: typing.Any, snippet: str, debt_name: str
) -> None:
    cc = tmp_path / "src/cc"
    cc.mkdir(parents=True)
    (cc / "solver.cpp").write_text(snippet)
    report = audit_electronic_structure_boundaries(tmp_path)
    assert report["duplicate_infrastructure"][debt_name]["allowed"] == 0
    assert report["duplicate_infrastructure"][debt_name]["count"] == 1
    assert any(
        f"duplicate infrastructure debt grew for {debt_name}" in error
        and "cc/solver.cpp:1" in error
        for error in report["errors"]
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


def test_duplicate_guard_ignores_calls_strings_and_declarations_without_bodies(
    tmp_path: typing.Any,
) -> None:
    cc = tmp_path / "src/cc"
    cc.mkdir(parents=True)
    (cc / "solver.cpp").write_text(
        'const char* example = "struct Diis {}; void run_diis() {}";\n'
        "template <class Diis> struct Box {};\n"
        "struct Derived : Diis {};\n"
        "struct Diis;\n"
        "void run_diis();\n"
        "void run_bounded_iterations();\n"
        "void predicate() { if (run_diis()) {} }\n"
        "void expression() { auto x = solve_linear() ? Result{} : Result{}; }\n"
        "void caller() { run_diis(); run_bounded_iterations(); }\n"
    )
    report = audit_electronic_structure_boundaries(tmp_path)
    assert report["errors"] == []
    assert all(
        item["count"] == 0 for item in report["duplicate_infrastructure"].values()
    )


def test_infrastructure_inventory_separates_direct_and_transitive_consumers(
    tmp_path: typing.Any,
) -> None:
    source = tmp_path / "src"
    (source / "response").mkdir(parents=True)
    (source / "posthf").mkdir()
    (source / "response/linear_problem.hpp").write_text("// shared owner\n")
    (source / "response/solve.hpp").write_text(
        '#include "response/linear_problem.hpp"\n'
    )
    (source / "posthf/force.cpp").write_text('#include "response/solve.hpp"\n')
    item = audit_electronic_structure_boundaries(tmp_path)["infrastructure_inventory"][
        "linear_response_problem"
    ]
    assert item["direct_consumers"] == ["response/solve.hpp"]
    assert item["transitive_consumers"] == ["posthf/force.cpp"]


def test_literal_comment_delimiters_do_not_hide_real_include(
    tmp_path: typing.Any,
) -> None:
    path = tmp_path / "src/solver/literal.hpp"
    path.parent.mkdir(parents=True)
    path.write_text(
        'const char* begin = "/*";\n#include "scf/types.hpp"\nconst char* end = "*/";\n'
    )
    report = audit_electronic_structure_boundaries(tmp_path)
    assert len(report["errors"]) == 1
    assert "solver/literal.hpp:2: forbidden solver dependency" in report["errors"][0]


def test_literal_comment_delimiters_do_not_hide_cpp_definitions(
    tmp_path: typing.Any,
) -> None:
    path = tmp_path / "src/cc/literal.cpp"
    path.parent.mkdir(parents=True)
    path.write_text(
        'const char* begin = "/*";\nclass Diis {};\ndouble solve_linear() { return 0; }\nconst char* end = "*/";\n'
    )
    report = audit_electronic_structure_boundaries(tmp_path)
    duplicates = report["duplicate_infrastructure"]
    assert duplicates["cc_cpu_diis_owner"]["locations"] == ["cc/literal.cpp:2"]
    assert duplicates["cc_cpu_local_linear_solver"]["locations"] == ["cc/literal.cpp:3"]


def test_raw_string_examples_do_not_create_dependency_edges(
    tmp_path: typing.Any,
) -> None:
    path = tmp_path / "src/solver/example.hpp"
    path.parent.mkdir(parents=True)
    path.write_text(
        'const char* example = R"sample(\n"quoted example"\n#include "scf/types.hpp"\n)sample";\n'
    )
    assert audit_electronic_structure_boundaries(tmp_path)["errors"] == []
