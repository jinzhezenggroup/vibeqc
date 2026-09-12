"""Keep method/backend ownership out of reusable CPU SCF reference interfaces."""

import pytest

from tools.check_scf_structure import audit_scf_structure


def test_current_shared_scf_dependencies_are_valid():
    report = audit_scf_structure()
    assert not report["errors"]
    assert report["modules"]


@pytest.mark.parametrize("include", ['"scf/rhf.hpp"', '"../rhf.hpp"', "<scf/rhf.hpp>"])
def test_method_dependency_cannot_hide_behind_include_spelling(tmp_path, include):
    source = tmp_path / "src/scf"
    (source / "reference").mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / "reference/linalg.cpp").write_text(f"#include {include}\n")
    report = audit_scf_structure(tmp_path)
    assert len(report["errors"]) == 1
    assert "forbidden reference dependency on scf/rhf.hpp" in report["errors"][0]


def test_initial_guess_consumes_reference_without_reverse_edge(tmp_path):
    source = tmp_path / "src/scf"
    for directory in ["reference", "initial_guess"]:
        (source / directory).mkdir(parents=True)
    reference = source / "reference/linalg.hpp"
    reference.write_text("// Independent reference declarations\n")
    guess = source / "initial_guess/density.hpp"
    guess.write_text('#include "scf/reference/linalg.hpp"\n')
    assert not audit_scf_structure(tmp_path)["errors"]
    reference.write_text('#include "scf/initial_guess/density.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


def test_documented_forbidden_example_is_not_an_include(tmp_path):
    source = tmp_path / "src/scf"
    (source / "reference").mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / "reference/linalg.cpp").write_text(
        '/* Forbidden example:\n#include "scf/rhf.hpp"\n*/\n'
        '// #include "scf/rhf.hpp"\n#include <vector>\n'
    )
    assert not audit_scf_structure(tmp_path)["errors"]
