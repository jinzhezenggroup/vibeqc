"""Different generated point programs must coexist in one linked executable."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc.bulk_aot import SourceVariant
from vibeqc_compiler.xc.bulk_point_program import SemilocalPointBinding

ROOT = Path(__file__).resolve().parents[2]


def binding(name: str, first: int, second: int) -> SemilocalPointBinding:
    # Distinct synthetic science makes accidental linker symbol coalescing visible.
    variant = SourceVariant(
        name=name,
        family="LDA",
        spin="polarized",
        import_identity=f"import-{name}",
        domain="synthetic-linkage-test",
        features=("rho_a", "rho_b"),
        derivative_order=1,
        backend="cpu",
        source=(
            "#include <math.h>\n"
            "void bulk_xc_point(const double* x, double* y) {\n"
            f"  y[0] = {first}.0*x[0] + {second}.0*x[1];\n"
            f"  y[1] = {first}.0; y[2] = {second}.0;\n"
            "}\n"
        ),
        energy_nodes=1,
        ssa={},
    )
    return SemilocalPointBinding(variant, f"capability-{name}", f"expression-{name}", 1)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_two_translation_units_keep_distinct_point_programs(
    tmp_path: Path, optimization: str
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    # Compile the emitted adapters against the production point ABI declarations.
    # No native library or unrelated runtime implementation is required to link.
    header = (ROOT / "src/dft/xc.hpp").read_text(encoding="utf-8")
    start = header.index("struct SemilocalPointValue {")
    stop = header.index("void validate_semilocal_point_program", start)
    include = tmp_path / "dft"
    include.mkdir()
    (include / "xc.hpp").write_text(
        "#pragma once\nnamespace vibeqc::dft {\n" + header[start:stop] + "}\n",
        encoding="utf-8",
    )
    paths = []
    for name, first, second in (("first", 2, 3), ("second", 5, 7)):
        point = binding(name, first, second)
        path = tmp_path / f"{name}.cpp"
        path.write_text(
            point.emit_source()
            + '\nextern "C" const vibeqc::dft::SemilocalPointProgram* '
            + f"{name}_program() {{ return &vibeqc::dft::bulk_generated::kPointProgram; }}\n",
            encoding="utf-8",
        )
        paths.append(path)
    driver = tmp_path / "main.cpp"
    driver.write_text(
        r'''
#include "dft/xc.hpp"
#include <cstring>
extern "C" const vibeqc::dft::SemilocalPointProgram* first_program();
extern "C" const vibeqc::dft::SemilocalPointProgram* second_program();
int main() {
  const auto* first = first_program();
  const auto* second = second_program();
  if (first == second || first->evaluate == second->evaluate) return 1;
  if (std::strcmp(first->identifier, "first") || std::strcmp(second->identifier, "second")) return 2;
  if (std::strcmp(first->expression_identity, "expression-first") ||
      std::strcmp(second->expression_identity, "expression-second")) return 3;
  const double rho[2]{1, 2}, gradient[2][3]{}, tau[2]{};
  const auto a = first->evaluate(rho, gradient, tau);
  const auto b = second->evaluate(rho, gradient, tau);
  if (a.energy != 8 || a.rho[0] != 2 || a.rho[1] != 3) return 4;
  if (b.energy != 19 || b.rho[0] != 5 || b.rho[1] != 7) return 5;
  return 0;
}
''',
        encoding="utf-8",
    )
    executable = tmp_path / "linked"
    compiled = subprocess.run(
        [compiler, "-std=c++20", optimization, "-I", str(tmp_path),
         *(str(path) for path in paths), str(driver), "-o", str(executable)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


def test_linkage_revision_invalidates_binding_but_not_scalar_artifact() -> None:
    point = binding("first", 2, 3)
    source = point.variant.source
    identity = point.variant.emission_identity
    old_payload = point.to_payload()
    old_payload["schema"] = "vibeqc.libxc-bulk-point-program-binding/v1"
    assert point.identity != canonical_hash(old_payload)
    assert point.variant.source == source
    assert point.variant.emission_identity == identity
    assert source in point.emit_source()
