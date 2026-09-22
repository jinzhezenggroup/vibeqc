"""The borrowed reference accessor must reject malformed public descriptors."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_virtual_orbital_access_is_checked(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = tmp_path / "bounds.cpp"
    source.write_text(r"""#include "core/electronic_reference.hpp"
#include <iostream>
int main() {
  using vibeqc::core::ElectronicReferenceView;
  ElectronicReferenceView view;
  view.basis_functions = 3; view.spin_channels = 2;
  view.channels[0].occupied = 2; view.channels[1].occupied = 0;
  if (view.virtual_orbitals(0) != 1 || view.virtual_orbitals(1) != 3) return 1;
  view.channels[0].occupied = 4;
  try { (void)view.virtual_orbitals(0); return 2; }
  catch (const std::invalid_argument&) {}
  view.channels[0].occupied = 2;
  view.spin_channels = 3;
  try { (void)view.virtual_orbitals(0); return 3; }
  catch (const std::invalid_argument&) {}
  try { (void)view.virtual_orbitals(2); return 4; }
  catch (const std::out_of_range&) {}
  view.spin_channels = 2;
  try { (void)view.virtual_orbitals(2); return 5; }
  catch (const std::out_of_range&) {}
  view.basis_functions = 0;
  try { (void)view.virtual_orbitals(1); return 6; }
  catch (const std::invalid_argument&) {}
  std::cout << "reference accessor bounds passed\n";
}
""")
    executable = tmp_path / "bounds"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, (
        result.stdout + result.stderr + str(result.returncode)
    )
