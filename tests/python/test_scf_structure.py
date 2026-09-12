"""Keep method/backend ownership out of reusable CPU SCF reference interfaces."""

import pytest

from tools.check_scf_structure import audit_scf_structure


def test_current_shared_scf_dependencies_are_valid():
    report = audit_scf_structure()
    assert not report["errors"]
    assert report["modules"]


@pytest.mark.parametrize("include", ['"scf/rhf.hpp"', '"../rhf.hpp"', "<scf/rhf.hpp>"])
@pytest.mark.parametrize("owner", ["reference", "solver", "gradient"])
def test_method_dependency_cannot_hide_behind_include_spelling(
    tmp_path, include, owner
):
    source = tmp_path / "src/scf"
    (source / owner).mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / owner / "implementation.cpp").write_text(f"#include {include}\n")
    report = audit_scf_structure(tmp_path)
    assert len(report["errors"]) == 1
    assert f"forbidden {owner} dependency on scf/rhf.hpp" in report["errors"][0]


def test_gradient_assembly_cannot_depend_on_solver_state(tmp_path):
    source = tmp_path / "src/scf"
    (source / "solver").mkdir(parents=True)
    (source / "gradient").mkdir()
    (source / "solver/diis.hpp").write_text("// Trajectory state\n")
    (source / "gradient/hf_gradient.cpp").write_text('#include "scf/solver/diis.hpp"\n')
    report = audit_scf_structure(tmp_path)
    assert len(report["errors"]) == 1
    assert "forbidden gradient dependency on scf/solver/diis.hpp" in report["errors"][0]


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


@pytest.mark.parametrize(
    "name, owner",
    [
        ("arena.cpp", "cuda_planning"),
        ("eigensolver.cpp", "cuda_eigensolver"),
        ("df_source_setup.cpp", "cuda_df_source"),
        ("df_plan_setup.cpp", "cuda_df_runtime"),
        ("df_rhf_scf.cpp", "cuda_df_runtime"),
        ("resources.cpp", "cuda_resources"),
        ("matrix_library.cpp", "cuda_matrix_library"),
    ],
)
@pytest.mark.parametrize("include", ['"scf/rhf.hpp"', '"../rhf.hpp"', "<scf/rhf.hpp>"])
def test_cuda_runtime_cannot_depend_on_method_driver(tmp_path, name, owner, include):
    source = tmp_path / "src/scf"
    (source / "cuda").mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / "cuda" / name).write_text(f"#include {include}\n")
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert f"forbidden {owner} dependency on scf/rhf.hpp" in errors[0]


def test_eigensolver_cannot_acquire_direct_queue_policy(tmp_path):
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "direct_constants.hpp").write_text("// Direct queue policy\n")
    (source / "eigensolver.cpp").write_text('#include "direct_constants.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner", ["df_jk_kernels.cu", "df_scf_kernels.cu", "scf_density_kernels.cu"]
)
def test_df_kernels_cannot_acquire_host_plan_state(tmp_path, owner):
    """Kernel changes must remain independent of resource and graph lifetimes."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "df_plan_internal.hpp").write_text("// Plan-owned allocations\n")
    (source / owner).write_text('#include "df_plan_internal.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


def test_matrix_library_cannot_acquire_bucket_resource_owner(tmp_path):
    """Matrix consumers borrow handles without depending on allocation lifetime."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "resources.hpp").write_text("// Stream/graph/arena owner\n")
    (source / "matrix_library.cpp").write_text('#include "resources.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner", ["direct_bounded_tasks.cu", "direct_queue_scan.cu", "direct_screening.cuh"]
)
@pytest.mark.parametrize(
    "dependency",
    ["resources.hpp", "one_electron_native_overlap.cuh", "df_plan_internal.hpp"],
)
def test_direct_queue_owners_cannot_acquire_plan_or_integral_state(
    tmp_path, owner, dependency
):
    """Queue rebuilds stay independent of host ownership and integral recurrences."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / dependency).write_text("// Separately owned plan or scientific code\n")
    (source / owner).write_text(f'#include "{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner",
    ["direct_jk.cpp", "one_electron_export.cpp", "one_electron_export_batch.cpp"],
)
@pytest.mark.parametrize(
    "dependency",
    ["direct_jk_kernels.cu", "one_electron_native_force.cuh", "resources.hpp"],
)
def test_provider_host_owners_cannot_import_recurrences_or_scf_lifetime(
    tmp_path, owner, dependency
):
    """Provider host rebuilds borrow launches instead of device implementations."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / dependency).write_text("// Separate recurrence or SCF owner\n")
    (source / owner).write_text(f'#include "{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner",
    [
        "direct_jk_kernels.hpp",
        "direct_jk_kernels.cu",
        "one_electron_export_kernels.hpp",
    ],
)
def test_provider_kernel_interfaces_cannot_acquire_plan_state(tmp_path, owner):
    """A consumer interface must remain usable without host plan allocations."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "direct_jk_plan.hpp").write_text("// Host allocation owner\n")
    (source / owner).write_text('#include "direct_jk_plan.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner",
    [
        "scalar_math.cuh",
        "hermite_recurrence.cuh",
        "one_electron_reference.cu",
        "one_electron_native_force.cuh",
        "nuclear_kernels.cu",
        "direct_pair_cache.cu",
    ],
)
@pytest.mark.parametrize(
    "dependency", ["direct_constants.hpp", "direct_jk_plan.hpp", "resources.hpp"]
)
def test_retained_numerics_cannot_acquire_queue_policy_or_plan_state(
    tmp_path, owner, dependency
):
    """Scientific implementations remain independent of scheduling and lifetime."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / dependency).write_text("// Queue policy or host ownership\n")
    (source / owner).write_text(f'#include "{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


def test_shared_numerics_cannot_import_operator_contractions(tmp_path):
    """Adding a consumer must not make the shared recurrence depend on it."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "one_electron_native_force.cuh").write_text("// Force contraction\n")
    (source / "coulomb_auxiliary.cuh").write_text(
        '#include "one_electron_native_force.cuh"\n'
    )
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner", ["direct_native_cartesian.cuh", "direct_native_order3_gradient.cuh"]
)
@pytest.mark.parametrize(
    "dependency", ["direct_constants.hpp", "direct_angular_fock.hpp", "resources.hpp"]
)
def test_direct_numerical_families_cannot_acquire_policy_or_consumers(
    tmp_path, owner, dependency
):
    """Native formulas borrow class indexing without acquiring launch policy."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / dependency).write_text("// Policy, consumer, or lifetime owner\n")
    (source / owner).write_text(f'#include "{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner",
    [
        "direct_fock_order2.cuh",
        "direct_force_low_order.cuh",
        "direct_bounded_fallback.cu",
    ],
)
def test_direct_contractions_cannot_acquire_host_resources(tmp_path, owner):
    """Fused native consumers stay independent of graph and allocation lifetime."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "resources.hpp").write_text("// Host-owned allocations\n")
    (source / owner).write_text('#include "resources.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


def test_direct_launch_interface_cannot_import_its_implementation(tmp_path):
    """Sharing a basename must not weaken the host/device interface boundary."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "direct_angular_fock.cu").write_text("// Separately compiled kernels\n")
    (source / "direct_angular_fock.hpp").write_text(
        '#include "direct_angular_fock.cu"\n'
    )
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "dependency", ["direct_native_cartesian.cuh", "direct_bounded_fallback.cu"]
)
def test_cpp_hf_driver_cannot_import_device_implementations(tmp_path, dependency):
    """C++ orchestration must keep borrowing launches after numerical extraction."""
    source = tmp_path / "src/scf"
    (source / "cuda").mkdir(parents=True)
    (source / "cuda" / dependency).write_text("// Separately compiled arithmetic\n")
    (source / "cuda_rhf.cpp").write_text(f'#include "cuda/{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "dependency", ["tensor/cuda_runtime.cuh", "scf/cuda/direct_native_cartesian.cuh"]
)
def test_reference_export_uses_only_host_cuda_interfaces(tmp_path, dependency):
    """Exporting a physical reference must not pull device syntax into C++."""
    source = tmp_path / "src"
    target = source / dependency
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("// Contains device kernels\n")
    bridge = source / "scf/cuda/reference_export.cuh"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    bridge.write_text(f'#include "{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1
