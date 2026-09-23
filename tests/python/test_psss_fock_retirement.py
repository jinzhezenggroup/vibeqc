"""Guards for compiler-owned psss Direct-HF Fock production."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_fixed_topology_psss_is_not_excluded_from_generated_fock() -> None:
    source = _source("src/scf/cuda/direct_constants.hpp")
    begin = source.index("kFixedTopologyGeneratedFockExclusionMask")
    selection = source[begin : begin + 180]
    assert "std::uint64_t{1} << 0U" in selection
    assert "kDdddShellClassMask" in selection
    assert "<< 1U" not in selection


def test_psss_fock_has_no_dedicated_handwritten_scientific_body() -> None:
    assert not (ROOT / "src/scf/cuda/direct_fock_psss.cuh").exists()
    angular = _source("src/scf/cuda/direct_angular_fock.cu")
    bounded = _source("src/scf/cuda/direct_bounded_fallback.cu")
    native_psss = _source("src/scf/cuda/direct_native_psss.cuh")
    assert "build_fock_direct_psss_persistent_kernel" not in angular
    assert "contract_fock_direct_psss_task" not in angular
    assert "contract_fock_direct_psss_task" not in bounded
    assert "contracted_eri_cartesian_source_psss(" not in native_psss
    assert "PsssIntegralVector" not in native_psss
    assert "generated_weighted_eri::psss_force" in native_psss


def test_bounded_fock_keeps_generic_order_one_fallback() -> None:
    source = _source("src/scf/cuda/direct_bounded_fallback.cu")
    marker = "Fock order one has no psss-specific handwritten fallback anymore."
    begin = source.index(marker)
    fallback = source[begin : begin + 450]
    assert "angular_order == 0U || angular_order == 2U" in fallback
    assert "contract_bounded_direct_fock_subtile<Unrestricted>" in source
