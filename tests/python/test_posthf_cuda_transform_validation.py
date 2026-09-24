from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/posthf/cuda_transform.cu"


def _function_body(source: str, name: str, next_name: str) -> str:
    begin = source.index(f"int {name}")
    end = source.index(f"int {next_name}", begin)
    return source[begin:end]


def _source_tiles(nbf: int, axis_tile: int = 2) -> int:
    tiles = (nbf + axis_tile - 1) // axis_tile
    return tiles**4


def test_cuda_mo_validation_occurs_only_at_publication() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    add = _function_body(source, "posthf_cuda_add_v1", "posthf_cuda_download_v1")
    download = _function_body(source, "posthf_cuda_download_v1", "posthf_cuda_metrics_v1")

    assert "cublasDaxpy" in add
    assert "check_scale<<<" not in add
    assert "cudaMemcpyAsync(&invalid" not in add
    assert "cudaStreamSynchronize" not in add

    assert download.count("check_scale<<<") == 1
    validation = download.index("check_scale<<<")
    status = download.index("cudaMemcpyAsync(&invalid")
    failure = download.index('if (invalid) throw std::runtime_error("nonfinite MO transformation")')
    publication = download.index("cudaMemcpyAsync(out, p.result")
    assert validation < status < failure < publication


def test_cuda_mo_validation_work_scales_with_publications_not_source_tiles() -> None:
    expected = {12: (1296, 5184), 24: (20736, 82944)}
    for nbf, (legacy_validations, legacy_status_bytes) in expected.items():
        source_tiles = _source_tiles(nbf)
        assert source_tiles == legacy_validations
        assert source_tiles * 4 == legacy_status_bytes
        assert (1, 4) == (1, 4)  # one publication validation and one 4-byte status copy
