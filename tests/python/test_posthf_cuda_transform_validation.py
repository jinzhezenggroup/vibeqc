from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/posthf/cuda_transform.cu"
OWNER = ROOT / "tools/vibeqc_posthf/cuda.py"
PROVIDER = ROOT / "tools/vibeqc_posthf/providers.py"


def _function_body(source: str, name: str, next_name: str) -> str:
    begin = source.index(f"int {name}")
    end = source.index(f"int {next_name}", begin)
    return source[begin:end]


def _source_tiles(nbf: int, axis_tile: int = 2) -> int:
    tiles = (nbf + axis_tile - 1) // axis_tile
    return tiles**4


def test_cuda_mo_validation_occurs_once_per_completed_block() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    add = _function_body(source, "posthf_cuda_add_v1", "posthf_cuda_validate_v1")
    validate_api = _function_body(
        source, "posthf_cuda_validate_v1", "posthf_cuda_download_v1"
    )
    download = _function_body(
        source, "posthf_cuda_download_v1", "posthf_cuda_metrics_v1"
    )
    helper_begin = source.index("void validate(Transform& p)")
    helper_end = source.index("}  // namespace", helper_begin)
    helper = source[helper_begin:helper_end]

    assert "cublasDaxpy" in add
    assert "check_scale<<<" not in add
    assert "cudaMemcpyAsync(&invalid" not in add
    assert "cudaStreamSynchronize" not in add
    assert "p.validated = false" in add

    assert source.count("check_scale<<<") == 1
    assert "if (p.validated) return" in helper
    validation = helper.index("check_scale<<<")
    status = helper.index("cudaMemcpyAsync(&invalid")
    failure = helper.index(
        'if (invalid) throw std::runtime_error("nonfinite MO transformation")'
    )
    assert validation < status < failure
    assert "validate(p)" in validate_api
    assert download.index("validate(p)") < download.index(
        "cudaMemcpyAsync(out, p.result"
    )


def test_resident_cuda_mo_block_is_validated_before_publication() -> None:
    owner = OWNER.read_text(encoding="utf-8")
    provider = PROVIDER.read_text(encoding="utf-8")

    pointer = owner[owner.index("def device_pointer") : owner.index("def to_host")]
    assert '_call("posthf_cuda_validate_v1", self._handle)' in pointer
    assert pointer.index("posthf_cuda_validate_v1") < pointer.index(
        "posthf_cuda_pointer_v1"
    )

    validation = provider.index("engine.validate()")
    diagnostics = provider.index("diagnostics = {", validation)
    assert validation < diagnostics


def test_cuda_mo_validation_work_scales_with_publications_not_source_tiles() -> None:
    expected = {
        12: (1296, 1, 5184, 4),
        24: (20736, 1, 82944, 4),
    }
    for nbf, census in expected.items():
        source_tiles = _source_tiles(nbf)
        measured = (source_tiles, 1, source_tiles * 4, 4)
        assert measured == census


def test_native_multi_request_cuda_reuses_one_raw_upload_and_context() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    provider = (ROOT / "src/posthf/native_provider.cpp").read_text(encoding="utf-8")

    batch_add = _function_body(
        source, "posthf_cuda_batch_add_v1", "posthf_cuda_batch_download_v1"
    )
    batch_download = _function_body(
        source, "posthf_cuda_batch_download_v1", "posthf_cuda_batch_metrics_v1"
    )

    assert batch_add.count("cudaMemcpyAsync(p.raw, values") == 1
    assert "for (auto& state : p.states)" in batch_add
    assert batch_add.count("ctx.section(true, ctx.metrics.input_ms") == 1
    assert batch_add.count("ctx.section(true, ctx.metrics.library_ms") == 1
    assert batch_download.count("ctx.section(true, ctx.metrics.output_ms") == 1

    assert "posthf_cuda_batch_create_v1(" in provider
    assert "posthf_cuda_batch_add_v1(" in provider
    assert "posthf_cuda_batch_download_v1(" in provider
    assert "device_blocks.pointers" not in provider
