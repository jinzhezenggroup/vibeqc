"""Timeline evidence must distinguish overlapping API, compute and DMA clocks."""

import sqlite3

import pytest

from benchmarks.df_response_timeline import (
    execution_coverage,
    intersect_duration,
    interval_union,
    summarize,
)


def test_gpu_coverage_unions_overlaps_and_includes_memsets():
    result = execution_coverage(
        (0, 30),
        [{"start": 2, "end": 12}, {"start": 8, "end": 18}],
        [{"start": 10, "end": 20}],
        [{"start": 25, "end": 28}],
    )
    assert result["active_union_ms"] == 21 / 1e6
    assert result["no_recorded_gpu_work_ms"] == 9 / 1e6


def test_panel_attribution_retains_kernels_after_host_range_exit(tmp_path):
    """Asynchronous kernels belong to the panel that submitted their API."""
    path = tmp_path / "panels.sqlite"
    with sqlite3.connect(path) as c:
        c.executescript("""
            create table StringIds(id integer, value text);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(
                start integer, end integer, globalTid integer,
                correlationId integer, nameId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(
                start integer, end integer, streamId integer,
                correlationId integer, demangledName integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(
                start integer, end integer, streamId integer,
                correlationId integer, bytes integer);
            create table NVTX_EVENTS(
                start integer, end integer, globalTid integer, text text, textId integer);
            insert into StringIds values
                (1,'cudaLaunchKernel'),(2,'derivative'),(3,'cudaMemcpyAsync');
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values
                (6,7,10,1,3),(15,16,10,2,1),(17,18,10,3,3),(21,22,10,4,1);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values
                (8,9,7,1,512),(26,27,7,3,256);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values
                (18,25,7,2,2),(28,35,7,4,2);
            insert into NVTX_EVENTS values
                (0,40,10,'force_response',null),
                (1,5,10,'coulomb_response',null),
                (6,7,10,'raw_value_slice_upload',null),
                (15,16,10,'three_center_derivative_contraction',null),
                (17,18,10,'raw_value_slice_upload',null),
                (21,22,10,'three_center_derivative_contraction',null);
        """)
    panels = summarize(path)["response_panels"]
    assert len(panels) == 2
    assert [p["raw_copy_bytes"] for p in panels] == [512, 256]
    for panel in panels:
        component = panel["components"]["three_center_derivative_contraction"]
        assert component == {"kernel_calls": 1, "device_ms": 7 / 1e6}


def test_intersections_do_not_double_count_concurrent_activity():
    assert interval_union([(8, 12), (0, 5), (4, 9), (15, 18)]) == [(0, 12), (15, 18)]
    assert intersect_duration([(0, 10), (5, 15)], [(8, 12), (10, 20)]) == 7
    assert intersect_duration([(0, 10)], [(12, 20)]) == 0
    with pytest.raises(ValueError, match="reversed"):
        interval_union([(2, 1)])


@pytest.mark.parametrize("traced", (False, True))
@pytest.mark.parametrize("extra_copy", (False, True))
def test_copy_wait_overlap_and_unassigned_host_residual(tmp_path, traced, extra_copy):
    """A copy API spans a prior kernel, DMA and host-only staging/control.

    Nested NVTX and the untraced original route must identify the same raw
    bytes; dense pinned copies need the NVTX origin to exclude density uploads.
    A bulk upload can produce multiple DMA records for one API submission.
    """
    path = tmp_path / "timeline.sqlite"
    with sqlite3.connect(path) as c:
        c.executescript("""
            create table StringIds(id integer, value text);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(
                start integer, end integer, globalTid integer,
                correlationId integer, nameId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(
                start integer, end integer, streamId integer,
                correlationId integer, demangledName integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(
                start integer, end integer, streamId integer,
                correlationId integer, bytes integer);
            insert into StringIds values(1,'cudaLaunchKernel'),(2,'prior_compute');
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values(0,1,10,1,1),(5,25,10,2,3);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values(2,15,7,1,2);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values(20,23,7,2,1024);
        """)
        c.execute(
            "insert into StringIds values(3,?)",
            ("cudaMemcpyAsync" if traced else "cudaMemcpy2DAsync",),
        )
        if extra_copy:
            c.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(26,28,7,2,256)")
        if traced:
            c.executescript("""
                create table NVTX_EVENTS(
                    start integer, end integer, globalTid integer, text text, textId integer);
                insert into NVTX_EVENTS values
                    (0,30,10,'force_response',null),
                    (3,26,10,'raw_value_slice_upload',null);
            """)
            if extra_copy:
                c.execute(
                    "update NVTX_EVENTS set text='raw_value_resident_upload' "
                    "where text='raw_value_slice_upload'"
                )
    result = summarize(path)
    raw = result["raw_copies"]
    assert raw["bytes"] == 1024 + 256 * extra_copy
    assert raw["calls"] == 1 + extra_copy
    assert raw["host_api_ms"] == 20 / 1e6
    assert raw["device_dma_ms"] == (3 + 2 * extra_copy) / 1e6
    assert raw["host_overlap_same_stream_kernels_ms"] == 10 / 1e6
    assert raw["host_overlap_same_stream_copies_ms"] == 3 / 1e6
    assert raw["host_residual_not_device_activity_ms"] == 7 / 1e6
    scope = result["raw_copy_scope"]
    if traced:
        assert scope["host_nvtx_ms"] == 23 / 1e6
        assert scope["host_overlap_same_stream_kernels_ms"] == 12 / 1e6
        assert scope["host_overlap_same_stream_copies_ms"] == 3 / 1e6
        assert scope["host_residual_not_device_activity_ms"] == 8 / 1e6
    else:
        assert scope is None


@pytest.mark.parametrize("identify_generation", (False, True))
def test_source_response_requires_executed_generation(tmp_path, identify_generation):
    """A density upload alone cannot substantiate a zero-raw-transfer claim."""
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as c:
        c.executescript("""
            create table StringIds(id integer, value text);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(
                start integer, end integer, globalTid integer,
                correlationId integer, nameId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(
                start integer, end integer, streamId integer,
                correlationId integer, demangledName integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(
                start integer, end integer, streamId integer,
                correlationId integer, bytes integer);
            create table NVTX_EVENTS(
                start integer, end integer, globalTid integer, text text, textId integer);
            insert into StringIds values
                (1,'cudaLaunchKernel'),(2,'generated_raw'),(3,'cudaMemcpyAsync');
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values(0,1,10,1,3),(4,5,10,2,1);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values(1,3,7,1,1024);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values(6,16,7,2,2);
            insert into NVTX_EVENTS values(0,20,10,'force_response',null);
        """)
        if identify_generation:
            c.execute(
                "insert into NVTX_EVENTS values(4,5,10,'raw_three_center_generation',null)"
            )
    if not identify_generation:
        with pytest.raises(ValueError, match="missing raw-value copies or generation"):
            summarize(path)
        return
    result = summarize(path)
    assert result["raw_value_provider"] == "source"
    assert result["raw_generation"] == {"calls": 1, "device_ms": 10 / 1e6}
    assert result["raw_copies"]["bytes"] == 0
    assert result["raw_copies"]["device_dma_ms"] == 0
    assert result["raw_copy_scope"] is None
