"""Reproducibility tests for fixed-root GPU4PySCF table slices."""

from tools.vibeqc_codegen.extract_rys_table import (
    DEGREE,
    GPU4PYSCF_COMMIT,
    GPU4PYSCF_SOURCE_SHA256,
    INTERVALS,
    emit_python_module,
    extract_fixed_root_tables,
)


def _cuda_array(name: str, values: list[float]) -> str:
    return f"__device__ double {name}[] = {{" + ",".join(map(str, values)) + "};"


def test_extract_fixed_root_tables_uses_gpu4pyscf_offsets():
    """Keep triangular and interpolation offsets aligned with rys_roots.cu."""

    fixed_values = [float(index) for index in range(10)]
    interpolation_width = (DEGREE + 1) * INTERVALS
    interpolation_values = [float(index) for index in range(6 * interpolation_width)]
    source = "\n".join(
        _cuda_array(
            name, interpolation_values if name == "ROOT_RW_DATA" else fixed_values
        )
        for name in (
            "ROOT_SMALLX_R0",
            "ROOT_SMALLX_R1",
            "ROOT_SMALLX_W0",
            "ROOT_SMALLX_W1",
            "ROOT_LARGEX_R_DATA",
            "ROOT_LARGEX_W_DATA",
            "ROOT_RW_DATA",
        )
    )
    tables = extract_fixed_root_tables(source, 2)
    assert tables["ROOT_SMALLX_R0"] == (1.0, 2.0)
    assert tables["ROOT_RW_DATA"] == tuple(
        interpolation_values[2 * interpolation_width : 6 * interpolation_width]
    )
    module = emit_python_module(2, tables)
    assert "RYS2_RW_DATA" in module
    assert "nroots == 2" in module
    assert GPU4PYSCF_COMMIT in module
    assert GPU4PYSCF_SOURCE_SHA256 in module
