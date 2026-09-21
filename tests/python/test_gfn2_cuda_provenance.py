import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/xtb/gfn2_runtime"
MANIFEST = RUNTIME / "CUDA_SOURCE_PROVENANCE.json"


def test_gfn2_cuda_source_manifest_is_current_and_gfn2_only() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == "3c21f50195389b093941eb5ed6f1143b8802f96e"
    assert manifest["cpu_snapshot_commit"] == "5a67cc59ace94c8296e873503b2ae1298e7c2861"
    assert len(manifest["cuda_sources"]) == 53
    assert manifest["files"]

    adapted = []
    for relpath, record in manifest["files"].items():
        path = RUNTIME / relpath
        assert path.is_file(), relpath
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["vendored_sha256"], relpath
        if record["adapted"]:
            adapted.append(relpath)

        lowered = relpath.lower()
        assert "model/gfn1" not in lowered
        assert "parameters/gfn1" not in lowered
        assert "gfn1_classical_corrections" not in lowered

    assert adapted == [
        "src/backends/cuda/gfn2_geometry.cu",
        "src/backends/cuda/gfn2_pairlist.cu",
        "src/backends/cuda/gfn2_preprocessing.cu",
        "src/backends/cuda/gfn2_repulsion.cu",
        "src/backends/cuda/gfn2_repulsion.cuh",
        "src/runtime/gfn2_cuda_execution.cu",
    ]


def test_gfn2_cuda_reuses_canonical_d4_data() -> None:
    source = (RUNTIME / "src/runtime/gfn2_cuda_execution.cu").read_text(
        encoding="utf-8"
    )
    assert '#include "dft/dispersion/d4_data.hpp"' in source
    assert '#include "data/parameters/d4.hpp"' not in source
    assert "canonical_d4::kReferenceC6" in source
    assert "high * (high + 1u) / 2u + low" in source
    assert not (RUNTIME / "data/parameters/d4.hpp").exists()


def test_gfn2_cuda_kernels_do_not_take_reference_parameters() -> None:
    import re

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    signature = re.compile(r"__global__\s+[^({;]+\((.*?)\)\s*\{", re.DOTALL)
    offenders = []
    for relpath in manifest["cuda_sources"]:
        source = (RUNTIME / relpath).read_text(encoding="utf-8")
        for match in signature.finditer(source):
            if "&" in match.group(1):
                offenders.append(relpath)
                break
    assert offenders == []
