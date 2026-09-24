import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/xtb/native"
MANIFEST = RUNTIME / "CUDA_SOURCE_PROVENANCE.json"


def test_gfn2_cuda_source_manifest_is_current_and_gfn2_only() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == "3c21f50195389b093941eb5ed6f1143b8802f96e"
    assert manifest["cpu_snapshot_commit"] == "5a67cc59ace94c8296e873503b2ae1298e7c2861"
    assert len(manifest["cuda_sources"]) == 52
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

    # Every imported unit changed ownership namespace/includes. Scientific
    # adaptations are recorded separately from this mechanical common cutover.
    assert adapted == list(manifest["files"])
    assert manifest["native_ownership"]["root"] == "src/xtb/native"
    assert "src/backends/cuda/gfn2_spin.cu" in manifest["adaptations"]


def test_gfn2_cuda_reuses_canonical_d4_data() -> None:
    source = (RUNTIME / "src/runtime/gfn2_cuda_execution.cu").read_text(
        encoding="utf-8"
    )
    assert '#include "dft/dispersion/d4_data.hpp"' in source
    assert '#include "data/parameters/d4.hpp"' not in source
    assert "canonical_d4::kReferenceC6" in source
    assert "high * (high + 1u) / 2u + low" in source
    assert not (RUNTIME / "data/parameters/d4.hpp").exists()


def test_gfn2_cuda_d4_reuses_shared_scalar_math() -> None:
    source = (RUNTIME / "src/backends/cuda/gfn2_d4.cu").read_text(encoding="utf-8")
    assert '#include "dft/dispersion/d4_math.hpp"' in source
    for helper in (
        "d4_math::atom_weights",
        "d4_math::coefficient",
        "d4_math::coordination_pair",
        "d4_math::pair_damping",
        "d4_math::damping_radius",
    ):
        assert helper in source
    for retired in (
        "__device__ double charge_scale(",
        "kReferenceWeightFactor",
        "kMinimumWeightNorm",
        "kCoordinationSteepness",
        "kEnK4",
        "kEnK5",
        "kEnK6",
    ):
        assert retired not in source


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
