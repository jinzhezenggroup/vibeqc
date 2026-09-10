"""Collect original object resources and bind them to endpoint worker identities."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--comparison", type=Path, required=True)
parser.add_argument("--baseline-build", type=Path, required=True)
parser.add_argument("--candidate-retained", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--cuda-home", type=Path, help="CUDA toolkit containing bin/nvcc and bin/cuobjdump"
)
args = parser.parse_args()


def cuda_tool(name):
    path = str(args.cuda_home / "bin" / name) if args.cuda_home else shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is unavailable; supply --cuda-home")
    return path


compiler = subprocess.check_output([cuda_tool("nvcc"), "--version"], text=True)
result = {}
for label, folder in (
    ("baseline", args.baseline_build),
    ("candidate", args.candidate_retained),
):
    worker = json.loads((args.comparison / (label + "-0.json")).read_text())
    assert (
        hashlib.sha256((folder / "libvibeqc.so").read_bytes()).hexdigest()
        == worker["library_sha256"]
    )
    identity_file = folder / "identity.json"
    build_identity = (
        json.loads(identity_file.read_text()) if identity_file.exists() else {}
    )
    if build_identity:
        for key in ("revision", "native_source_identity", "library_sha256"):
            assert build_identity[key] == worker[key], key
    for part, relative in (
        ("rhf", "src/scf/cuda_rhf.cu.o"),
        ("weighted", "src/scf/cuda/df_derivatives.cu.o"),
    ):
        obj = folder / "CMakeFiles/vibeqc.dir" / relative
        # Retained timing builds carry their original objects beside the library.
        if label == "candidate" and not obj.exists():
            obj = folder / (part + ".o")
        dump = subprocess.check_output(
            [cuda_tool("cuobjdump"), "--dump-resource-usage", str(obj)]
        )
        resources = {}
        for symbol, usage in re.findall(
            r" Function (\S+):\n\s+([^\n]+)", dump.decode()
        ):
            if ("build_cuda_df_" if part == "rhf" else "derivative_tile") not in symbol:
                continue
            name = subprocess.check_output(["c++filt", symbol], text=True).strip()
            assert name not in resources
            resources[name] = usage
        assert len(resources) == (6 if part == "rhf" else 1), resources
        result[label + "-" + part] = {
            "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            "object_bytes": obj.stat().st_size,
            "measurement": "original-object",
            "original_native_build_revision": build_identity.get(
                "original_native_build_revision", worker["revision"]
            ),
            "build_reuse_note": build_identity.get(
                "compatibility_note",
                "Objects and timing library were built at the measured revision.",
            ),
            "compiler": compiler,
            "resources": resources,
            "dump_sha256": hashlib.sha256(dump).hexdigest(),
            "provenance": {
                key: worker[key]
                for key in (
                    "revision",
                    "native_source_identity",
                    "library_sha256",
                    "build_settings",
                )
            },
        }
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(args.output)
