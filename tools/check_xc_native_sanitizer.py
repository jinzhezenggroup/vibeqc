"""Exercise generated CPU XC point ABIs under Address/UndefinedBehaviorSanitizer.

This standalone diagnostic compiles executable harnesses with explicit sanitizer
flags. It does not inject instrumentation into the production artifact cache or
claim that a sanitizer harness measures complete endpoint performance.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.native import emit_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in ("LDA_XC_PW", "PBE"):
        for spin in ("polarized", "unpolarized"):
            for observable in ("energy", "potential", "response", "geometry"):
                program = ContractionProgram(functional(name, spin=spin), observable)
                source, metadata = emit_native(program)
                driver = [source, "#include <vector>", "int main() {"]
                for symbol, layout in metadata["layouts"].items():
                    ni, no = len(layout["variables"]), layout["outputs"]
                    driver += [
                        "{",
                        f"std::vector<double> x({ni}*7, 0.2), y({no}*7, 0.0);",
                    ]
                    # Equal positive gradients define a physical rank-one sigma
                    # Gram matrix; all coefficient/contracted-jet inputs are finite.
                    driver += [
                        f"if ({symbol}(x.data(),x.size(),y.data(),y.size(),7)) return 1;",
                        f"if ({symbol}(x.data(),x.size()+1,y.data(),y.size(),7) != -1) return 2;",
                        f"if ({symbol}(nullptr,x.size(),y.data(),y.size(),7) != -1) return 3;",
                        f"if ({symbol}(x.data(),x.size(),nullptr,y.size(),7) != -1) return 4;",
                        f"if ({symbol}(nullptr,0,nullptr,0,0)) return 5;",
                        f"if ({symbol}(nullptr,0,nullptr,0,SIZE_MAX) != -1) return 6;",
                        "}",
                    ]
                driver += ["return 0;", "}"]
                label = f"{name}-{spin}-{observable}"
                path, executable = args.output / f"{label}.cpp", args.output / label
                path.write_text("\n".join(driver) + "\n")
                command = [
                    args.cxx,
                    "-std=c++17",
                    "-O1",
                    "-g",
                    "-ffp-contract=off",
                    "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer",
                    "-no-pie",
                    str(path),
                    "-o",
                    str(executable),
                ]
                subprocess.run(command, check=True, timeout=180)
                subprocess.run(
                    [str(executable.resolve())],
                    check=True,
                    timeout=30,
                    env={
                        **os.environ,
                        "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                        "UBSAN_OPTIONS": "halt_on_error=1",
                    },
                )
                rows.append(
                    {
                        "case": label,
                        "point_source_sha256": metadata["source_sha256"],
                        "harness_sha256": file_hash(path),
                        "command": command,
                        "passed": True,
                    }
                )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "schema": "vibeqc.xc-native-sanitizer.v1",
                "compiler": subprocess.check_output([args.cxx, "--version"], text=True),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(rows)} native ABI sanitizer cases passed")


if __name__ == "__main__":
    main()
