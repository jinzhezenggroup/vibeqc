"""Measure native SASS entry sizes without loading CUDA or probing a GPU.

Run from the repository root. Each entry includes its inlined libdevice and
diagnostic paths, so these are compiled kernel sizes, not primitive IR sizes.
The complete library size also includes host code, PTX and fatbin metadata.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def measure(library, cuobjdump):
    """Locate the derivative module by its symbols instead of a build index."""
    library = library.resolve()
    symbols = subprocess.check_output(
        [cuobjdump, "--dump-elf-symbols", str(library)], text=True
    )
    modules = symbols.split("Fatbin elf code:")[1:]
    selected = [
        index
        for index, module in enumerate(modules, 1)
        if "shell_packetILj0ELj0ELj0E" in module
    ]
    assert len(selected) == 1, "requires one production GPU architecture"
    listing = subprocess.check_output(
        [cuobjdump, "--list-elf", str(library)], text=True
    )
    filename = re.search(rf"ELF file\s+{selected[0]}:\s+(\S+)", listing).group(1)
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [cuobjdump, "--extract-elf", filename, str(library)],
            cwd=directory,
            check=True,
            capture_output=True,
        )
        cubin = Path(directory) / filename
        result = subprocess.run(
            ["readelf", "--wide", "--symbols", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        )
        # GNU readelf warns about CUDA-specific local symbol ordering. Keep its
        # successful symbol table and validate every parsed entry independently.
        table = subprocess.check_output(["c++filt"], input=result.stdout, text=True)
        rows = []
        for line in table.splitlines():
            match = re.search(
                r"::shell_(packet|panel)<(\d+)u, (\d+)u, (\d+)u, (\d+)u"
                r"(?:, (true|false))?>",
                line,
            )
            if match and " FUNC " in line:
                size = int(line.split()[2], 0)
                assert size > 0
                rows.append(
                    {
                        "family": match[1],
                        "angular": list(map(int, match.groups()[1:4])),
                        "variant": int(match[5]),
                        "lowering": "rys" if match[6] == "true" else "polynomial",
                        "sass_entry_bytes": size,
                    }
                )
        assert rows and any(row["angular"] == [0, 0, 0] for row in rows)
        with library.open("rb") as stream:
            library_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "library_sha256": library_digest,
            "library_bytes": library.stat().st_size,
            "derivative_cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
            "derivative_cubin_bytes": cubin.stat().st_size,
            "all_shell_entry_bytes": sum(row["sass_entry_bytes"] for row in rows),
            "shell_entry_count": len(rows),
            "000_entries": [row for row in rows if row["angular"] == [0, 0, 0]],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "ELF FUNC symbol sizes of complete sm_120 SASS kernel entries; includes inlined helpers and diagnostic branches",
        "baseline": measure(args.baseline, args.cuobjdump),
        "candidate": measure(args.candidate, args.cuobjdump),
    }
    result["library_growth_bytes"] = (
        result["candidate"]["library_bytes"] - result["baseline"]["library_bytes"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
