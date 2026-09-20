"""Archive bounded DF counters and independently observed CUDA transfer sizes."""

import argparse
import hashlib
import json
import sqlite3
import typing
from pathlib import Path


def summarize(directory: typing.Any) -> typing.Any:
    """Retain raw timing samples and trace hashes; distinguish capture scopes."""
    components = {
        path.name: json.loads(path.read_text())
        for path in sorted(directory.glob("component-*.json"))
    }
    if len(components) != 11:
        raise ValueError("expected eleven component records")
    profiles = []
    for path in sorted(directory.glob("profile-*.sqlite")):
        with sqlite3.connect(path) as database:
            transfers = [
                {"direction": kind, "bytes_per_copy": size, "copies": count}
                for kind, size, count in database.execute(
                    "select e.label,m.bytes,count(*) from CUPTI_ACTIVITY_KIND_MEMCPY m "
                    "join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id "
                    "group by e.label,m.bytes order by e.label,m.bytes"
                )
            ]
            profiles.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "scope": (
                        "five complete RHF energy-plus-force replays, batch 3"
                        if "complete" in path.name
                        else "five DF force responses, batch 1; setup outside capture"
                    ),
                    "transfers": transfers,
                }
            )
    if len(profiles) != 3:
        raise ValueError("expected two force traces and one complete HF trace")
    # A force response returns exactly six gradient doubles per replay. This
    # independently checks the bridge counters, rather than trusting zeros
    # printed by the implementation. H2D includes metadata and input densities.
    for profile in profiles:
        if "force-" in profile["file"]:
            downloads = [
                row
                for row in profile["transfers"]
                if row["direction"] in {"DtoH", "Device-to-Host"}
            ]
            if [(row["bytes_per_copy"], row["copies"]) for row in downloads] != [
                (48, 5)
            ]:
                raise ValueError(f"unexpected force downloads: {downloads}")
            spin = "uhf" if "-uhf" in profile["file"] else "rhf"
            expected = components[f"component-sp8-{spin}-generated_source-16384.json"]
            uploads = sum(
                row["bytes_per_copy"] * row["copies"]
                for row in profile["transfers"]
                if row["direction"] in {"HtoD", "Host-to-Device"}
            )
            if uploads != 5 * expected["response_resources"]["h2d_bytes"]:
                raise ValueError("force H2D counters disagree with the CUDA trace")
    return {"components": components, "profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (args.directory / "summary.json").write_text(
        json.dumps(summarize(args.directory), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
