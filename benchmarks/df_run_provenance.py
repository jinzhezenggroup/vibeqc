"""Write identities and trace summaries inside a fresh DF benchmark run directory."""

import argparse
import ctypes
import gzip
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibeqc import _native
from vibeqc.autotune import source_identity

from tools.vibeqc_validation.schema import file_hash


def summarize_profiles(directory):
    """Read exact transfer bytes and kernel durations from each captured window."""
    records = []
    for path in sorted(directory.glob("profile-*.sqlite.gz")):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "trace.sqlite"
            with gzip.open(path, "rb") as stream:
                database.write_bytes(stream.read())
            with sqlite3.connect(database) as connection:
                records.append(
                    {
                        "profile": path.name.removesuffix(".sqlite.gz"),
                        "scope": "complete captured HF window; five warm replays of three systems",
                        "transfers": {
                            row[0]: {"bytes": row[1], "count": row[2]}
                            for row in connection.execute(
                                "select e.label,sum(m.bytes),count(*) from CUPTI_ACTIVITY_KIND_MEMCPY m "
                                "join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id group by e.label"
                            )
                        },
                        "synchronizations": {
                            row[0]: row[1]
                            for row in connection.execute(
                                "select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r "
                                "join StringIds s on r.nameId=s.id where s.value like '%Synchronize%' group by s.value"
                            )
                        },
                        "kernels": [
                            {
                                "name": row[0],
                                "milliseconds": row[1] / 1e6,
                                "calls": row[2],
                            }
                            for row in connection.execute(
                                "select s.value,sum(k.end-k.start),count(*) from CUPTI_ACTIVITY_KIND_KERNEL k "
                                "join StringIds s on k.demangledName=s.id group by s.value order by sum(k.end-k.start) desc"
                            )
                        ],
                    }
                )
    if len(records) != 4:
        raise RuntimeError("expected four complete profile databases")
    return {
        "scope": "This run only; identities are in provenance.json",
        "records": records,
    }


def main():
    """Require native/source agreement without creating a GPU execution context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profiles", action="store_true")
    args = parser.parse_args()
    if args.profiles:
        payload = summarize_profiles(args.directory)
        name = "profile-summary.json"
    else:
        library = _native.load_library()
        library.vibeqc_get_source_identity.restype = ctypes.c_char_p
        identity = library.vibeqc_get_source_identity().decode()
        if identity != source_identity(ROOT):
            raise RuntimeError("loaded library does not match the scientific source")
        payload = {
            "source_identity": identity,
            "binary_sha256": file_hash(library._name),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "scope": "Fresh run; no historical benchmark records or aggregates are reused",
        }
        name = "provenance.json"
    # Exclusive creation also prevents accidental replacement within a run.
    with (args.directory / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
