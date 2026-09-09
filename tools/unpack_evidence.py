"""Verify a historical evidence archive; optionally restore into a new directory."""

import argparse
import io
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def unpack(directory, output=None):
    """Check every byte before writing; never execute historical scripts."""
    directory = Path(directory)
    manifest = json.loads((directory / "raw-evidence.manifest.json").read_text())
    if manifest["schema"] != "vibeqc.evidence-archive.v1":
        raise ValueError("unsupported evidence archive schema")
    raw = (directory / "raw-evidence.zip").read_bytes()
    if sha256(raw).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("archive SHA-256 mismatch")
    records = manifest["files"]
    names = [record["path"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("duplicate manifest paths")
    payloads = {}
    with ZipFile(io.BytesIO(raw)) as archive:
        if sorted(archive.namelist()) != sorted(names):
            raise ValueError("archive members differ from manifest")
        for record in records:
            name = record["path"]
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or path.as_posix() != name
                or ".." in path.parts
                or "\\" in name
                or ":" in name
            ):
                raise ValueError(f"unsafe archive path: {name}")
            data = archive.read(name)
            if (
                len(data) != record["bytes"]
                or sha256(data).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"member identity mismatch: {name}")
            payloads[name] = data
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
    return len(payloads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="new, non-existing directory")
    args = parser.parse_args()
    count = unpack(args.directory, args.output)
    print(
        f"Verified {count} files"
        + (f"; restored to {args.output}" if args.output else "")
    )
