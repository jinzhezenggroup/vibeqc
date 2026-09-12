"""Generate scalar ECP Gaussian values/center derivatives for native CUDA."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import argparse

from vibeqc_compiler.integral.ecp import emit_ecp_ao_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = emit_ecp_ao_cuda()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
