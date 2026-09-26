#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate provider-free lazy-dlopen CUDA trampolines from curated symbols.

The emitted C/assembly sources use the vendored MIT-licensed Implib.so
architecture templates. They define the CUDA host symbols referenced by
libvibeqc while resolving the real provider SONAME only on first use, so the
wheel build needs nvcc and headers but no CUDA provider shared libraries.
"""

from __future__ import annotations

import argparse
import configparser
import re
import string
from pathlib import Path

_SUPPORTED_TARGETS = ("x86_64", "aarch64")


def _target_dir(target: str) -> str:
    candidate = target.split("-")[0]
    if candidate in {"amd64", "x64"}:
        candidate = "x86_64"
    elif candidate == "arm64":
        candidate = "aarch64"
    if candidate not in _SUPPORTED_TARGETS:
        raise SystemExit(f"unsupported CUDA trampoline target: {target}")
    return candidate


def _pointer_size(path: Path) -> int:
    parser = configparser.ConfigParser()
    parser.read(path / "config.ini")
    return int(parser["Arch"]["PointerSize"])


def generate(
    base_name: str,
    symbols: list[str],
    load_name: str,
    target: str,
    implib_root: Path,
    outdir: Path,
) -> None:
    arch = _target_dir(target)
    arch_dir = implib_root / "arch" / arch
    ptr_size = _pointer_size(arch_dir)
    lib_suffix = re.sub(r"[^a-zA-Z_0-9]+", "_", base_name)
    outdir.mkdir(parents=True, exist_ok=True)

    table = string.Template((arch_dir / "table.S.tpl").read_text())
    trampoline = string.Template((arch_dir / "trampoline.S.tpl").read_text())
    with (outdir / f"{base_name}.tramp.S").open("w") as handle:
        handle.write(
            table.substitute(
                lib_suffix=lib_suffix, table_size=ptr_size * (len(symbols) + 1)
            )
        )
        for index, symbol in enumerate(symbols):
            handle.write(
                trampoline.substitute(
                    lib_suffix=lib_suffix,
                    sym=symbol,
                    offset=index * ptr_size,
                    number=index,
                )
            )

    init = string.Template((implib_root / "arch" / "common" / "init.c.tpl").read_text())
    sym_names = ",\n  ".join(f'"{symbol}"' for symbol in symbols)
    (outdir / f"{base_name}.init.c").write_text(
        init.substitute(
            lib_suffix=lib_suffix,
            load_name=load_name,
            dlopen_callback="",
            dlsym_callback="",
            has_dlopen_callback=0,
            has_dlsym_callback=0,
            no_dlopen=0,
            lazy_load=1,
            sym_names=sym_names + ",",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--symbol-list", required=True)
    parser.add_argument("--load-name", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--implib-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    symbols = [
        line.split("#", 1)[0].strip()
        for line in Path(args.symbol_list).read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not symbols:
        raise SystemExit(f"empty CUDA symbol list: {args.symbol_list}")
    generate(
        args.base_name,
        symbols,
        args.load_name,
        args.target,
        args.implib_root,
        args.outdir,
    )


if __name__ == "__main__":
    main()
