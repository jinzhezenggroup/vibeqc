"""Draw the README's warm WB97M-V comparison from retained endpoint reports.

The figure describes the measured explicitly enabled integration, not the master
default. Reuse the evidence verifier before plotting so incomplete or inaccurate
pairs cannot silently become benchmark points. No GPU or native library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import statistics
import subprocess
import sys
from pathlib import Path

from tools.render_readme_benchmarks import COLORS, plt, save_svg, style_axes


def checked_reports(directory: Path) -> tuple[dict, list[dict]]:
    """Check all retained controls and bind plotted reports to their manifest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    reports = []
    for atoms in sorted(map(int, manifest["comparison_modes"])):
        subprocess.run(
            [sys.executable, str(directory / "verify.py"), str(atoms)], check=True
        )
        path = directory / f"matched{atoms}-sparse.json.xz"
        stored = path.read_bytes()
        payload = lzma.decompress(stored)
        identity = manifest["files"][path.name]
        if (
            hashlib.sha256(stored).hexdigest() != identity["sha256"]
            or hashlib.sha256(payload).hexdigest() != identity["decoded_sha256"]
        ):
            raise ValueError(f"report changed after verification: {path}")
        report = json.loads(payload)
        for engine in ("native", "reference"):
            values = [row["seconds"] for row in report[f"{engine}_samples"]]
            values.append(report[f"{engine}_complete_cold"]["seconds"])
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"invalid endpoint time: {path}, {engine}")
        reports.append(report)
    return manifest, sorted(reports, key=lambda row: row["aos"])


def draw(directory: Path) -> None:
    """Match the HF README figure: warm medians/ranges on the full AO axis."""
    _, reports = checked_reports(directory)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "generativeqc-wb97mv-active-ao",
        }
    )
    aos = [row["aos"] for row in reports]
    ao_ticks = [24, 48, 96, 192, 384, 768]
    if aos != ao_ticks:
        raise ValueError("the HF-matched WB97M-V curve requires all six AO sizes")
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    for key, engine, label, style in (
        ("native", "GenerativeQC", "GenerativeQC", "o-"),
        ("reference", "GPU4PySCF", "GPU4PySCF 1.8.1", "s--"),
    ):
        values = [
            [1000 * sample["seconds"] for sample in row[f"{key}_samples"]]
            for row in reports
        ]
        medians = [statistics.median(samples) for samples in values]
        ax.errorbar(
            aos,
            medians,
            yerr=[
                [m - min(s) for m, s in zip(medians, values, strict=True)],
                [max(s) - m for m, s in zip(medians, values, strict=True)],
            ],
            fmt=style,
            color=COLORS[engine],
            label=label,
            capsize=3,
            linewidth=1.8,
            markersize=4,
        )
    style_axes(
        ax,
        "ωB97M-V · spherical def2-SVP · RTX 5090",
        ao_ticks,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Warm energy + analytic forces · median and min–max of 3 fixed-density "
        "runs · one SCF iteration each",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_svg(fig, directory / "wb97mv.svg")
    plt.close(fig)


def main() -> None:
    """Regenerate the tracked SVG from the hash-bound benchmark evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("benchmarks/results/wb97mv-active-ao-20261003"),
    )
    draw(parser.parse_args().directory)


if __name__ == "__main__":
    main()
