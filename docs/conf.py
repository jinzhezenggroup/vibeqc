from __future__ import annotations

project = "VibeQC"
author = "VibeQC contributors"
language = "en"

extensions = ["myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "AGENTS.md",
    "superpowers/**",
]

html_theme = "furo"
html_title = "VibeQC documentation"

myst_heading_anchors = 3
