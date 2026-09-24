from __future__ import annotations

project = "VibeQC"
author = "VibeQC contributors"
language = "en"

extensions = [
    "myst_parser",
    "sphinx_book_theme",
]

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

html_theme = "sphinx_book_theme"
html_title = "VibeQC documentation"

myst_heading_anchors = 3
