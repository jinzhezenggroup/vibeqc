from __future__ import annotations

project = "VibeQC"
author = "VibeQC contributors"
language = "en"

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
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

html_theme = "sphinx_rtd_theme"
html_title = "VibeQC documentation"
html_theme_options = {
    # Keep the audience-oriented guide tree visible instead of showing only
    # the currently selected branch.
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": True,
}

myst_heading_anchors = 3
