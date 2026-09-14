"""Shared pytest process setup for CI-efficient native test helpers."""

from __future__ import annotations

import os
from pathlib import Path


# Ubuntu's ccache package installs compiler-wrapper symlinks here.  CI already
# restores ~/.cache/ccache before pytest starts, but many Python tests invoke
# ``c++`` directly through subprocess/shutil.which and therefore bypassed that
# cache.  Prepending the wrapper directory lets those exact test-local compiles
# reuse ccache without changing any generated source, compiler flags, or test
# assertions.  Developer environments without this directory are untouched.
_CCACHE_WRAPPERS = Path("/usr/lib/ccache")
if _CCACHE_WRAPPERS.is_dir():
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    wrapper = str(_CCACHE_WRAPPERS)
    if wrapper not in entries:
        os.environ["PATH"] = wrapper + (os.pathsep + path if path else "")
