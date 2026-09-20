# Decision: normalize project-managed ccache paths at the build directory

Status: implemented in PR #736; required CI remains the integration gate
Date: 2026-09-21

## Problem

Generated sources and AOT shards reside under the CMake binary directory. Absolute
paths to byte-identical generated files can prevent useful ccache reuse when the
same checkout is configured into another build directory. CI previously supplied
raw ccache launchers, bypassing project-owned normalization.

## Decision

When the selected cache executable's basename is `ccache`, construct the
project-managed launcher as `cmake -E env CCACHE_BASEDIR=<CMAKE_BINARY_DIR>
<ccache>`. Pass the launcher as a CMake list, so spaces in paths are not shell
word boundaries. Route repository-owned CI builds through
`VIBEQC_COMPILER_CACHE=ccache` instead of hard-coding raw compiler launchers.

Only project-managed launchers change. Existing nonempty
`CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_CUDA_COMPILER_LAUNCHER` values remain
authoritative. The pre-existing behavior for an empty launcher is unchanged.
Other cache executables, including sccache, retain their existing unwrapped path.
The wrapper sets its base directory for that compiler invocation; it does not
modify the user's global ccache configuration or clear any shared cache.

## Identity and scientific invariants

This affects build-cache path normalization, not scientific compatibility.
Compiler identity, source/header contents, compiler options, generated equations,
source inventories and native artifact validation remain checked. No sloppiness,
ignored header dependency, disabled compiler check or generated-math rewrite is
introduced. A changed included header must still invalidate the cached object.

The claim is bounded: an equivalent generated compile in another binary
directory can reuse cached work. This is not a guarantee that arbitrary source
worktrees, compilers, flags, debug information, platform paths or source files
outside the selected base directory produce a cache hit. Cache misses remain
valid compilations, not numerical failures.

## Rejected alternatives

Keeping raw CI launchers leaves the project normalization unused. Replacing a
caller-owned launcher would break toolchain ownership. Relaxing input or compiler
identity to manufacture hits would make stale objects possible. Hard-coding one
workspace path would fail for other build directories and installed environments.

## Independent review validation

At source head `08996bb02d5870421708477365c97b84c8efbda8`, both committed
`tests/python/test_compiler_cache_build.py` cases passed on node3. They configure
the actual project with Ninja and inspect generated compile commands, checking
normalization and preservation of an explicit caller launcher.

An additional isolated CPU compilation used the same `cmake -E env` launcher
structure, a private cache directory, two build directories containing spaces,
and identical generated C++ plus an included data header. The first compile
missed, the second directory hit the cache and produced byte-identical object
contents. Changing the included header produced another miss and a different
object. No global cache state or compiler-content validation was disabled.

This experiment demonstrates reuse and invalidation, not a complete project or
CUDA performance claim. The implementation's PR description separately reports
an eight-shard CUDA 12.9/sm_120 campaign: 0/8 hits and 38.32 s before normalization,
versus 8/8 direct hits and 2.94 s after. Those timings were not rerun by this
independent review and must not be generalized to every build or machine.

## Revisit when

The launcher precedence contract, generated-source location, supported cache
executables, debug/path semantics or CMake toolchain policy changes. Preserve
correct invalidation first; measure complete build endpoints before extending
performance claims.

## References

PR #736; `CMakeLists.txt`; `.github/workflows/ci.yml`;
`tests/python/test_compiler_cache_build.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
