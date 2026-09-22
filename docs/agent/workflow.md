# Agent task workflow

Before editing:

1. read root and scoped `AGENTS.md` files;
2. identify the public/scientific endpoint affected;
3. find existing tests, validation gates, benchmarks, and Agent Notes;
4. check whether a local change alters algorithmic work, precision, ownership, or fallback semantics.

Before opening a PR, run relevant tests/validation, record independent numerical evidence where required, and use complete endpoint measurements for performance claims.

If a non-trivial diagnosis or decision will matter again, preserve it under `.agents/notes/`.

Release/tag/publishing authority remains governed by repository-root instructions.
