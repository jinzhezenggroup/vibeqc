# Decision: preserve a qualified DF baseline inside a candidate campaign

Status: implemented
Date: 2026-09-17

## Problem

#404 requires interleaved baseline/candidate endpoints from the same frozen
density, retaining already delivered SSS Rys, schedules, final validation and
projection reuse. The historical `legacy` arm changes those derivative controls
and therefore is not the post-#411 baseline. Alternating separate library
processes repeats expensive initialization, while retaining two complete large
DF plans requires the sum of their resident storage.

## Decision

An unqualified architecture profile may contain one qualified `baseline`
profile. The generated selector accepts the existing `candidate` policy bit;
automatic execution retains the baseline, and candidate execution selects the
proposed mapping. Both maps are constant generated data, parsed and validated
only by the compiler. The source manifest remains part of native library and
prepared-resource identity. No new environment variable or per-class override
is introduced.

Use the existing endpoint runner to interleave both policies in one prepared
state with warm-start updates disabled. Record the complete manifest, generated
headers, source/library identities and per-arm actual work. The separate pinned
baseline library remains available for a control against changes in the
campaign build itself.

## Invariants and evidence

The baseline must be independently qualified under the same provenance gates
as an ordinary production manifest. A nested baseline, unqualified baseline,
or simultaneous qualified candidate/baseline pair is rejected. Ordinary
single-map manifests retain their selection behavior and unsupported
architectures/classes retain their bounded fallbacks. The native automatic
384/768 size-domain check is unchanged. Remove the campaign baseline when a
mapping is qualified for production; a kernel win alone cannot qualify it.

CPU C++ static assertions exercise emitted baseline/candidate selection,
preservation of already-promoted SSS, and unsupported architectures/classes.
Parser tests reject incomplete baseline evidence and unavailable mathematics.
Full numerical, runtime and endpoint qualification remains required separately.

References: #394, #404, #206; `docs/df_tuning.md`.

The first campaign is complete. Its embedded baseline was removed when the
[combined mapping was accepted](2026-09-17-combined-rys-promotion.md); the
archived campaign manifest preserves both measured arms for reproduction.
