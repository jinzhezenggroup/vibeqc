# Generated ECP AO and fixed-weight consumers

This qualifies the replacement of the CUDA AO primitive/component and
fixed-weight derivative contractions with compiler-owned lowering. The supported
scalar ECP domain, quadrature, reduction order, FP64 storage and complete direct
RHF/UHF contract are unchanged. Refs #171; this does not close the issue.

## Source and build identity

The baseline is the exact `git archive` snapshot of master
`1c3f2ab8d6df6f06bee526b89a807511ef726301`. The candidate is that snapshot
**modified by the three files in the archived `source-overlay.tar.gz`**. These
are exported snapshots, not Git checkouts; no clean Git state is inferred.
All 2,683 source files were checked against baseline/overlay hashes after each
qualification. The archive retains complete per-file identities, exact overlay
bytes, build-cache/header/library hashes and reproduction scripts. The summary
also records LF-normalized hashes matching the repository's scientific sources.
The measured CUDA adapter has CRLF line endings; normalization changes no code.

RTX 4090, CUDA 12.9.86, sm_89, Release, AOT shells disabled; one BLAS/OpenMP
thread. Baseline and candidate use the same isolated CMake build directory and
flags. Explicit header regeneration and fresh native input timestamps prevent
archive mtimes from leaving baseline objects in an incremental candidate build.
The first stale-header failure and the initially missing sanitizer PATH entry
are retained in the raw logs. All final gates below ran against the candidate.

## Results

- CPU native tests: **30/30**. CPU ECP/IR/input Python: **43 passed, 10 GPU skips**.
- CUDA native ECP tests: **2/2**. CUDA ECP/IR/input Python: **53/53**.
- CUDA 12.8 Compute Sanitizer memcheck of the CUDA 12.9 error/recovery executable:
  **0 errors**.
- Independent native tests cover contracted s/p/d values, signed primitive and
  component mixtures, primitive offsets, coincident nodes, value-only slots,
  two finite-difference steps and nonsymmetric full AO weighted force signs.
- Libcint matrix maximum error: **3.47e-12 Eh**. Complete RHF/UHF force errors:
  **5.54e-13 / 7.98e-16 Eh/bohr**. CPU/GPU raw component and derivative errors
  are at most **1.39e-17** on the matched benchmark fixture.

| Complete synchronous endpoint | Baseline median / ms | Candidate median / ms | Planned device peak / bytes |
| --- | ---: | ---: | ---: |
| RHF NaH | 1007.27 | 1005.77 | 2,343,376 |
| UHF NaH+ | 1002.14 | 999.90 | 2,378,528 |

Three warm samples per endpoint are retained, with all component timings. All
planned host/device peaks match. These small sequential measurements are a
regression check, not a statistical speedup or schedule-promotion claim.

Native adapter code decreases from 287 to 272 nonblank/noncomment lines.
The AO kernel uses 56 registers / 64 stack bytes versus 48 / 176; the weighted
consumer uses 38 registers versus 40, with zero stack bytes. Projector,
pair-contraction and convergence kernels retain their resource counts. The
generated header grows from 492 to 545 lines. Host normalization metadata,
grid/harmonic construction and convergence policy remain explicit; the adapter
keeps its conservative scientific classification and the independent CPU oracle
is retained. DFT/ECP and broader angular/element capabilities remain open.

## Reproduction

Verify or restore the byte-checked evidence:

```sh
python -m tools.unpack_evidence benchmarks/results/ecp-consumers-171 --output /tmp/ecp-consumers
```

Export the stated baseline, overlay `source-overlay.tar.gz` for the candidate,
and configure CPU/CUDA Release builds as described in `docs/ecp.md`. The retained
`consumer-*.sh` files contain the exact build/test commands; adapt their workspace
paths and GPU architecture. `consumer-provenance.py` verifies every source byte.
The CPU tests, CUDA tests, full endpoints and sanitizer commands must all pass
before any further native scientific body is retired.

The accompanying evidence CLI fix treats Git member names as POSIX paths even
on Windows. Both valid and corrupted publication cases pass its simulated
Windows regression test, and the staged repository passes the retention CLI.
A broader local retention test still encounters the unchanged XC historical
fixture manifest's CRLF checkout mismatch; its HEAD blobs match every expected
hash. This does not affect the Linux ECP qualification or this byte-checked bundle.
