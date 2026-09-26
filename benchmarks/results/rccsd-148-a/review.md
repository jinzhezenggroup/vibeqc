# Independent read-only reviews

Both reviewers were spawned after implementation and qz validation completed.
Neither participated in implementation. They reviewed the same 16-file
`source-snapshot.json` and verified every SHA-256. No reviewed source changed
afterward; this review record and the validation report are delivery metadata.

## Mathematical review (`review_math`)

No blocking findings. Confirmed spatial/spin-orbital normalization, restricted
pair symmetry, single-alpha residual projection, all 5 energy and 30 singles
terms, full off-diagonal Fock dependence, and `R1=D*(update-t1)`. Verified that
the packed coordinate metric and physical excitation-state overlap are
distinct and explicitly documented. The determinant oracle shares no
inventory contractions; polynomial groups and actual-DAG ablations provide
independent diagnostics.

Read-only local recheck: CC equations and fixed references, cache/bytecode
disabled, 20 passed and 3 native-library skips. The reviewer additionally ran
9 cases: `(o,v)=(3,2),(3,1),(1,3)`, three seeds each, amplitudes multiplied by
two. Maximum energy error was 1.39e-16 Eh and maximum R1 error 6.66e-16.
The reviewer did not claim to rerun remote/native tests.

## Architecture review (`review_architecture`)

No blocking P0/P1/P2 findings. Confirmed reuse of TensorIR, PackedLayout and
#147 interfaces, absence of production PySCF/oracle imports, reference identity
checks, existing FP64/shape/finiteness/symmetry validation, honest independent
budget scopes and clear A/B/C boundaries. Read the corresponding archived
204-pass Python, 10/10 CTest, Ruff, pip-check and source-verification logs.
No DFT/XC changes or duplicate public CC capability were found.

Nonblocking follow-up: `evaluate` obtains the five provider blocks before
TensorIR validates amplitudes and its logical-byte budget. Invalid inputs can
therefore incur unnecessary transformations. Provider allocations still obey
their own preflight budget. This does not alter the mathematical result or
the documented aggregate-memory limitations; prevalidate at solver entry
when slice C introduces repeated user/solver state handling.

## Main-agent disposition

Both reports were checked against the implementation and the already verified
qz archive. No mathematical correction or tolerance change was required.
The early-input optimization is explicitly deferred, not reported as fixed.
The reviewed code remains the tested snapshot. T2, full residual/optimizer
equivalence, CPU iteration and converged HF→CCSD endpoints remain B/C work.
