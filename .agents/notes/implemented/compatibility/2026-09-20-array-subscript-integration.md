# Decision: preserve explicit slicing while adding array subscripts

Status: implemented
Date: 2026-09-20

The existing preview namespace uses `slice(x, ((start, stop), ...))` with strict
static integer ranges. The SCF frontend branch adds ordinary `x[start:stop]`
syntax. These are complementary construction interfaces, not replacement
scientific operations. Keep the namespace contract and its existing failures;
route Python subscripts through a separate private adapter to the same TensorIR
slice node. Require a genuine integer unit step, not equality-to-one coercion.

Retain the earlier reshape/broadcast missing-metadata error contract and all
explicit Index-domain checks. The newly added SCF density frontend delegates to
the shared TensorIR equations and input specifications; it does not introduce
SCF iteration, dynamic array conversion or implicit population inference.

The integrated tests include both branches' logical-identity, numerical replay,
repeated-index AD and rejection checks. Two new Boolean/float unit-step cases
fail on the original subscript adapter and pass after strict admission. The
final integrated frontend/SCF/AD/structure run passes 104 tests without skips.
No Array API conformance, device performance or production SCF cutover is claimed.

Agent: ChatGPT
Model: GPT-6 Astra Pro
