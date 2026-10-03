# Decision: explicitly configure local AO maps in composite resident forces

Status: implemented (private opt-in caller; complete endpoint qualification pending)
Date: 2026-10-03

The composite force owner now accepts an explicit positive active-AO cutoff and
host cache allowance. Defaults remain dense. Its existing planner first admits
full AO and geometry scratch; only the remaining enclosing host budget can be
assigned to the shared `ResidentAoMapCache`. No assumed mean AO count reduces
any capacity. A zero or insufficient map allowance selects dense AO execution.

The cache domain includes exact basis/geometry/grid identities, actual native
resident-grid pointer/device, order-2 jets and point-tile boundaries. It is
released alongside geometry owners, rebuilt on allocation/domain changes, and
reset only for work accounting on warm calls. Source tokens remain validated
by the native snapshot owner. Both semilocal and nonlocal consumers receive
the same selected lease; an empty AO map still executes grid/weight responses,
VV10 seed handling and complete point accounting. Snapshot feature seeding and
the bounded two-pass capability fallback are unchanged.

Seventy-nine focused host routing/resource/cache tests passed, including
selected/empty maps reaching both consumers and rejection of mismatched actual
grid leases. All prek hooks passed. These establish caller plumbing only; no
new cutoff or default is scientifically promoted. Complete cold/warm/moved
energy/force measurements and constrained-budget controls remain necessary.
