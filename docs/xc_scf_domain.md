# Semilocal SCF point domain and spin boundary

The native SCF point evaluator uses the identity
`semilocal-scaled-v1/pbe-spin-c2-1e-18`. Its compositions are exactly
`LDA_X + LDA_C_PW` and `GGA_X_PBE + GGA_C_PBE`; PBE correlation uses modified
PW constants. The source parameters and conventions follow the audited
Libxc 7.0.0 sources in `external/libxc-7.0.0` and
`python/vibeqc_compiler/xc/expressions.py`. There is no exact exchange or
density fitting in these method compositions.

This replaces the provisional SCF PBE-to-LDA tail fallback. The independent
`interior-v1` Python reference consumer and its #214 fixtures keep their
original domain and identities. The native `integrate_pbe_rks` diagnostic
entry also retains its interior-domain guard; `integrate_pbe_rks_with_tail`
now evaluates the versioned domain below. Its old function name does not
authorize reuse of results produced with the old tail identity.

## Density and variational convention

UKS has independent symmetric real matrices `Da` and `Db`, with unit spin
occupations. RKS stores `D=Da+Db` and has `Da=Db=D/2`. At fixed geometry:

```
E = Enuc + Tr((Da+Db) h) + 1/2 Tr((Da+Db) J[Da+Db]) + Exc
Fa = h + J[Da+Db] + Vxc,a
Fb = h + J[Da+Db] + Vxc,b
dExc = sum_s Tr(Vxc,s^T dDs)
```

Every full matrix entry contributes to the trace. A symmetric off-diagonal
perturbation changes two entries and therefore contributes twice. Applying
the HF half-trace energy formula to these Fock matrices would double count
XC incorrectly. Any later exact-exchange term must be declared in the common
Fock strategy and carry its quadratic energy factor separately.

The point evaluator returns `e`, `de/drho_s` and `de/dgrad(rho_s)`. The latter
is exactly `2 e_sigma_ss grad(rho_s) + e_sigma_ab grad(rho_other)`, with the
existing convention `sigma_ab=grad(rho_a).grad(rho_b)`. Assembling it against
`grad(phi_mu) phi_nu + phi_mu grad(phi_nu)` is the same first-derivative AO
contraction used by the independent XC integrator. Each quadrature weight
is applied once. LDA requests only AO values; PBE requests values and three
first spatial derivatives. Neither computes tau or higher jets.

## Stable positive-density algebra

The evaluator does not clip rho or sigma and has no positive density floor.
It rejects negative/nonfinite densities, nonfinite gradients, a nonzero
gradient in an exactly empty spin, and any unrepresentable output. Exact
vacuum has zero energy and potential coefficients. Positive-density
underflow of a final energy follows ordinary FP64 arithmetic; density
derivatives are evaluated independently and are retained when representable.

For correlation, choose the fixed numerical density scale `N=rho_a+rho_b`
and differentiate in `a=rho_a/N`, `b=rho_b/N`. A separate fixed scale `M>=N`
keeps the normalized total-gradient components
`h=(grad(rho_a)+grad(rho_b))/M` bounded. Sum finite components before
normalizing to retain cancellation; if the sum exceeds FP64, normalize the
finite summands instead. If `e=N f(a,b,h)`, physical density derivatives
are partials of `f`, and each spin's gradient derivative is `N/M` times the
corresponding `h` partial. Both scales are held constant when differentiating.
The tiny energy is never divided by N to recover a potential after underflow.

PW92 uses `x=rho^(1/6)` and its original polynomial rewritten as

```
q = b1 sqrt(c) x^3 + b2 c x^2 + b3 c^(3/2) x + b4 c^2
u = x^4/(2 A q),   c = (3/(4 pi))^(1/3)
epsilon_PW = -(x^2 + alpha c) x^2/q * log1p(u)/u
```

The analytic continuation of `log1p(u)/u` is evaluated with a polynomial
through fifth order for `|u|<1e-4`, including its derivative. The first
omitted value term is below `1.5e-25` at the switch. This rewrite avoids
inverse-density powers, without replacing the physical tail expression.

Spin exchange evaluates its enhancement and analytic first derivatives using
`u=|grad(rho_s)|/rho_s^(4/3)` when `u<=1`, and its reciprocal when `u>1`.
The two bounded rational forms are algebraically identical. Computing the
reciprocal as `(rho_s/max_abs_grad)*(cbrt(rho_s)/scaled_grad_norm)` avoids
squaring quantities that both underflow for an extremely small minority
spin. The density derivative uses `cbrt(rho_s)` independently of any
underflow in the exchange energy. This introduces no density floor or new
boundary prescription.

In the large-gradient tail, PBE correlation combines `epsilon_PW+H`
*before* evaluation. With `G=gamma phi^3`, `u=A_PBE t^2` and `v=1/(1+u)`,

```
epsilon_c = G log1p(expm1(epsilon_PW/G) v^2/(1-v+v^2))
```

This is algebraically the original PBE expression. It avoids both `u^2`
overflow and cancellation between PW and H. Differentiating that
cancellation numerically would otherwise generate incorrect tail
potentials even when the total energy appeared accurate.

Write `t^2=|h|^2/d`, where `d` contains `(N/M)^2 N^(1/3)` and the usual
density/spin constants. Evaluate `v=d/(d+A_PBE |h|^2)` directly: neither
`grad(rho)/N`, `t^2` nor `A_PBE t^2` can overflow as an intermediate. For
`v>=1/2`, the equivalent original `epsilon_PW+H` expression has no severe
cancellation and avoids losing PW when `1-exp(epsilon_PW/G)` rounds to one.
The two forms and all first derivatives join continuously. Exactly zero
total gradient returns PW; nonzero components whose squared norm underflows
still retain their first gradient derivatives.

## Explicit PBE spin endpoint extension

PBE's mathematical interpolation `u^(2/3)` in
`phi=((2rho_a/rho)^(2/3)+(2rho_b/rho)^(2/3))/2` has a divergent first density
derivative at exactly empty spin with nonzero total gradient. A finite
minority-spin potential therefore needs an explicit extension. This version
uses the following C2 prescription **only in phi**, with `delta=1e-18`:

```
p(u) = u^(2/3)                                      u >= delta
p(u) = delta^(2/3) (14t - 7t^2 + 2t^3)/9, t=u/delta  0 <= u < delta
```

Value, first derivative and second derivative match at delta, and the exact
empty-spin value is preserved. The finite endpoint derivative is part of
the model identity; it is not the divergent derivative of unregularized
PBE. Energy and potential use the same extension, preserving the discrete
variational identity across its connection. LDA and PBE exchange retain
their analytic zero-spin first derivatives. No later meta-GGA, response or
force capability is inferred from this first-derivative implementation.

The CPU RKS CPKS consumer explicitly differentiates these same scaled formulas
in a total-density/Cartesian-gradient direction. Correlation uses a directional
derivative of the existing point jet; exchange differentiates the shared
potential formulas. Both numerical scales stay fixed through differentiation.
The zero-total-gradient branch retains the PBE second derivative even though
its first gradient coefficient is zero. The UKS consumer seeds
independent spin directions through the same correlation jet and exchange
potentials. At an empty spin, only zero density/gradient directions are
admissible; the other spin may vary. No finite full spin-endpoint Hessian is
claimed. At positive-density zero-gradient points where rho^(4/3) underflows,
the exchange reduced-gradient direction uses (gradient/rho)/cbrt(rho), retaining
finite directional coefficients without constructing an infinite Hessian.
Exact vacuum requires a zero direction, and nonrepresentable directional
coefficients reject the action. The interior diagnostic domain is unchanged.

Libxc has its own low-density and spin-boundary screening conventions.
Therefore exact endpoint comparisons use the independently differentiated
high precision formula with the declared extension. Ordinary positive-spin
points use Libxc directly. Molecular comparisons record any remaining
boundary-policy differences instead of silently modifying reference inputs.

## Executable evidence

`tests/data/xc/scf_domain.tsv` contains 97 energy/potential points: Libxc 7
interior points and independent 450-digit mpmath evaluations of the original
unscaled equations. `tools/generate_xc_scf_references.py` regenerates them.
`vibeqc_xc_point_tests` checks each coefficient with a relative tolerance,
including total densities down to `1e-300`, minority densities down to the
smallest positive FP64 value, empty spin channels, and both sides of the
C2 connection and exchange/correlation numerical branches. It also covers
finite gradients up to `1e308`, cancellation of huge opposite spin gradients,
and underflowing gradient squares; a loose absolute energy-only gate cannot
pass these tests.

`tests/data/xc/rks_response.tsv` contains 30 restricted point directions,
generated by `tools/generate_xc_rks_response_references.py` from 450-digit mixed
derivatives of the original unscaled PW92/PBE energy. The native
`vibeqc_rks_response_tests` target calls the private library ABI directly and
covers densities from `1e-280` to `1e10`, ordinary zero gradient, and both sides
of the exchange direct/reciprocal branch. Each component uses
`3e-11*abs(reference) + 32*epsilon*(abs(delta_X)+abs(delta_C)) + 8*denorm_min`,
where the exchange/correlation magnitudes are also independent high-precision
references. This admits FP64 roundoff when PBE gradient terms cancel while
retaining relative acceptance for tiny nonzero tail responses. The same target
checks ABI layout/domain errors, exact zero directions, and native snapshot
energy equality and revocation after same-geometry replay.

`tests/data/xc/uks_response.tsv` and
`tools/generate_xc_uks_response_references.py` provide 48 independent spin
directions from the original 450-digit energy oracle. The native
`vibeqc_uks_response_tests` target covers both spin potentials, densities down
to `1e-280`, empty/near-empty spin fractions, both sides of the PBE C2 spin
connection and exchange/correlation branches, opposing gradients, and zero
gradients whose rho^(4/3) underflows. Empty-spin energy derivatives are
one-sided with extra working precision; the response direction preserves the
empty spin. The per-component gate is `3e-10*abs(reference)` plus 64 machine
epsilons times independent `abs(delta_X)+abs(delta_C)` and 8 minimum subnormals.
The target also verifies spin permutation, batched layout and domain errors.

`vibeqc_dft_tests` retains the #214 identical-grid oracle and adds unequal
spin directional tests, isolated symmetric off-diagonal perturbations,
spin swaps, and deliberate half/double-factor failures. These tests precede
SCF acceptance. The CPU slice's empty/near-empty-spin continuity and
returned-state regressions remain enabled. Native single-system CUDA SCF and
replay tests are documented separately in `xc_native_cuda.md`; prepared ragged
batching and full resource evidence remain part of the complete #162 milestone.

If CPU UKS energy and the unshifted physical commutator pass their gates while
the proposed density still changes, subsequent orbital updates use a
`0.1 Eh` virtual-space level shift. For each unit-occupation spin density,
the proposal operator gains `0.1 (S-SDS)`. This stabilizes the alternating
symmetry-related pi occupations in linear OH, whose physical frontier
splitting is small but nonzero. Energy, residual and returned density still
come from the unshifted operator; convergence requires a later iteration to
pass the original density-change gate as well. No fractional occupations or
relaxed tolerances are used. This is a convergence aid, not an SCF stability
analysis or a guarantee of the global minimum. The OH returned-state tests
also reconstruct spin traces and verify `DSD=D`.
