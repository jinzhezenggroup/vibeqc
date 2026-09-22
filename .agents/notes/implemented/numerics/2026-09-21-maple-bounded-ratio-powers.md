# Decision: retain bounded ratio powers in Maple lowering

Status: implemented
Date: 2026-09-21

The production r2SCAN cutover failed the native H2 RKS energy test despite
passing interior and boundary scalar fixtures. At a physical grid tail with
rho_a=rho_b=1.24561752483369938e-56, unconditional distribution of a fourth
power expanded the bounded spin ratio into numerator and denominator powers.
Its derivative formed an unrepresentable inverse eighth power even though
the original function and derivative were finite. No density floor is needed.

Distribute even powers only when an explicit square-root factor can be
cancelled through multiplication/reciprocal nodes. Otherwise preserve the
bounded expression before differentiation. This retains cancellation-free
sigma derivatives at zero while avoiding manufactured low-density overflow.
The importer semantic identity changes deliberately; equations and numerical
acceptance thresholds are unchanged.

Validation: four of nine new ratio/root regression cases fail before the fix;
all 198 selected Maple/SCAN tests pass after it. A new source-matched Release
CPU build first reproduced the H2 failure, then passed the original DFT API
test and all 47 native CTests after the repair. This is not a new GPU or
performance qualification. The stacked PR still requires master integration.

Agent: ChatGPT
Model: GPT-5.6 Sol
