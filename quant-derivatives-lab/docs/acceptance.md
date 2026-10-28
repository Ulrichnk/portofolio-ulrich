# Acceptance evidence

Run `python -m unittest discover -s tests -v` from the installed repository.
The original exercise is preserved. This map connects its acceptance tests to
executable checks, rather than assigning a self-certified academic grade.

| Assignment ID | Evidence in `tests/test_black_scholes.py` |
|---|---|
| T01 Put-call parity | `test_properties`, 500 seeded random parameter vectors, absolute 2e-12 and relative 1e-12 tolerance |
| T02 Nonnegativity | `test_properties`, lower bounds enforce nonnegativity |
| T03 Arbitrage bounds | `test_properties`, price between analytical bounds within 1e-11 |
| T04 Call monotonicity | `test_properties`, 1,000 ordered spots, difference >= -1e-12 |
| T05 Put monotonicity | `test_properties`, difference <= 1e-12 |
| T06 Convexity | `test_properties`, uniform spot grid, second differences >= -1e-11 |
| T07 Expiry | `test_degenerate`, exact payoff at zero and convergence at 1e-12 years |
| T08 Zero volatility | `test_degenerate`, discounted deterministic payoff, both directions |
| T09 Independent benchmarks | `test_reference`, known reference values to 11 places; `test_independent_quadrature`, normal-payoff integration to 9 places |
| T10 Vectorization | `test_broadcast`, scalar and broadcast grid agree |
| T11 Invalid inputs | `test_inputs_and_extremes`, NaN, infinity, negative domain, incompatible shape, unknown direction |
| T12 Extremes | `test_inputs_and_extremes`, finite results across extreme spots, explicit overflow exception |

Tests of advanced engines cover analytical-versus-finite-difference Greeks,
implied-volatility round trips, CRR convergence and American dominance, Monte
Carlo uncertainty and variance reduction, independent exotic references, PDE
convergence and rejection of unstable explicit grids, Heston limits/calibration,
discrete surface diagnostics and hedge-error/cost behavior.

Analytical tolerances are not reused for stochastic or discretized methods.
Stochastic checks use conservative multiples of standard error on fixed seeds.
The CI coverage experiment is an additional replicated diagnostic, not a promise
that a particular run must contain the analytical price.

The grid of Heston quotes and synthetic calibration is implemented. A continuous
arbitrage-free interpolated volatility surface, real-market calibration and an
independent human review are outside this delivery. The original roadmap should
not be read as a claim that these broader activities have been completed.
