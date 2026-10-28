# Assignment 01 — European Options & Black–Scholes Pricing Engine

**Level:** MSc 2 / Junior Quantitative Researcher  
**Language:** Python 3  
**Target:** v0.1.0  
**Status:** Problem statement only — no implementation or validated results yet.  
**Assessment:** 100 points. No solutions provided.

## Mission
Design a reusable analytical European option pricer under the Black–Scholes–Merton model, derive its mathematical foundations, implement it independently, and validate financial and numerical correctness. Calls and puts, constant interest rates, and constant continuous dividend yield must be supported.

## Part I — Mathematical modelling (25 points)
Assume a filtered probability space carrying Brownian motion and, initially, a non-dividend-paying stock with dynamics under the historical measure:
`dS_t = mu S_t dt + sigma S_t dW_t^P`
and riskless asset `dB_t = r B_t dt`.

**Exercise 1 — Model construction.** State hypotheses; solve the stock SDE; establish conditional law, positivity and relevant moments; contrast arithmetic and logarithmic returns; discuss limitations.

**Exercise 2 — Replication and PDE.** Apply Itô to a sufficiently smooth derivative value; construct a locally riskless self-financing hedging portfolio; derive the pricing PDE and terminal conditions; discuss completeness and uniqueness.

**Exercise 3 — Risk-neutral pricing.** State and apply the appropriate Girsanov theorem; construct an equivalent martingale measure; justify the discounted martingale property and discounted conditional expectation representation; reconcile it with replication.

**Exercise 4 — Analytical pricing.** Derive European call and put prices from the risk-neutral representation; establish put-call parity, no-arbitrage bounds, asymptotic limits and monotonicity/convexity conditions. Extend the appropriate equations to constant continuous dividend yield `q`. Explicitly treat limiting cases rather than naively evaluating singular formulas.

## Part II — Software specifications (15 points)
**Exercise 5 — Functional scope.** Represent European call/put contracts and parameters (spot, strike, expiry as year fraction, volatility, risk-free rate, continuous dividend yield). Provide analytical prices, scalar/vectorised evaluation, documented units and conventions, validation of invalid inputs, and handling of zero maturity / zero volatility.

**Exercise 6 — Architecture.** Separate instrument definitions, model parameters, pricing routines, and research notebooks. Propose interfaces that allow later replacement with trees, Monte Carlo, and PDE solvers. Provide a design diagram and decision log.

**Exercise 7 — Numerical robustness.** Address floating-point accuracy, short maturities, near-zero volatility, deep in/out-of-the-money regimes, non-finite values, and scalar/vector agreement. State numerical decisions and their consequences.

## Part III — Numerical research (25 points)
**Exercise 8 — Spot behaviour.** Produce call/put price vs spot curves; mark strike and moneyness regions; compare intrinsic value and no-arbitrage bounds; analyse monotonicity and convexity.

**Exercise 9 — Parameter sensitivity.** Separately investigate volatility, maturity, risk-free rate, dividend yield and strike; state theoretically expected properties *before* running numerical experiments; interpret outcomes with the relevant assumptions.

**Exercise 10 — Stress scenarios.** Examine deep ITM/OTM, very short maturity, near-zero volatility, high volatility, negative interest rates, and positive dividend yield; discuss numerical or financial difficulties.

**Exercise 11 — Precision and performance.** Design reproducible benchmarks for scalar vs vectorised execution, input size scaling, double precision and extreme inputs. Report environment, repeated measurements and interpretation. Do not confuse model error with numerical error.

## Part IV — Acceptance tests (20 points)
Automate and document all of the following:
- T01: put-call parity within a justified tolerance.
- T02: non-negative option values.
- T03: correct no-arbitrage lower/upper bounds.
- T04/T05: call nondecreasing / put nonincreasing in spot.
- T06: convexity in spot.
- T07: payoff at expiry.
- T08: zero-volatility deterministic limit.
- T09: agreement with independent trusted reference values.
- T10: scalar/vector agreement.
- T11: explicit failures on invalid input.
- T12: no silently incorrect behaviour in extreme scenarios.

**Mandatory scenarios:** all maturities below in years, annualised rates and vols.
| Case | S | K | r | q | sigma | T |
|---|---:|---:|---:|---:|---:|---:|
| A ATM | 100 | 100 | 0.05 | 0 | 0.20 | 1 |
| B OTM call | 80 | 100 | 0.03 | 0 | 0.25 | 0.5 |
| C ITM call | 120 | 100 | 0.02 | 0.01 | 0.30 | 2 |
| D Short expiry | 100 | 100 | 0.04 | 0 | 0.20 | 0.01 |
| E Negative rate | 100 | 100 | -0.01 | 0 | 0.20 | 1 |
| F High volatility | 100 | 100 | 0.03 | 0.01 | 0.80 | 1 |
| G At expiry | 100 | 100 | 0.03 | 0 | 0.20 | 0 |
| H Zero volatility | 100 | 100 | 0.03 | 0 | 0 | 1 |

Provide reference sources, absolute/relative deviations and **pre-declared, context-appropriate tolerances**. At-expiry and zero-volatility cases require explicit limiting logic.

## Part V — Documentation and research communication (15 points)
**Exercise 15:** Research report (10–15 pages excluding appendices): introduction; assumptions; proofs; architecture; validation; numerical results; limitations; outlook.

**Exercise 16:** English-language README: scope, installation, usage examples *once implemented*, assumptions, tests, reproducibility, roadmap and status.

**Exercise 17:** Prepare a 10-minute technical defence and a 15-minute Q&A.

## Required deliverables
- Python pricing library independent of notebooks.
- Automated acceptance tests and independent numerical benchmarks.
- Mathematical derivations (PDF or Markdown/LaTeX).
- Research notebooks and plots (each labelled with parameters and units).
- Research report, English README, architecture diagram and brief technical presentation.

## Grading and publication gate
| Criterion | Points |
|---|---:|
| Mathematical proofs and modelling | 25 |
| Python architecture and implementation | 15 |
| Numerical experiments | 25 |
| Acceptance tests and validation | 20 |
| Documentation and defence | 15 |
| **Total** | **100** |

Publish as a completed v0.1.0 only after a score of at least 85/100, all mandatory financial acceptance tests pass, no unexplained arbitrage violation remains, and the results are reproducible. A numerical or financial correctness failure is blocking independently of the total score.

## Suggested milestones
A. Mathematical proofs; B. software design; C. implementation; D. numerical validation; E. documentation and release.

**Future versions:** Greeks and implied volatility (v0.2), binomial/American (v0.3), Monte Carlo (v0.4), exotics (v0.5), PDE (v0.6), volatility surface/Heston (v0.7), hedging/model risk (v1.0). None of these are claimed as implemented here.
