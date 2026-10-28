# Technical defense preparation

This guide supports a ten-minute presentation and a fifteen-minute discussion.
It does not record a defense as already completed.

| Time | Topic | Evidence to show |
|---|---|---|
| 0-1 min | Financial question and model assumptions | European payoff and parameter conventions |
| 1-3 min | Replication and risk-neutral valuation | Self-financing identity, PDE, dividend gain process |
| 3-4 min | Software and numerical reliability | Independent package, log-tail calculation and degenerate branches |
| 4-6 min | Validation of the analytical engine | Reference cases A-H, parity, independent quadrature |
| 6-8 min | Numerical methods | Tree/PDE convergence and Monte Carlo confidence intervals |
| 8-9 min | Research extensions | Synthetic Heston recovery and hedging with costs |
| 9-10 min | Limits and next experiments | Discrete surface checks, synthetic inputs and model risk |

## Questions to work through without memorized scripts

1. Derive the density change, including the sign of lambda and the dividend
   convention. Explain why the discounted ex-dividend stock is not generally a
   martingale when q is nonzero.
2. Reconstruct the self-financing identity and identify where rebalancing costs
   enter. Explain why simply differentiating delta times spot is insufficient.
3. Derive the call expectation by completing the Gaussian square. Recover the
   put using terminal payoff parity.
4. Explain why maturity monotonicity needs assumptions about dividends and rates.
5. Explain why log-CDF evaluation and `expm1` help an OTM option. Give a situation
   where float64 still cannot represent a result accurately.
6. Explain why antithetic paths must be paired before computing standard errors.
   Describe the independent pilot for the control coefficient.
7. Separate barrier monitoring bias from Monte Carlo variance. Explain the
   Brownian-bridge survival benchmark and why nested monitoring helps.
8. State the explicit PDE sufficient conditions and why time refinement alone
   does not repair negative off-diagonal spatial coefficients.
9. Explain what near-zero synthetic Heston residuals prove, and what they do not.
   Discuss parameter scaling and the Jacobian condition number.
10. Explain why faster delta hedging reduces discretization error while costs can
    increase, and why wrong volatility is not repaired by daily rebalancing.

A useful independent exercise is to reimplement one engine without consulting
this implementation, rerun the references, and record any discrepancies and
corrections in actual new commits.
