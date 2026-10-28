# Quantitative Finance Research Portfolio

Three reproducible educational Python projects covering derivatives, interest
rates and financial risk. Implementation and experiments added on **2026-10-28**.
See [development provenance](CONTRIBUTIONS.md) for the scope of AI assistance.
The repository's existing website and assignment are preserved.

| Project | Implemented scope | Evidence |
|---|---|---|
| [Quant Derivatives Lab](quant-derivatives-lab/) | BSM, Greeks, implied volatility, CRR, Monte Carlo, Asian/barrier payoffs, PDE, synthetic Heston calibration, delta hedging | [Report](quant-derivatives-lab/docs/research_report.pdf), independent integration, convergence studies, raw results |
| [Interest Rate Analytics](interest-rate-analytics/) | Single-curve bootstrap, deposits, bonds, swaps, PV01, duration and convexity | [Report](interest-rate-analytics/docs/research_report.pdf), exact quote repricing, analytical risk checks |
| [Machine Learning for Financial Risk](financial-risk-ml/) | Causal features, Ridge/boosting, chronological selection, expanding refits, variance and VaR backtests | [Report](financial-risk-ml/docs/research_report.pdf), leakage tests, held-out forecasts, multiple seeds |

All quotes and return series used in the numerical experiments are **synthetic**.
These are research implementations, with explicitly limited conventions, rather
than production pricing or regulatory risk systems. The Heston deliverable is a
quote grid with diagnostics and synthetic calibration; it does not implement a
continuous arbitrage-free surface or a calibration to live market quotes.

## Install and reproduce

Python 3.11 or later, in a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . -c requirements-tested.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/reproduce.py
```

The environment used to generate the committed evidence is recorded in
[environment.json](experiments/environment.json). Numerical results should agree
up to floating-point/library differences. Timings depend on the machine and load.
The experiments run without downloading data or requiring credentials.

For notebooks, install Jupyter separately with `pip install -e '.[notebooks]'`.
Notebooks call the same reusable library as the tests; they are not required for
installation or pricing.

## Minimal use

```python
from quantlab.black_scholes import EuropeanOption, Market, implied_vol
from quantlab.rates import DiscountCurve, bond_pv

market = Market(spot=100, volatility=0.2, rate=0.05)
option = EuropeanOption(strike=100, maturity=1, kind="call")
print(market.price(option))  # about 10.4505835722
```

## Repository map

- `src/quantlab/`: reusable engines, no notebook or plotting dependency.
- `tests/`: deterministic references, financial properties and statistical checks.
- `experiments/reproduce.py`: complete seeded experimental protocol.
- Each project folder: English README, French research report, notebook and raw results.
- `quant-derivatives-lab/assignments/`: original MSc assignment, retained as written.
- `portofolio-ulrich/`: pre-existing website, unchanged by this delivery.

The reports distinguish implemented features from limitations and proposed
extensions. No historical completion dates, professional usage, trading profits
or personal authorship claims are inferred from the presence of code.
