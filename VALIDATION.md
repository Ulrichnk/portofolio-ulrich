# Validation record 

This records executed development checks, not an independent academic grade or
production certification. Development assistance is described in CONTRIBUTIONS.md.

## Executed checks

- 33 unittest cases pass locally, including independent pricing references,
  financial properties, tree/PDE convergence, Monte Carlo and exotic checks,
  synthetic Heston calibration, rates repricing/risk, causal features and forecasts.
- The complete experiment script was executed and produced the committed raw
  results. No credentials or external data downloads are used by the experiments.
- All three notebooks were executed cell by cell by `experiments/build_notebooks.py`.
- A fresh virtual environment installed the package and its dependencies successfully.
  All 33 tests and the full experiment protocol then passed in that environment.
  Runtime dependency versions are pinned in requirements-tested.txt.
- Package installation and bytecode compilation succeeded.
- Python sources were formatted and checked with Black.
- Three A4 PDF reports were compiled with Pandoc/XeLaTeX, rendered with Poppler
  and visually inspected. The final reports contain 13, 5 and 6 pages respectively.

The original MSc assignment's 10-15-page core report requirement is met by the
11-page derivatives body plus two pages of graphical appendices. The oral defense
is supported by a timed preparation guide, not recorded as having taken place.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/reproduce.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/build_notebooks.py
python experiments/build_reports.py
```

The report build additionally requires `pandoc`, `xelatex`, the DejaVu fonts and
LaTeX packages including `float`. These are documentation tools, not library
runtime dependencies. Notebooks may also be opened with the optional Jupyter
installation described in the README.

## Limits of the evidence

Tests of probabilistic methods on fixed seeds do not establish universal
confidence coverage. The coverage experiment uses 200 independent replications.
All Heston quotes, rates quotes and return series are synthetic. The surface
checks do not certify a continuous arbitrage-free interpolator. The library has
not been validated on live market data or independently reviewed by a human quant.

A GitHub Actions workflow is included. Its remote execution status must be checked
on GitHub separately; local success alone is not reported as a green remote run.
