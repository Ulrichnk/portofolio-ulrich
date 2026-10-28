"""Leakage-aware next-day conditional variance forecasts on synthetic returns.

Return units are percentages (1.0 means 1%). Features at origin t only contain
observations through t; target is r[t+1]^2. No trading profitability is claimed.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, binomtest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


def simulate_garch(
    n: int = 6000,
    seed: int = 42,
    omega: float = 0.03,
    alpha: float = 0.08,
    beta: float = 0.89,
    burn: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary zero-mean Gaussian GARCH(1,1); returns and true variances."""
    if not isinstance(n, int) or n < 100 or not isinstance(burn, int) or burn < 0:
        raise ValueError("n>=100 and nonnegative burn required")
    if (
        not np.all(np.isfinite([omega, alpha, beta]))
        or omega <= 0
        or min(alpha, beta) < 0
        or alpha + beta >= 1
    ):
        raise ValueError("stationary positive GARCH parameters required")
    rng = np.random.default_rng(seed)
    m = n + burn
    r = np.empty(m)
    h = np.empty(m)
    h[0] = omega / (1 - alpha - beta)
    for i in range(m):
        if i:
            h[i] = omega + alpha * r[i - 1] ** 2 + beta * h[i - 1]
        r[i] = np.sqrt(h[i]) * rng.standard_normal()
    return r[burn:], h[burn:]


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    next_returns: np.ndarray
    feature_names: tuple


def features(returns: ArrayLike, window: int = 60) -> Dataset:
    """Fixed causal feature construction. Last row origin is n-2."""
    r = np.asarray(returns, float)
    if (
        not isinstance(window, int)
        or window < 20
        or r.ndim != 1
        or not np.all(np.isfinite(r))
        or len(r) < window + 2
    ):
        raise ValueError(
            "finite 1D returns, window>=20, and enough observations required"
        )
    x = []
    y = []
    origins = []
    for i in range(window - 1, len(r) - 1):
        hist = r[i - window + 1 : i + 1]
        x.append(
            [
                r[i],
                r[i] ** 2,
                r[i - 1] ** 2,
                np.mean(hist[-5:] ** 2),
                np.mean(hist[-20:] ** 2),
                np.mean(hist**2),
            ]
        )
        y.append(r[i + 1] ** 2)
        origins.append(i)
    origins = np.asarray(origins)
    return Dataset(
        np.asarray(x),
        np.asarray(y),
        origins,
        r[origins + 1],
        ("return_t", "square_t", "square_t_minus_1", "rv5", "rv20", "rv60"),
    )


def model(name, parameter):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    if name == "boosting":
        return HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=100,
            max_leaf_nodes=7,
            learning_rate=0.05,
            l2_regularization=parameter,
            early_stopping=False,
            random_state=42,
        )
    raise ValueError("unknown model")


def qlike(y: ArrayLike, prediction: ArrayLike) -> float:
    """log(v_hat)+y/v_hat, proper loss for conditional variance up to constant."""
    y, p = np.broadcast_arrays(np.asarray(y), np.asarray(prediction))
    if (
        y.size == 0
        or np.any(y < 0)
        or np.any(~np.isfinite(y))
        or np.any(p <= 0)
        or np.any(~np.isfinite(p))
    ):
        raise ValueError(
            "finite nonnegative targets and positive finite forecasts required"
        )
    return float(np.mean(np.log(p) + y / p))


def forecast(
    dataset: Dataset, test_start: int, parameter: float, name: str, block: int = 100
) -> np.ndarray:
    """Expanding-window refits; every training target is observed at origin.

    Prediction blocks do not update model coefficients until the next refit,
    but each row's lagged-return features incorporate currently observed data.
    """
    if (
        not 100 <= test_start < len(dataset.y)
        or not isinstance(block, int)
        or block < 1
    ):
        raise ValueError("invalid split or refit block")
    out = []
    for start in range(test_start, len(dataset.y), block):
        end = min(start + block, len(dataset.y))
        estimator = model(name, parameter)
        estimator.fit(dataset.x[:start], dataset.y[:start])
        # A train-only floor prevents zero variance and explosive QLIKE values.
        floor = max(1e-8, float(np.mean(dataset.y[:start])) * 0.01)
        out.extend(np.maximum(estimator.predict(dataset.x[start:end]), floor))
    return np.asarray(out)


def var_backtest(
    returns: ArrayLike, variance: ArrayLike, probability: float = 0.01
) -> dict:
    """One-sided lower-tail Gaussian VaR, exact binomial coverage and Kupiec LR.

    Exact binomial coverage assumes iid exceptions. Volatility clustering may
    violate that assumption; this is coverage evidence, not model certification.
    """
    r, v = np.broadcast_arrays(np.asarray(returns, float), np.asarray(variance, float))
    if (
        r.ndim != 1
        or not len(r)
        or np.any(v <= 0)
        or not np.all(np.isfinite(r + v))
        or not 0 < probability < 0.5
    ):
        raise ValueError("finite returns, positive variance and 0<p<.5 required")
    threshold = norm.ppf(probability) * np.sqrt(v)
    hits = r < threshold
    n = len(r)
    count = int(hits.sum())
    frequency = count / n

    def xlog(x, p):
        return 0.0 if x == 0 else x * np.log(p)

    ll0 = xlog(count, probability) + xlog(n - count, 1 - probability)
    ll1 = xlog(count, frequency) + xlog(n - count, 1 - frequency)
    lr = max(0.0, -2 * (ll0 - ll1))
    return {
        "observations": n,
        "exceptions": count,
        "exception_rate": frequency,
        "expected_rate": probability,
        "kupiec_lr": float(lr),
        "kupiec_pvalue": float(chi2.sf(lr, 1)),
        "exact_coverage_pvalue": float(binomtest(count, n, probability).pvalue),
    }


def run_research(n: int = 4000, seed: int = 42) -> tuple:
    """60/20/20 chronology; tuning on validation, expanding evaluation on test.

    Hyperparameters remain frozen during test evaluation; refits incorporate
    newly observed past outcomes. Oracle is synthetic-only, not feasible live.
    """
    if not isinstance(n, int) or n < 300:
        raise ValueError("research protocol requires at least 300 returns")
    r, h = simulate_garch(n, seed)
    data = features(r)
    train = int(0.6 * len(data.y))
    test = int(0.8 * len(data.y))
    selected = {}
    validation = {}
    for name, grid in {
        "ridge": [1.0, 10.0, 100.0],
        "boosting": [1.0, 10.0, 100.0],
    }.items():
        scores = []
        for p in grid:
            est = model(name, p)
            est.fit(data.x[:train], data.y[:train])
            pred = np.maximum(
                est.predict(data.x[train:test]), 0.01 * np.mean(data.y[:train])
            )
            scores.append(qlike(data.y[train:test], pred))
        selected[name] = grid[int(np.argmin(scores))]
        validation[name] = dict(zip(map(str, grid), scores))
    predictions = {name: forecast(data, test, p, name) for name, p in selected.items()}
    predictions["historical_20"] = data.x[test:, 4]
    predictions["constant_train"] = np.full(len(data.y) - test, np.mean(data.y[:test]))
    predictions["oracle_garch"] = h[data.origins[test:] + 1]
    metrics = {
        name: {
            "qlike": qlike(data.y[test:], p),
            "mse": float(np.mean((data.y[test:] - p) ** 2)),
            "var_1pct": var_backtest(data.next_returns[test:], p),
        }
        for name, p in predictions.items()
    }
    return (
        {
            "seed": seed,
            "n_returns": n,
            "train_rows": train,
            "validation_rows": test - train,
            "test_rows": len(data.y) - test,
            "selected": selected,
            "validation_scores": validation,
            "metrics": metrics,
        },
        data,
        predictions,
        test,
    )
