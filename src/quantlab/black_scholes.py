"""Vectorized European pricing with continuous rates and dividend yields.

Time is years; sigma and rates are decimal annual quantities. Prices are per
unit of underlying. Greeks are derivatives per unit parameter, not per 1%.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, log_ndtr
from scipy.optimize import brentq


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    maturity: float
    kind: str = "call"

    def __post_init__(self):
        _kind(self.kind)
        if not np.isfinite(self.strike) or self.strike <= 0:
            raise ValueError("strike must be positive and finite")
        if not np.isfinite(self.maturity) or self.maturity < 0:
            raise ValueError("maturity must be nonnegative and finite")


@dataclass(frozen=True)
class Market:
    spot: float
    volatility: float
    rate: float = 0.0
    dividend: float = 0.0

    def __post_init__(self):
        values = inputs(self.spot, 1.0, 1.0, self.volatility, self.rate, self.dividend)
        if any(x.ndim for x in values):
            raise ValueError("Market fields are scalar; use price for broadcasting")

    def price(self, option: EuropeanOption):
        return price(
            self.spot,
            option.strike,
            option.maturity,
            self.volatility,
            self.rate,
            self.dividend,
            option.kind,
        )


def _kind(kind):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def inputs(spot, strike, tau, sigma, rate, dividend):
    """Validate and broadcast; reject nonfinite or economically invalid inputs."""
    try:
        a = np.broadcast_arrays(
            *[
                np.asarray(x, dtype=float)
                for x in (spot, strike, tau, sigma, rate, dividend)
            ]
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("parameters must be numeric and broadcast-compatible") from exc
    if any(np.any(~np.isfinite(x)) for x in a):
        raise ValueError("all parameters must be finite")
    s, k, t, v, r, q = a
    if np.any(s < 0) or np.any(k <= 0) or np.any(t < 0) or np.any(v < 0):
        raise ValueError("require spot>=0, strike>0, tau>=0, sigma>=0")
    return a


def _return(x):
    return float(x) if np.ndim(x) == 0 else x


def bounds(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend: ArrayLike = 0.0,
    kind: str = "call",
) -> tuple:
    """European no-arbitrage bounds, including negative rates."""
    _kind(kind)
    s, k, t, _, r, q = inputs(spot, strike, tau, 0.0, rate, dividend)
    with np.errstate(over="raise", invalid="raise"):
        a, b = s * np.exp(-q * t), k * np.exp(-r * t)
    return (
        (_return(np.maximum(a - b, 0)), _return(a))
        if kind == "call"
        else (_return(np.maximum(b - a, 0)), _return(b))
    )


def _positive_difference(loga, logb):
    # exp(loga)-exp(logb), avoiding catastrophic subtraction in OTM tails.
    return np.exp(loga) * (-np.expm1(np.minimum(logb - loga, 0.0)))


def price(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend: ArrayLike = 0.0,
    kind: str = "call",
) -> float | np.ndarray:
    """BSM price, scalar or NumPy broadcast array; exact degenerate branches.

    Underflow in negligible option tails is allowed. Overflow is an explicit
    FloatingPointError rather than a silently infinite price.
    """
    _kind(kind)
    s, k, t, v, r, q = inputs(spot, strike, tau, sigma, rate, dividend)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a, b = s * np.exp(-q * t), k * np.exp(-r * t)
        result = np.array(
            np.maximum(a - b if kind == "call" else b - a, 0.0), copy=True
        )
        mask = (t > 0) & (v > 0) & (s > 0)
        if np.any(mask):
            sm, km, tm, vm, rm, qm = [x[mask] for x in (s, k, t, v, r, q)]
            z = vm * np.sqrt(tm)
            d1 = (np.log(sm) - np.log(km) + (rm - qm) * tm) / z + z / 2
            d2 = d1 - z
            la, lb = np.log(sm) - qm * tm, np.log(km) - rm * tm
            otmcall = _positive_difference(la + log_ndtr(d1), lb + log_ndtr(d2))
            otmput = _positive_difference(lb + log_ndtr(-d2), la + log_ndtr(-d1))
            am, bm = a[mask], b[mask]
            c = np.where(am <= bm, otmcall, otmput + (am - bm))
            p = np.where(am >= bm, otmput, otmcall + (bm - am))
            result[mask] = c if kind == "call" else p
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("price not representable in float64")
    return _return(result)


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend: ArrayLike = 0.0,
    kind: str = "call",
) -> dict[str, float | np.ndarray]:
    """Delta, gamma, vega, calendar theta and rho; strict smooth domain only."""
    _kind(kind)
    s, k, t, v, r, q = inputs(spot, strike, tau, sigma, rate, dividend)
    if np.any((s <= 0) | (t <= 0) | (v <= 0)):
        raise ValueError("Greeks require spot, tau and sigma strictly positive")
    z = v * np.sqrt(t)
    d1 = (np.log(s) - np.log(k) + (r - q) * t) / z + z / 2
    d2 = d1 - z
    density = np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi)
    eq, er = np.exp(-q * t), np.exp(-r * t)
    sign = 1 if kind == "call" else -1
    return {
        name: _return(value)
        for name, value in {
            "delta": sign * eq * ndtr(sign * d1),
            "gamma": eq * density / (s * z),
            "vega": s * eq * density * np.sqrt(t),
            "theta": -s * eq * density * v / (2 * np.sqrt(t))
            - sign * r * k * er * ndtr(sign * d2)
            + sign * q * s * eq * ndtr(sign * d1),
            "rho": sign * k * t * er * ndtr(sign * d2),
        }.items()
    }


def implied_vol(
    target: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    max_vol: float = 10.0,
) -> float:
    """Scalar Brent inversion. Upper bound has no finite implied volatility."""
    _kind(kind)
    inputs(spot, strike, tau, 0.0, rate, dividend)
    if not np.isfinite(target) or not np.isfinite(max_vol) or max_vol <= 0:
        raise ValueError("finite target and positive finite max_vol required")
    if tau <= 0 or spot <= 0:
        raise ValueError("implied volatility is unidentified at expiry or zero spot")
    low, high = bounds(spot, strike, tau, rate, dividend, kind)
    if not low <= target < high:
        raise ValueError("target outside invertible no-arbitrage interval")
    if target == low:
        return 0.0
    f = lambda vol: price(spot, strike, tau, vol, rate, dividend, kind) - target
    if f(max_vol) < 0:
        raise ValueError("root exceeds max_vol; enlarge bracket explicitly")
    return brentq(f, 0.0, max_vol, xtol=1e-13, rtol=1e-13)
