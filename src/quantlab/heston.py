"""Heston Fourier pricing and bounded synthetic calibration.

Risk-neutral CIR variance with parameters (v0, kappa, theta, xi, rho).
No Feller restriction is imposed: CIR remains nonnegative but may hit zero.
Fourier quadrature truncation must be checked independently of optimizer error.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import least_squares
from .black_scholes import inputs, price, bounds


@dataclass(frozen=True)
class Heston:
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7

    def __post_init__(self):
        a = np.asarray([self.v0, self.kappa, self.theta, self.xi, self.rho])
        if not np.all(np.isfinite(a)) or np.any(a[:4] <= 0) or abs(self.rho) >= 1:
            raise ValueError("positive variance parameters and |rho|<1 required")

    @property
    def feller_margin(self):
        return 2 * self.kappa * self.theta - self.xi**2


def characteristic(u, tau, spot, rate, dividend, model):
    """Little-Heston-trap characteristic function E[exp(i*u*log(S_T))]."""
    v0, k, theta, xi, rho = model.v0, model.kappa, model.theta, model.xi, model.rho
    iu = 1j * u
    b = k - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (u * u + iu))
    g = (b - d) / (b + d)
    decay = np.exp(-d * tau)
    c = (rate - dividend) * iu * tau + (k * theta / xi**2) * (
        (b - d) * tau - 2 * np.log((1 - g * decay) / (1 - g))
    )
    D = (b - d) / xi**2 * (1 - decay) / (1 - g * decay)
    return np.exp(iu * np.log(spot) + c + D * v0)


def heston_price(
    spot: float,
    strike: float,
    tau: float,
    model: Heston,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    cutoff: float = 150.0,
    epsabs: float = 1e-8,
) -> float:
    """Lewis damped integral (shift -i/2), price in currency units."""
    inputs(spot, strike, tau, 0.0, rate, dividend)
    if (
        kind not in ("call", "put")
        or not np.all(np.isfinite([cutoff, epsabs]))
        or cutoff <= 0
        or epsabs <= 0
    ):
        raise ValueError("invalid kind or quadrature controls")
    if tau == 0 or spot == 0:
        return price(spot, strike, tau, 0.0, rate, dividend, kind)

    # Integrate CF of log return rather than log stock for numerical conditioning.
    def integrand(u):
        z = u - 0.5j
        cf = characteristic(z, tau, spot, rate, dividend, model) * np.exp(
            -1j * z * np.log(spot)
        )
        return (np.exp(1j * u * np.log(spot / strike)) * cf).real / (u * u + 0.25)

    integral, error = quad(integrand, 0, cutoff, epsabs=epsabs, epsrel=1e-9, limit=300)
    call = (
        spot * np.exp(-dividend * tau)
        - np.sqrt(spot * strike) * np.exp(-rate * tau) * integral / np.pi
    )
    value = (
        call
        if kind == "call"
        else call - spot * np.exp(-dividend * tau) + strike * np.exp(-rate * tau)
    )
    lo, hi = bounds(spot, strike, tau, rate, dividend, kind)
    if not np.isfinite(value) or value < lo - 1e-6 or value > hi + 1e-6:
        raise ArithmeticError(
            "Fourier result violates bounds; increase cutoff/check parameters"
        )
    return float(value)


def calibrate(
    spot: float,
    strikes: ArrayLike,
    maturities: ArrayLike,
    prices: ArrayLike,
    initial: Heston | None = None,
    rate: float = 0.0,
    dividend: float = 0.0,
    max_nfev: int = 150,
) -> tuple:
    """Unweighted call-price least squares; returns model and full diagnostics.

    Requires >=5 quotes. No claim of global optimality or parameter uniqueness.
    Use several starts and inspect Jacobian singular values and holdout errors.
    """
    strikes, maturities, prices = np.broadcast_arrays(strikes, maturities, prices)
    if strikes.size < 5 or not np.all(np.isfinite(prices)):
        raise ValueError("at least five finite quotes required")
    for k, t, p in zip(strikes.flat, maturities.flat, prices.flat):
        lo, hi = bounds(spot, k, t, rate, dividend)
        if t <= 0 or not lo <= p < hi:
            raise ValueError("invalid quote")
    start = initial or Heston(0.05, 1.5, 0.05, 0.4, -0.5)
    x0 = np.array([start.v0, start.kappa, start.theta, start.xi, start.rho])

    def residual(x):
        m = Heston(*x)
        return np.array(
            [
                heston_price(spot, k, t, m, rate, dividend) - p
                for k, t, p in zip(strikes.flat, maturities.flat, prices.flat)
            ]
        )

    fit = least_squares(
        residual,
        x0,
        bounds=([0.001, 0.1, 0.001, 0.03, -0.98], [0.5, 10, 0.5, 2, 0.98]),
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-9,
        x_scale="jac",
    )
    return Heston(*fit.x), fit


def surface_diagnostics(
    strikes: ArrayLike,
    maturities: ArrayLike,
    calls: ArrayLike,
    spot: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    tol: float = 1e-8,
) -> dict[str, bool]:
    """Discrete strike bounds, monotonicity and convexity diagnostics.

    Calendar condition is checked at fixed forward moneyness via C/(S*e^-qT),
    interpolated on overlapping K/F knots. Discrete checks do not certify an
    arbitrage-free continuous interpolation between quoted points.
    """
    k = np.asarray(strikes, float)
    t = np.asarray(maturities, float)
    c = np.asarray(calls, float)
    if (
        k.ndim != 1
        or t.ndim != 1
        or c.shape != (len(t), len(k))
        or len(k) < 3
        or len(t) < 2
    ):
        raise ValueError("require ordered strike/maturity axes and (T,K) price matrix")
    if (
        not np.all(np.diff(k) > 0)
        or not np.all(np.diff(t) > 0)
        or not np.all(np.isfinite(c))
        or spot <= 0
    ):
        raise ValueError("axes must strictly increase and prices must be finite")
    lo, hi = bounds(spot, k[None, :], t[:, None], rate, dividend)
    slopes = np.diff(c, axis=1) / np.diff(k)
    calendar = []
    for i in range(len(t) - 1):
        x1 = k / (spot * np.exp((rate - dividend) * t[i]))
        x2 = k / (spot * np.exp((rate - dividend) * t[i + 1]))
        common = np.unique(np.r_[x1, x2])
        common = common[(common >= max(x1[0], x2[0])) & (common <= min(x1[-1], x2[-1]))]
        if not len(common):
            raise ValueError("no overlapping forward moneyness for calendar check")
        c1 = c[i] / (spot * np.exp(-dividend * t[i]))
        c2 = c[i + 1] / (spot * np.exp(-dividend * t[i + 1]))
        calendar.extend(np.interp(common, x2, c2) - np.interp(common, x1, c1))
    return {
        "bounds": bool(np.all(c >= lo - tol) and np.all(c <= hi + tol)),
        "decreasing_strike": bool(np.all(slopes <= tol)),
        "convex_strike": bool(np.all(np.diff(slopes, axis=1) >= -tol)),
        "vertical_spread_bound": bool(
            np.all(slopes >= -np.exp(-rate * t[:, None]) - tol)
        ),
        "calendar_interpolated": bool(np.all(np.asarray(calendar) >= -tol)),
    }
