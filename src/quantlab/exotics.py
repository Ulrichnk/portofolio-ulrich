"""Independent semi-analytic benchmarks for path-dependent options."""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from .black_scholes import inputs, _kind


def geometric_asian(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    steps: int = 64,
) -> float:
    """Discrete geometric-average Asian; monitoring dt,...,T, excludes S0."""
    inputs(spot, strike, tau, sigma, rate, dividend)
    _kind(kind)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("positive integer monitoring count required")
    sign = 1 if kind == "call" else -1
    if spot == 0:
        return 0.0 if kind == "call" else strike * np.exp(-rate * tau)
    m = np.log(spot) + (rate - dividend - 0.5 * sigma**2) * tau * (steps + 1) / (
        2 * steps
    )
    variance = sigma**2 * tau * (steps + 1) * (2 * steps + 1) / (6 * steps**2)
    if variance == 0:
        return float(np.exp(-rate * tau) * max(sign * (np.exp(m) - strike), 0))
    z = np.sqrt(variance)
    d2 = (m - np.log(strike)) / z
    d1 = d2 + z
    return float(
        np.exp(-rate * tau)
        * sign
        * (np.exp(m + variance / 2) * ndtr(sign * d1) - strike * ndtr(sign * d2))
    )


def continuous_up_out_call(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    barrier: float,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """Continuous up-and-out call via killed lognormal transition density.

    Conditional Brownian-bridge survival is integrated against the terminal
    Gaussian density, independently of the simulation. Zero rebate; immediate
    knockout at S>=B. Numerical quadrature, not a second Monte Carlo engine.
    """
    inputs(spot, strike, tau, sigma, rate, dividend)
    if not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("positive finite barrier required")
    if spot >= barrier or strike >= barrier or spot == 0:
        return 0.0
    if tau == 0:
        return max(spot - strike, 0.0)
    if sigma == 0:
        terminal = spot * np.exp((rate - dividend) * tau)
        return (
            float(np.exp(-rate * tau) * max(terminal - strike, 0))
            if max(spot, terminal) < barrier
            else 0.0
        )
    x = np.log(spot)
    b = np.log(barrier)
    v = sigma**2 * tau
    mean = x + (rate - dividend - 0.5 * sigma**2) * tau

    def f(y):
        survival = -np.expm1(-2 * (b - x) * (b - y) / v)
        density = np.exp(-0.5 * (y - mean) ** 2 / v) / np.sqrt(2 * np.pi * v)
        return (np.exp(y) - strike) * survival * density

    return float(np.exp(-rate * tau) * quad(f, np.log(strike), b, epsabs=1e-10)[0])
