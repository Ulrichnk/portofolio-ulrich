"""Independent CRR, simulation and finite-difference pricing engines."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import t as student_t
from .black_scholes import inputs, price, greeks, _kind


def _integer(value, minimum, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _scalar(spot, strike, tau, sigma, rate, dividend, kind):
    _kind(kind)
    a = inputs(spot, strike, tau, sigma, rate, dividend)
    if any(x.ndim for x in a):
        raise ValueError("this engine accepts scalar parameters only")


def crr(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    steps: int = 500,
    american: bool = False,
) -> float:
    """Recombining Cox-Ross-Rubinstein tree, O(steps) working memory.

    American zero-volatility cases maximize discounted deterministic exercise
    payoffs over the same exercise grid. Increase steps for continuous exercise.
    """
    _scalar(spot, strike, tau, sigma, rate, dividend, kind)
    _integer(steps, 1, "steps")
    sign = 1 if kind == "call" else -1
    if tau == 0 or spot == 0 or sigma == 0:
        times = np.linspace(0, tau, steps + 1) if american else np.array([tau])
        return float(
            np.max(
                np.exp(-rate * times)
                * np.maximum(
                    sign * (spot * np.exp((rate - dividend) * times) - strike), 0
                )
            )
        )
    dt = tau / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((rate - dividend) * dt) - d) / (u - d)
    if not 0 <= p <= 1:
        raise ValueError("tree probability outside [0,1]; increase steps")
    j = np.arange(steps + 1)
    st = spot * np.exp((2 * j - steps) * np.log(u))
    value = np.maximum(sign * (st - strike), 0)
    for n in range(steps - 1, -1, -1):
        value = np.exp(-rate * dt) * ((1 - p) * value[:-1] + p * value[1:])
        if american:
            st = spot * np.exp((2 * np.arange(n + 1) - n) * np.log(u))
            value = np.maximum(value, sign * (st - strike))
    return float(value[0])


@dataclass(frozen=True)
class Estimate:
    value: float
    standard_error: float
    ci_low: float
    ci_high: float
    independent_samples: int


def _estimate(samples):
    samples = np.asarray(samples)
    if np.any(~np.isfinite(samples)):
        raise FloatingPointError("nonfinite Monte Carlo observations")
    n = len(samples)
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(n))
    critical = student_t.ppf(0.975, n - 1)
    return Estimate(mean, se, mean - critical * se, mean + critical * se, n)


def monte_carlo(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    samples: int = 10000,
    steps: int = 64,
    seed: int = 42,
    antithetic: bool = True,
    payoff: str = "european",
    barrier: float | None = None,
    control: bool = False,
    pilot_samples: int = 1024,
) -> Estimate:
    """Exact GBM steps; antithetic PAIRS are independent observations.

    `samples` counts independent pairs when antithetic=True, otherwise paths.
    Asian payoff averages monitoring dates dt,...,T (excludes S0). Barrier is a
    discretely monitored up-and-out, including S0 and T, with no rebate.
    Control variate: discounted terminal stock, coefficient from an independent
    pilot, then held fixed. Confidence intervals quantify sampling error only.
    """
    _scalar(spot, strike, tau, sigma, rate, dividend, kind)
    _integer(samples, 2, "samples")
    _integer(steps, 1, "steps")
    _integer(pilot_samples, 2, "pilot_samples")
    if payoff not in ("european", "asian", "geometric-asian", "up-and-out"):
        raise ValueError("unknown payoff")
    if payoff == "up-and-out" and (
        barrier is None or not np.isfinite(barrier) or barrier <= 0
    ):
        raise ValueError("positive finite barrier required")
    rng = np.random.default_rng(seed)
    sign = 1 if kind == "call" else -1
    dt = tau / steps

    def draw(n):
        # Chunk paths to cap peak memory; each block preserves pair independence.
        values = []
        controls = []
        for start in range(0, n, 2048):
            z = rng.standard_normal((min(2048, n - start), steps))
            ys = []
            xs = []
            for eps in ((1, -1) if antithetic else (1,)):
                paths = spot * np.exp(
                    np.cumsum(
                        (rate - dividend - 0.5 * sigma * sigma) * dt
                        + sigma * np.sqrt(dt) * eps * z,
                        axis=1,
                    )
                )
                terminal = paths[:, -1]
                level = paths.mean(axis=1) if payoff == "asian" else terminal
                if payoff == "geometric-asian":
                    level = (
                        np.zeros(len(paths))
                        if spot == 0
                        else np.exp(np.mean(np.log(paths), axis=1))
                    )
                y = np.exp(-rate * tau) * np.maximum(sign * (level - strike), 0)
                if payoff == "up-and-out":
                    y *= (np.max(paths, axis=1) < barrier) & (spot < barrier)
                ys.append(y)
                xs.append(np.exp(-rate * tau) * terminal)
            values.append(np.mean(ys, axis=0))
            controls.append(np.mean(xs, axis=0))
        return np.concatenate(values), np.concatenate(controls)

    beta = 0.0
    if control:
        yp, xp = draw(pilot_samples)
        variance = np.var(xp, ddof=1)
        beta = 0.0 if variance < 1e-25 else np.cov(yp, xp, ddof=1)[0, 1] / variance
    y, x = draw(samples)
    return _estimate(y - beta * (x - spot * np.exp(-dividend * tau)))


def finite_difference(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    space_steps: int = 400,
    time_steps: int = 400,
    smax: float | None = None,
    theta: float = 0.5,
) -> float:
    """European theta scheme on a uniform spot grid; no American exercise.

    theta=0: explicit, 1: implicit, .5: Crank-Nicolson with two initial
    backward-Euler half steps (Rannacher smoothing). Caller controls truncation.
    Explicit steps enforce sufficient positive-weight stability conditions.
    """
    _scalar(spot, strike, tau, sigma, rate, dividend, kind)
    _integer(space_steps, 4, "space_steps")
    _integer(time_steps, 1, "time_steps")
    if theta not in (0.0, 0.5, 1.0):
        raise ValueError("theta must be 0, .5 or 1")
    if tau == 0 or sigma == 0 or spot == 0:
        return price(spot, strike, tau, sigma, rate, dividend, kind)
    smax = float(smax if smax is not None else max(4 * strike, 2 * spot))
    if not np.isfinite(smax) or smax <= max(spot, strike):
        raise ValueError("smax must exceed spot and strike")
    grid = np.linspace(0, smax, space_steps + 1)
    ds = grid[1]
    sign = 1 if kind == "call" else -1
    v = np.maximum(sign * (grid - strike), 0)
    j = np.arange(1, space_steps)
    a = 0.5 * (sigma * sigma * j * j - (rate - dividend) * j)
    b = -sigma * sigma * j * j - rate
    c = 0.5 * (sigma * sigma * j * j + (rate - dividend) * j)

    def boundary(elapsed):
        if kind == "call":
            return 0.0, max(
                smax * np.exp(-dividend * elapsed) - strike * np.exp(-rate * elapsed),
                0.0,
            )
        return strike * np.exp(-rate * elapsed), 0.0

    elapsed = 0.0
    schedule = [(tau / time_steps, theta)] * time_steps
    if theta == 0.5:
        schedule = [(tau / (2 * time_steps), 1.0)] * 2 + schedule[1:]
    for dt, weight in schedule:
        if weight == 0 and (np.any(a < 0) or np.any(c < 0) or np.any(1 + dt * b < 0)):
            raise ValueError(
                "explicit grid fails sufficient monotonicity/stability test"
            )
        rhs = (1 + (1 - weight) * dt * b) * v[1:-1]
        rhs += (1 - weight) * dt * (a * v[:-2] + c * v[2:])
        lo, hi = boundary(elapsed + dt)
        rhs[0] += weight * dt * a[0] * lo
        rhs[-1] += weight * dt * c[-1] * hi
        ab = np.zeros((3, space_steps - 1))
        ab[1] = 1 - weight * dt * b
        ab[0, 1:] = -weight * dt * c[:-1]
        ab[2, :-1] = -weight * dt * a[1:]
        v[1:-1] = solve_banded((1, 1), ab, rhs)
        v[0], v[-1] = lo, hi
        elapsed += dt
    return float(np.interp(spot, grid, v))


def delta_hedge(
    spot: float,
    strike: float,
    tau: float,
    realized_vol: float,
    hedge_vol: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    kind: str = "call",
    paths: int = 5000,
    steps: int = 52,
    cost_bps: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Short one option, long BSM delta; terminal discounted hedge P&L.

    Physical drift is set to r-q to isolate discretization/model risk. Dividend
    cash uses a left-point q*S_start*dt approximation at each step end.
    Proportional costs apply to initial, rebalancing and final share turnover.
    """
    _scalar(spot, strike, tau, realized_vol, rate, dividend, kind)
    _integer(paths, 2, "paths")
    _integer(steps, 1, "steps")
    if spot <= 0 or tau <= 0 or not np.isfinite(hedge_vol) or hedge_vol <= 0:
        raise ValueError("hedge requires positive spot, maturity and hedge volatility")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("nonnegative finite costs required")
    rng = np.random.default_rng(seed)
    dt = tau / steps
    cost = cost_bps / 10000
    s = np.full(paths, float(spot))
    delta = np.full(
        paths, greeks(spot, strike, tau, hedge_vol, rate, dividend, kind)["delta"]
    )
    cash = (
        np.full(paths, price(spot, strike, tau, hedge_vol, rate, dividend, kind))
        - delta * s
        - cost * np.abs(delta) * s
    )
    for j in range(steps):
        old = s.copy()
        s *= np.exp(
            (rate - dividend - 0.5 * realized_vol**2) * dt
            + realized_vol * np.sqrt(dt) * rng.standard_normal(paths)
        )
        # Discrete cash dividend convention q*S_start*dt, paid at step end.
        cash = cash * np.exp(rate * dt) + delta * old * dividend * dt
        if j < steps - 1:
            new = greeks(
                s, strike, tau - (j + 1) * dt, hedge_vol, rate, dividend, kind
            )["delta"]
            cash -= (new - delta) * s + cost * np.abs(new - delta) * s
            delta = new
    payoff = np.maximum((1 if kind == "call" else -1) * (s - strike), 0)
    pnl = np.exp(-rate * tau) * (cash + delta * s - payoff - cost * np.abs(delta) * s)
    return pnl
