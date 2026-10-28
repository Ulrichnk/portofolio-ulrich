"""Single-curve rates analytics with explicit year-fraction schedules.

Discount factors are log-linear between knots, with no extrapolation. Bootstrap
supports consecutive simple deposits followed by spot-starting par swaps; swaps
must extend the existing payment grid by exactly one coupon. No date calendars,
credit, collateral basis or multi-curve forwarding are included.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class DiscountCurve:
    times: tuple
    discounts: tuple

    def __post_init__(self):
        t = np.asarray(self.times, float)
        d = np.asarray(self.discounts, float)
        if t.ndim != 1 or len(t) < 2 or d.shape != t.shape or t[0] != 0 or d[0] != 1:
            raise ValueError("curve requires matching knots starting at (0,1)")
        if (
            not np.all(np.isfinite(t))
            or not np.all(np.isfinite(d))
            or np.any(np.diff(t) <= 0)
            or np.any(d <= 0)
        ):
            raise ValueError(
                "ordered finite times and positive finite discount factors required"
            )
        object.__setattr__(self, "times", tuple(t))
        object.__setattr__(self, "discounts", tuple(d))

    def df(self, t):
        t = np.asarray(t, float)
        if np.any(~np.isfinite(t)) or np.any(t < 0) or np.any(t > self.times[-1]):
            raise ValueError("requested time outside curve; extrapolation disabled")
        out = np.exp(np.interp(t, self.times, np.log(self.discounts)))
        return float(out) if out.ndim == 0 else out

    def zero(self, t):
        if np.any(np.asarray(t) <= 0):
            raise ValueError("zero yield requires positive maturity")
        return -np.log(self.df(t)) / np.asarray(t)

    def forward(self, start, end):
        if not 0 <= start < end:
            raise ValueError("require 0 <= start < end")
        return (self.df(start) / self.df(end) - 1) / (end - start)

    def shifted(self, bump):
        if not np.isfinite(bump):
            raise ValueError("finite bump required")
        return DiscountCurve(
            self.times,
            tuple(np.asarray(self.discounts) * np.exp(-bump * np.asarray(self.times))),
        )


def _schedule(times):
    times = np.asarray(times, float)
    if (
        times.ndim != 1
        or not len(times)
        or not np.all(np.isfinite(times))
        or times[0] <= 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("positive strictly increasing payment year fractions required")
    return times, np.diff(np.r_[0.0, times])


def bootstrap(deposits: list, swaps: tuple | list = ()) -> DiscountCurve:
    """deposits: (maturity, simple rate); swaps: (coupon_times, par rate)."""
    ts = [0.0]
    ds = [1.0]
    for t, r in deposits:
        if not np.isfinite(t) or not np.isfinite(r) or t <= ts[-1] or 1 + r * t <= 0:
            raise ValueError("invalid deposit quote")
        ts.append(float(t))
        ds.append(1 / (1 + r * t))
    for schedule, rate in swaps:
        pay, alpha = _schedule(schedule)
        if not np.isfinite(rate) or pay[-1] <= ts[-1] or np.any(pay[:-1] > ts[-1]):
            raise ValueError("each swap must extend curve by only one coupon")
        curve = DiscountCurve(tuple(ts), tuple(ds))
        denominator = 1 + rate * alpha[-1]
        if denominator <= 0:
            raise ValueError("invalid swap denominator")
        last = (1 - rate * np.dot(alpha[:-1], curve.df(pay[:-1]))) / denominator
        if last <= 0:
            raise ValueError("quotes imply nonpositive discount factor")
        ts.append(float(pay[-1]))
        ds.append(float(last))
    return DiscountCurve(tuple(ts), tuple(ds))


def annuity(curve, times):
    t, alpha = _schedule(times)
    return float(np.dot(alpha, curve.df(t)))


def par_swap_rate(curve: DiscountCurve, times: ArrayLike) -> float:
    t, _ = _schedule(times)
    return (1 - curve.df(t[-1])) / annuity(curve, t)


def swap_pv(
    curve: DiscountCurve,
    times: ArrayLike,
    fixed_rate: float,
    notional: float = 1.0,
    payer: bool = True,
) -> float:
    """Spot-starting payer fixed: PV(float)-PV(fixed), single curve."""
    if not np.isfinite(fixed_rate) or not np.isfinite(notional) or notional < 0:
        raise ValueError("finite rate and nonnegative notional required")
    t, _ = _schedule(times)
    return (
        (1 if payer else -1)
        * notional
        * (1 - curve.df(t[-1]) - fixed_rate * annuity(curve, t))
    )


def bond_pv(
    curve: DiscountCurve, times: ArrayLike, coupon_rate: float, notional: float = 100.0
) -> float:
    """Clean equals dirty at time zero on coupon date; annualized coupons."""
    if not np.isfinite(coupon_rate) or not np.isfinite(notional) or notional < 0:
        raise ValueError("finite coupon and nonnegative notional required")
    t, alpha = _schedule(times)
    return float(
        notional * (coupon_rate * np.dot(alpha, curve.df(t)) + curve.df(t[-1]))
    )


def risk_measures(value, curve: DiscountCurve, bump: float = 1e-4) -> dict[str, float]:
    """Central derivatives wrt parallel continuously compounded zero shift.

    signed_pv01 is approximately the PV change for +1bp. Dollar duration is
    -dPV/dy; dollar convexity is d2PV/dy2 (no price normalization).
    """
    if not np.isfinite(bump) or bump <= 0:
        raise ValueError("positive finite bump required")
    base = value(curve)
    up = value(curve.shifted(bump))
    down = value(curve.shifted(-bump))
    first = (up - down) / (2 * bump)
    return {
        "pv": float(base),
        "signed_pv01": float(first * 1e-4),
        "dollar_duration": float(-first),
        "dollar_convexity": float((up - 2 * base + down) / bump**2),
    }
