"""Regenerate all numerical evidence from the repository root.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/reproduce.py
Outputs use deterministic seeds except wall-clock timings. No web downloads.
"""

from pathlib import Path
import csv
import json
import platform
import time
import sys
import numpy as np
import scipy
import sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from quantlab.black_scholes import price, greeks, implied_vol
from quantlab.numerical import crr, monte_carlo, finite_difference, delta_hedge
from quantlab.heston import Heston, heston_price, calibrate, surface_diagnostics
from quantlab.rates import (
    DiscountCurve,
    bootstrap,
    par_swap_rate,
    swap_pv,
    bond_pv,
    risk_measures,
)
from quantlab.risk import run_research
from quantlab.exotics import geometric_asian, continuous_up_out_call

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update(
    {"figure.figsize": (8, 4.5), "axes.grid": True, "grid.alpha": 0.2, "font.size": 10}
)


def output(project):
    p = ROOT / project / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fig(path):
    plt.tight_layout()
    plt.savefig(path, metadata={"Date": None})
    plt.savefig(path.with_suffix(".png"), dpi=160)
    plt.close()


def derivatives():
    p = output("quant-derivatives-lab")
    rows = []
    cases = [
        ("A", 100, 100, 0.05, 0, 0.2, 1),
        ("B", 80, 100, 0.03, 0, 0.25, 0.5),
        ("C", 120, 100, 0.02, 0.01, 0.3, 2),
        ("D", 100, 100, 0.04, 0, 0.2, 0.01),
        ("E", 100, 100, -0.01, 0, 0.2, 1),
        ("F", 100, 100, 0.03, 0.01, 0.8, 1),
        ("G", 100, 100, 0.03, 0, 0.2, 0),
        ("H", 100, 100, 0.03, 0, 0, 1),
    ]
    for case, s, k, r, q, v, t in cases:
        call = price(s, k, t, v, r, q)
        put = price(s, k, t, v, r, q, "put")
        if t > 0 and v > 0:
            z = (np.log(k / s) - (r - q - v * v / 2) * t) / (v * np.sqrt(t))
            ref = quad(
                lambda x: np.exp(-r * t)
                * (s * np.exp((r - q - v * v / 2) * t + v * np.sqrt(t) * x) - k)
                * norm.pdf(x),
                z,
                14,
                epsabs=1e-11,
            )[0]
        else:
            ref = max(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
        rows.append(
            dict(
                case=case,
                spot=s,
                strike=k,
                rate=r,
                dividend=q,
                sigma=v,
                tau=t,
                call=call,
                put=put,
                quadrature_call=ref,
                absolute_error=abs(call - ref),
                parity_residual=call - put - s * np.exp(-q * t) + k * np.exp(-r * t),
            )
        )
    table(p / "reference_cases.csv", rows)
    spots = np.linspace(40, 160, 301)
    plt.plot(spots, price(spots, 100, 1, 0.2, 0.05), label="Call")
    plt.plot(spots, price(spots, 100, 1, 0.2, 0.05, kind="put"), label="Put")
    plt.plot(spots, np.maximum(spots - 100, 0), "--", label="Call intrinsic")
    plt.axvline(100, color="gray", linestyle=":")
    plt.xlabel("Spot")
    plt.ylabel("Price")
    plt.title("BSM prices: K=100, T=1, sigma=20%, r=5%, q=0")
    plt.legend()
    fig(p / "spot.svg")
    sensitivities = [
        ("sigma", np.linspace(0.01, 1, 100)),
        ("tau", np.linspace(0.01, 3, 100)),
        ("rate", np.linspace(-0.05, 0.15, 100)),
        ("dividend", np.linspace(0, 0.15, 100)),
        ("strike", np.linspace(60, 140, 100)),
    ]
    sensitivity_rows = []
    for param, x in sensitivities:
        kwargs = dict(spot=100, strike=100, tau=1, sigma=0.2, rate=0.05, dividend=0)
        kwargs[param] = x
        c = price(**kwargs)
        pp = price(**kwargs, kind="put")
        for a, b, d in zip(x, c, pp):
            sensitivity_rows.append(dict(parameter=param, value=a, call=b, put=d))
        plt.plot(x, c, label="Call")
        plt.plot(x, pp, label="Put")
        plt.legend()
        plt.xlabel(param)
        plt.ylabel("Price")
        plt.title("Single-parameter BSM sensitivity: other parameters at reference A")
        fig(p / f"sensitivity_{param}.svg")
    table(p / "sensitivities.csv", sensitivity_rows)
    ref = price(100, 100, 1, 0.2, 0.05)
    convergence = []
    for n in [25, 50, 100, 200, 400, 800]:
        tree = crr(100, 100, 1, 0.2, 0.05, steps=n)
        fd = finite_difference(
            100, 100, 1, 0.2, 0.05, space_steps=n if n >= 50 else 50, time_steps=n
        )
        convergence.append(
            dict(
                steps=n,
                tree=tree,
                tree_error=abs(tree - ref),
                pde=fd,
                pde_error=abs(fd - ref),
            )
        )
    table(p / "convergence.csv", convergence)
    plt.loglog(
        [x["steps"] for x in convergence],
        [x["tree_error"] for x in convergence],
        "o-",
        label="CRR",
    )
    plt.loglog(
        [x["steps"] for x in convergence],
        [x["pde_error"] for x in convergence],
        "o-",
        label="Rannacher-CN",
    )
    plt.xlabel("Steps (PDE space floor=50)")
    plt.ylabel("Absolute error")
    plt.title("Convergence to BSM reference A")
    plt.legend()
    fig(p / "convergence.svg")
    mc = []
    for n in [1000, 4000, 16000]:
        for anti, control in [(False, False), (True, False), (True, True)]:
            for seed in [7, 42, 2026]:
                start = time.perf_counter()
                est = monte_carlo(
                    100,
                    100,
                    1,
                    0.2,
                    0.05,
                    samples=n,
                    steps=1,
                    seed=seed,
                    antithetic=anti,
                    control=control,
                )
                elapsed = time.perf_counter() - start
                mc.append(
                    dict(
                        samples=n,
                        antithetic=anti,
                        control=control,
                        seed=seed,
                        price=est.value,
                        standard_error=est.standard_error,
                        ci_low=est.ci_low,
                        ci_high=est.ci_high,
                        contains_reference=est.ci_low <= ref <= est.ci_high,
                        seconds=elapsed,
                        efficiency=1 / (max(elapsed, 1e-12) * est.standard_error**2),
                    )
                )
    table(p / "monte_carlo.csv", mc)
    coverage = []
    for seed in range(200):
        est = monte_carlo(
            100, 100, 1, 0.2, 0.05, samples=2000, steps=1, seed=seed, control=True
        )
        coverage.append(est.ci_low <= ref <= est.ci_high)
    exotics = []
    for steps in [16, 64, 256]:
        for payoff in ["asian", "up-and-out"]:
            est = monte_carlo(
                100,
                100,
                1,
                0.2,
                0.05,
                payoff=payoff,
                barrier=130,
                samples=16000,
                steps=steps,
                control=True,
            )
            exotics.append(dict(payoff=payoff, monitoring_dates=steps, **est.__dict__))
    table(p / "exotics.csv", exotics)
    geometric = monte_carlo(
        100, 100, 1, 0.2, 0.05, payoff="geometric-asian", samples=20000, control=True
    )
    dump(
        p / "exotic_references.json",
        dict(
            geometric_64_dates=geometric_asian(100, 100, 1, 0.2, 0.05),
            geometric_mc=geometric.__dict__,
            continuous_up_out_130=continuous_up_out_call(100, 100, 1, 0.2, 130, 0.05),
        ),
    )
    independent_grids = []
    for m in [100, 200, 400, 800]:
        val = finite_difference(100, 100, 1, 0.2, 0.05, space_steps=m, time_steps=1600)
        independent_grids.append(
            dict(axis="space", steps=m, price=val, error=abs(val - ref))
        )
    for n in [25, 50, 100, 200, 400]:
        val = finite_difference(100, 100, 1, 0.2, 0.05, space_steps=1600, time_steps=n)
        independent_grids.append(
            dict(axis="time", steps=n, price=val, error=abs(val - ref))
        )
    for cutoff in [200, 300, 400, 600]:
        val = finite_difference(
            100, 100, 1, 0.2, 0.05, space_steps=cutoff * 2, time_steps=800, smax=cutoff
        )
        independent_grids.append(
            dict(axis="smax_fixed_ds", steps=cutoff, price=val, error=abs(val - ref))
        )
    table(p / "pde_isolated_grids.csv", independent_grids)
    # Barrier monitoring-bias study using nested grids on identical Brownian paths.
    rng = np.random.default_rng(2026)
    z = rng.standard_normal((12000, 256))
    paths = 100 * np.exp(np.cumsum((0.05 - 0.5 * 0.2**2) / 256 + 0.2 / 16 * z, axis=1))
    barrier_nested = []
    for steps in [16, 64, 256]:
        selected = paths[:, 256 // steps - 1 :: 256 // steps]
        value = (
            np.exp(-0.05)
            * np.maximum(paths[:, -1] - 100, 0)
            * (selected.max(axis=1) < 130)
        )
        barrier_nested.append(
            dict(
                steps=steps,
                price=float(value.mean()),
                se=float(value.std(ddof=1) / np.sqrt(len(value))),
            )
        )
    table(p / "barrier_nested.csv", barrier_nested)
    hedge = []
    for steps in [12, 52, 252]:
        for realized, cost in [(0.2, 0), (0.2, 10), (0.3, 0)]:
            pnl = delta_hedge(
                100,
                100,
                1,
                realized,
                0.2,
                0.05,
                paths=10000,
                steps=steps,
                cost_bps=cost,
            )
            hedge.append(
                dict(
                    steps=steps,
                    realized_vol=realized,
                    hedge_vol=0.2,
                    cost_bps=cost,
                    mean=float(pnl.mean()),
                    std=float(pnl.std(ddof=1)),
                    quantile_01=float(np.quantile(pnl, 0.01)),
                    quantile_99=float(np.quantile(pnl, 0.99)),
                )
            )
    table(p / "hedging.csv", hedge)
    model = Heston()
    strikes = np.tile([80, 90, 100, 110, 120], 3)
    maturities = np.repeat([0.5, 1, 2], 5)
    quotes = np.array(
        [heston_price(100, k, t, model, 0.03) for k, t in zip(strikes, maturities)]
    )
    fits = []
    for initial in [
        Heston(0.05, 1.5, 0.05, 0.4, -0.5),
        Heston(0.03, 3, 0.06, 0.5, -0.3),
    ]:
        fitted, res = calibrate(100, strikes, maturities, quotes, initial, 0.03)
        singular = np.linalg.svd(res.jac, compute_uv=False)
        holdout = [
            abs(
                heston_price(100, k, t, fitted, 0.03)
                - heston_price(100, k, t, model, 0.03)
            )
            for k, t in [(95, 0.75), (105, 1.5)]
        ]
        fits.append(
            dict(
                parameters=fitted.__dict__,
                success=bool(res.success),
                nfev=res.nfev,
                price_rmse=float(np.sqrt(np.mean(res.fun**2))),
                jacobian_singular_values=singular.tolist(),
                jacobian_condition=float(singular[0] / singular[-1]),
                holdout_max_error=max(holdout),
            )
        )
    surface_rows = []
    for k, t, c in zip(strikes, maturities, quotes):
        iv = implied_vol(c, 100, k, t, 0.03)
        surface_rows.append(
            dict(strike=int(k), maturity=t, call=c, implied_volatility=iv)
        )
    table(p / "heston_surface.csv", surface_rows)
    for t in [0.5, 1, 2]:
        subset = [x for x in surface_rows if x["maturity"] == t]
        plt.plot(
            [x["strike"] for x in subset],
            [x["implied_volatility"] for x in subset],
            "o-",
            label=f"T={t}",
        )
    plt.xlabel("Strike")
    plt.ylabel("BSM implied volatility")
    plt.title("Synthetic Heston skew, S=100, r=3%, q=0")
    plt.legend()
    fig(p / "heston_smile.svg")
    diag = surface_diagnostics(
        np.array([80, 90, 100, 110, 120]),
        np.array([0.5, 1, 2]),
        quotes.reshape(3, 5),
        100,
        0.03,
    )
    dump(
        p / "heston_calibration.json",
        dict(true_parameters=model.__dict__, fits=fits, discrete_surface_checks=diag),
    )
    timings = []
    for n in [1, 100, 10000]:
        s = np.linspace(80, 120, n)
        price(s, 100, 1, 0.2, 0.05)
        vector = []
        scalar = []
        for _ in range(5):
            start = time.perf_counter()
            price(s, 100, 1, 0.2, 0.05)
            vector.append(time.perf_counter() - start)
            start = time.perf_counter()
            [price(x, 100, 1, 0.2, 0.05) for x in s]
            scalar.append(time.perf_counter() - start)
        timings.append(
            dict(
                n=n,
                vector_median_seconds=float(np.median(vector)),
                scalar_median_seconds=float(np.median(scalar)),
            )
        )
    table(p / "performance.csv", timings)
    errors = []
    exact = greeks(100, 100, 1, 0.2, 0.05)["gamma"]
    for h in np.logspace(-6, 0, 25):
        approx = (
            price(100 + h, 100, 1, 0.2, 0.05)
            - 2 * ref
            + price(100 - h, 100, 1, 0.2, 0.05)
        ) / h**2
        errors.append(dict(step=h, gamma=approx, absolute_error=abs(approx - exact)))
    table(p / "gamma_steps.csv", errors)
    summary = dict(
        reference_call=ref,
        max_reference_error=max(x["absolute_error"] for x in rows),
        max_parity_residual=max(abs(x["parity_residual"]) for x in rows),
        mc_coverage_200_replicates=float(np.mean(coverage)),
        heston_fits=fits,
        finest_convergence=convergence[-1],
        hedging=hedge,
    )
    dump(p / "summary.json", summary)
    return summary


def rates():
    p = output("interest-rate-analytics")
    times = np.r_[0, np.arange(0.5, 10.01, 0.5)]
    zero = 0.02 + 0.012 * (1 - np.exp(-times / 3))
    original = DiscountCurve(tuple(times), tuple(np.exp(-zero * times)))
    deposits = [(0.5, (1 / original.df(0.5) - 1) / 0.5), (1, 1 / original.df(1) - 1)]
    swaps = [
        (
            np.arange(0.5, t + 0.01, 0.5),
            par_swap_rate(original, np.arange(0.5, t + 0.01, 0.5)),
        )
        for t in np.arange(1.5, 10.01, 0.5)
    ]
    curve = bootstrap(deposits, swaps)
    rows = [
        dict(
            instrument="deposit", maturity=t, quote=r, reprice=(1 / curve.df(t) - 1) / t
        )
        for t, r in deposits
    ]
    rows += [
        dict(
            instrument="swap",
            maturity=float(t[-1]),
            quote=r,
            reprice=par_swap_rate(curve, t),
        )
        for t, r in swaps
    ]
    table(p / "synthetic_quotes.csv", rows)
    grid = np.linspace(0.05, 10, 250)
    table(
        p / "curve.csv",
        [dict(time=t, discount=curve.df(t), zero=float(curve.zero(t))) for t in grid],
    )
    plt.plot(grid, 100 * curve.zero(grid))
    plt.xlabel("Maturity (years)")
    plt.ylabel("Continuous zero rate (%)")
    plt.title("Bootstrapped synthetic single-curve term structure")
    fig(p / "zero_curve.svg")
    schedule = np.arange(0.5, 5.01, 0.5)
    fixed = par_swap_rate(curve, schedule)
    bond = lambda c: bond_pv(c, schedule, 0.035, 100.0)
    swap = lambda c: swap_pv(c, schedule, fixed, 1e6)
    br = risk_measures(bond, curve)
    sr = risk_measures(swap, curve)
    scenarios = []
    for bp in [-100, -50, -10, 10, 50, 100]:
        shift = bp / 10000
        for name, f, risk in [("bond", bond, br), ("payer_swap", swap, sr)]:
            actual = f(curve.shifted(shift)) - risk["pv"]
            linear = -risk["dollar_duration"] * shift
            quadratic = linear + 0.5 * risk["dollar_convexity"] * shift**2
            scenarios.append(
                dict(
                    instrument=name,
                    shift_bp=bp,
                    actual_pnl=actual,
                    duration_pnl=linear,
                    duration_convexity_pnl=quadratic,
                    quadratic_error=actual - quadratic,
                )
            )
    table(p / "scenarios.csv", scenarios)
    summary = dict(
        max_quote_repricing_error=max(abs(x["quote"] - x["reprice"]) for x in rows),
        five_year_par_swap_rate=fixed,
        bond_risk=br,
        payer_swap_risk=sr,
        max_discount_error=float(np.max(np.abs(curve.df(times) - original.df(times)))),
    )
    dump(p / "summary.json", summary)
    return summary


def risk():
    p = output("financial-risk-ml")
    result, data, pred, start = run_research(n=4000, seed=42)
    dump(p / "research.json", result)
    rows = []
    for i in range(len(data.y) - start):
        row = dict(
            origin_index=int(data.origins[start + i]),
            next_return=float(data.next_returns[start + i]),
        )
        row.update({name: float(values[i]) for name, values in pred.items()})
        rows.append(row)
    table(p / "out_of_sample_forecasts.csv", rows)
    plt.plot(
        data.origins[start:],
        pred["oracle_garch"],
        label="True conditional variance",
        alpha=0.8,
    )
    plt.plot(data.origins[start:], pred["ridge"], label="Ridge", alpha=0.7)
    plt.plot(data.origins[start:], pred["boosting"], label="Boosting", alpha=0.7)
    plt.xlabel("Forecast origin index")
    plt.ylabel("Next-day variance (% return squared)")
    plt.title("Synthetic GARCH: expanding chronological out-of-sample forecasts")
    plt.legend()
    fig(p / "variance_forecasts.svg")
    seeds = []
    for seed in [7, 2026]:
        other, *_ = run_research(n=4000, seed=seed)
        seeds.append(other)
    dump(p / "additional_seeds.json", seeds)
    # Moving block bootstrap of paired loss difference vs historical baseline.
    y = data.y[start:]
    baseline = np.log(pred["historical_20"]) + y / pred["historical_20"]
    rng = np.random.default_rng(42)
    intervals = {}
    block = 20
    n = len(y)
    for name in ["ridge", "boosting"]:
        loss = np.log(pred[name]) + y / pred[name] - baseline
        means = []
        for _ in range(2000):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
            means.append(float(np.mean(loss[idx])))
        intervals[name] = dict(
            mean_qlike_difference=float(np.mean(loss)),
            block_length=block,
            bootstrap_95pct=np.quantile(means, [0.025, 0.975]).tolist(),
        )
    dump(p / "paired_loss_uncertainty.json", intervals)
    return result


if __name__ == "__main__":
    print("Derivatives experiments", flush=True)
    d = derivatives()
    print("Rates experiments", flush=True)
    r = rates()
    print("Risk experiments", flush=True)
    m = risk()
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seed_policy": "42 unless a replicate or scenario states otherwise",
    }
    dump(ROOT / "experiments" / "environment.json", env)
    print(json.dumps(dict(derivatives=d, rates=r, risk=m), indent=2), flush=True)
