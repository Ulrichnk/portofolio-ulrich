import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from quantlab.black_scholes import (
    price,
    bounds,
    greeks,
    implied_vol,
    EuropeanOption,
    Market,
)


class BlackScholesTests(unittest.TestCase):
    def test_reference(self):
        self.assertAlmostEqual(
            price(100, 100, 1, 0.2, 0.05), 10.450583572185565, places=11
        )
        self.assertAlmostEqual(
            price(100, 100, 1, 0.2, 0.05, kind="put"), 5.573526022256971, places=11
        )

    def test_independent_quadrature(self):
        for s, k, r, q, v, t in [
            (80, 100, 0.03, 0, 0.25, 0.5),
            (120, 100, 0.02, 0.01, 0.3, 2),
            (100, 100, -0.01, 0, 0.2, 1),
        ]:
            zstar = (np.log(k / s) - (r - q - 0.5 * v * v) * t) / (v * np.sqrt(t))
            ref = quad(
                lambda z: np.exp(-r * t)
                * (s * np.exp((r - q - 0.5 * v * v) * t + v * np.sqrt(t) * z) - k)
                * norm.pdf(z),
                zstar,
                12,
                epsabs=1e-10,
            )[0]
            self.assertAlmostEqual(price(s, k, t, v, r, q), ref, places=9)

    def test_properties(self):
        rng = np.random.default_rng(5)
        n = 500
        s = rng.uniform(1, 300, n)
        k = rng.uniform(1, 300, n)
        t = rng.uniform(0.001, 5, n)
        v = rng.uniform(0.001, 1, n)
        r = rng.uniform(-0.03, 0.1, n)
        q = rng.uniform(0, 0.08, n)
        c = price(s, k, t, v, r, q)
        p = price(s, k, t, v, r, q, "put")
        np.testing.assert_allclose(
            c - p, s * np.exp(-q * t) - k * np.exp(-r * t), atol=2e-12, rtol=1e-12
        )
        for name, val in [("call", c), ("put", p)]:
            lo, hi = bounds(s, k, t, r, q, name)
            self.assertTrue(np.all(val >= lo - 1e-11))
            self.assertTrue(np.all(val <= hi + 1e-11))
        grid = np.linspace(1, 300, 1000)
        call = price(grid, 100, 1, 0.2, 0.05)
        put = price(grid, 100, 1, 0.2, 0.05, kind="put")
        self.assertTrue(np.all(np.diff(call) >= -1e-12))
        self.assertTrue(np.all(np.diff(put) <= 1e-12))
        self.assertTrue(np.all(np.diff(call, 2) >= -1e-11))
        self.assertTrue(np.all(np.diff(put, 2) >= -1e-11))

    def test_degenerate(self):
        for name, sign in [("call", 1), ("put", -1)]:
            for s in (0, 80, 100, 120):
                self.assertEqual(
                    price(s, 100, 0, 0.2, kind=name), max(sign * (s - 100), 0)
                )
                self.assertAlmostEqual(
                    price(s, 100, 1, 0, 0.03, 0.01, name),
                    max(sign * (s * np.exp(-0.01) - 100 * np.exp(-0.03)), 0),
                )
                self.assertAlmostEqual(
                    price(s, 100, 1e-12, 0.2, kind=name),
                    max(sign * (s - 100), 0),
                    delta=1e-4,
                )

    def test_broadcast(self):
        s = np.array([80, 100, 120])[:, None]
        v = np.array([0.1, 0.3])[None, :]
        val = price(s, 100, 1, v, 0.03, 0.01)
        for i in range(3):
            for j in range(2):
                self.assertEqual(val[i, j], price(s[i, 0], 100, 1, v[0, j], 0.03, 0.01))

    def test_inputs_and_extremes(self):
        for bad in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                price(bad, 100, 1, 0.2)
        for key, value in [
            ("strike", 0),
            ("tau", -1),
            ("sigma", -1),
            ("rate", np.inf),
            ("kind", "bad"),
        ]:
            args = dict(spot=100, strike=100, tau=1, sigma=0.2)
            args[key] = value
            with self.assertRaises(ValueError):
                price(**args)
        with self.assertRaises(ValueError):
            price([1, 2], 100, [1, 2, 3], 0.2)
        for s in (1e-100, 1e-8, 1e8, 1e100):
            self.assertTrue(np.isfinite(price(s, 100, 1, 0.8, -0.01, 0.02)))
        with self.assertRaises(FloatingPointError):
            price(100, 100, 100, 0.2, -100)

    def test_greeks(self):
        s, k, t, v, r, q = 103, 100, 1.3, 0.27, 0.02, 0.01
        h = 1e-3
        small = 1e-5
        for kind in ("call", "put"):
            f = lambda s, t, v, r: price(s, k, t, v, r, q, kind)
            g = greeks(s, k, t, v, r, q, kind)
            self.assertAlmostEqual(
                g["delta"], (f(s + h, t, v, r) - f(s - h, t, v, r)) / (2 * h), places=8
            )
            self.assertAlmostEqual(
                g["gamma"],
                (f(s + h, t, v, r) - 2 * f(s, t, v, r) + f(s - h, t, v, r)) / h**2,
                places=6,
            )
            self.assertAlmostEqual(
                g["vega"],
                (f(s, t, v + small, r) - f(s, t, v - small, r)) / (2 * small),
                places=6,
            )
            self.assertAlmostEqual(
                g["theta"],
                -(f(s, t + small, v, r) - f(s, t - small, v, r)) / (2 * small),
                places=6,
            )
            self.assertAlmostEqual(
                g["rho"],
                (f(s, t, v, r + small) - f(s, t, v, r - small)) / (2 * small),
                places=6,
            )

    def test_iv(self):
        for kind in ("call", "put"):
            for v in (0.08, 0.2, 0.8):
                target = price(105, 100, 1, v, 0.03, 0.01, kind)
                self.assertAlmostEqual(
                    implied_vol(target, 105, 100, 1, 0.03, 0.01, kind), v, places=10
                )
        with self.assertRaises(ValueError):
            implied_vol(200, 100, 100, 1)
        with self.assertRaises(ValueError):
            implied_vol(0, 100, 100, 0)

    def test_instrument(self):
        self.assertAlmostEqual(
            Market(100, 0.2, 0.05).price(EuropeanOption(100, 1)), 10.450583572185565
        )

    def test_market_validation(self):
        with self.assertRaises(ValueError):
            Market(-1, 0.2)
        with self.assertRaises(ValueError):
            Market(100, -0.2)
        with self.assertRaises(ValueError):
            Market(np.array([90, 100]), 0.2)
