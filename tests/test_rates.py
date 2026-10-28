import unittest
import numpy as np
from quantlab.rates import (
    DiscountCurve,
    bootstrap,
    par_swap_rate,
    swap_pv,
    bond_pv,
    risk_measures,
)


class RateTests(unittest.TestCase):
    def setUp(self):
        self.curve = DiscountCurve(
            (0, 0.5, 1, 2, 3, 4, 5),
            tuple(np.exp(-0.03 * np.array([0, 0.5, 1, 2, 3, 4, 5]))),
        )

    def test_interpolation(self):
        t = np.array([0.25, 0.75, 1.5, 4.9])
        np.testing.assert_allclose(self.curve.df(t), np.exp(-0.03 * t), rtol=1e-14)
        self.assertAlmostEqual(self.curve.forward(1, 2), np.expm1(0.03))

    def test_bootstrap_reprices(self):
        deposits = [
            (0.5, (1 / self.curve.df(0.5) - 1) / 0.5),
            (1, 1 / self.curve.df(1) - 1),
        ]
        swaps = [
            (
                np.arange(0.5, t + 0.01, 0.5),
                par_swap_rate(self.curve, np.arange(0.5, t + 0.01, 0.5)),
            )
            for t in np.arange(1.5, 5.01, 0.5)
        ]
        curve = bootstrap(deposits, swaps)
        for times, rate in swaps:
            self.assertAlmostEqual(swap_pv(curve, times, rate, 1e6), 0, delta=1e-8)
        np.testing.assert_allclose(
            curve.df([0.5, 1, 3, 5]), self.curve.df([0.5, 1, 3, 5]), atol=1e-14
        )

    def test_bond_and_risk(self):
        f = lambda c: bond_pv(c, [5], 0)
        ref = 100 * np.exp(-0.15)
        risk = risk_measures(f, self.curve)
        self.assertAlmostEqual(risk["pv"], ref)
        self.assertAlmostEqual(risk["dollar_duration"], 5 * ref, delta=2e-5)
        self.assertAlmostEqual(risk["dollar_convexity"], 25 * ref, delta=1e-4)
        bumped = f(self.curve.shifted(0.001))
        approximation = (
            ref
            - risk["dollar_duration"] * 0.001
            + 0.5 * risk["dollar_convexity"] * 0.001**2
        )
        self.assertLess(abs(bumped - approximation), 2e-6)

    def test_negative_rates(self):
        c = bootstrap([(1, -0.01)], [([1, 2], -0.005)])
        self.assertGreater(c.df(1), 1)
        self.assertAlmostEqual(swap_pv(c, [1, 2], -0.005), 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            self.curve.df(6)
        with self.assertRaises(ValueError):
            DiscountCurve((0, 1), (1, -1))
        with self.assertRaises(ValueError):
            bond_pv(self.curve, [1, 0.5], 0.03)
        with self.assertRaises(ValueError):
            bootstrap([(1, 0.03)], [([1, 2, 3], 0.04)])

    def test_swap_direction(self):
        rate = par_swap_rate(self.curve, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(swap_pv(self.curve, [1, 2, 3, 4, 5], rate), 0)
        self.assertGreater(swap_pv(self.curve.shifted(0.001), [1, 2, 3, 4, 5], rate), 0)
