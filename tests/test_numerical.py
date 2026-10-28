import unittest
import numpy as np
from quantlab.black_scholes import price
from quantlab.numerical import crr, monte_carlo, finite_difference, delta_hedge
from quantlab.heston import Heston, heston_price, calibrate, surface_diagnostics
from quantlab.exotics import geometric_asian, continuous_up_out_call


class NumericalTests(unittest.TestCase):
    def test_tree_convergence(self):
        for kind in ("call", "put"):
            ref = price(100, 100, 1, 0.2, 0.05, 0.01, kind)
            self.assertAlmostEqual(
                crr(100, 100, 1, 0.2, 0.05, 0.01, kind, steps=800), ref, delta=0.004
            )

    def test_american(self):
        european = crr(100, 100, 1, 0.2, 0.05, kind="put", steps=400)
        american = crr(100, 100, 1, 0.2, 0.05, kind="put", steps=400, american=True)
        self.assertGreater(american, european)
        self.assertAlmostEqual(
            crr(100, 100, 1, 0.2, 0.05, steps=400, american=True),
            crr(100, 100, 1, 0.2, 0.05, steps=400),
            places=10,
        )
        self.assertGreaterEqual(crr(80, 100, 1, 0, 0.05, kind="put", american=True), 20)

    def test_invalid_tree(self):
        with self.assertRaises(ValueError):
            crr(100, 100, 1, 0.01, 1, steps=1)
        with self.assertRaises(ValueError):
            crr(100, 100, 1, 0.2, steps=0)

    def test_mc_statistics(self):
        ref = price(100, 100, 1, 0.2, 0.05)
        mc = monte_carlo(100, 100, 1, 0.2, 0.05, samples=20000, steps=1, control=True)
        # Fixed seed regression plus conservative statistical tolerance; not CI coverage proof.
        self.assertLess(abs(mc.value - ref), 4 * mc.standard_error)
        raw = monte_carlo(100, 100, 1, 0.2, 0.05, samples=20000, steps=1, control=False)
        self.assertLess(mc.standard_error, raw.standard_error)
        self.assertEqual(mc.independent_samples, 20000)
        self.assertEqual(
            mc,
            monte_carlo(100, 100, 1, 0.2, 0.05, samples=20000, steps=1, control=True),
        )

    def test_exotics(self):
        euro = monte_carlo(100, 100, 1, 0.2, 0.05, samples=10000)
        asian = monte_carlo(100, 100, 1, 0.2, 0.05, samples=10000, payoff="asian")
        barrier = monte_carlo(
            100, 100, 1, 0.2, 0.05, samples=10000, payoff="up-and-out", barrier=130
        )
        self.assertLess(asian.value, euro.value)
        self.assertLess(barrier.value, euro.value)
        self.assertEqual(
            monte_carlo(100, 100, 1, 0.2, payoff="up-and-out", barrier=99).value, 0
        )
        self.assertEqual(
            monte_carlo(100, 100, 1, 0.2, steps=1, payoff="asian"),
            monte_carlo(100, 100, 1, 0.2, steps=1),
        )

    def test_pde(self):
        for kind in ("call", "put"):
            ref = price(100, 100, 1, 0.2, 0.05, 0.01, kind)
            fine = finite_difference(
                100, 100, 1, 0.2, 0.05, 0.01, kind, space_steps=800, time_steps=400
            )
            coarse = finite_difference(
                100, 100, 1, 0.2, 0.05, 0.01, kind, space_steps=100, time_steps=100
            )
            self.assertLess(abs(fine - ref), abs(coarse - ref))
            self.assertAlmostEqual(fine, ref, delta=0.002)
            self.assertAlmostEqual(
                finite_difference(100, 100, 1, 0.2, 0.05, 0.01, kind, theta=1),
                ref,
                delta=0.03,
            )
        with self.assertRaises(ValueError):
            finite_difference(
                100, 100, 1, 0.2, 0.05, theta=0, space_steps=100, time_steps=1
            )
        self.assertAlmostEqual(
            finite_difference(
                100, 100, 1, 0.2, 0.03, theta=0, space_steps=100, time_steps=2000
            ),
            price(100, 100, 1, 0.2, 0.03),
            delta=0.05,
        )

    def test_exotic_independent_benchmarks(self):
        est = monte_carlo(
            100,
            100,
            1,
            0.2,
            0.05,
            payoff="geometric-asian",
            samples=20000,
            control=True,
        )
        ref = geometric_asian(100, 100, 1, 0.2, 0.05)
        self.assertLess(abs(est.value - ref), 4 * est.standard_error)
        self.assertAlmostEqual(
            geometric_asian(100, 100, 1, 0.2, 0.05, steps=1),
            price(100, 100, 1, 0.2, 0.05),
            places=10,
        )
        continuous = continuous_up_out_call(100, 100, 1, 0.2, 130, 0.05)
        discrete = monte_carlo(
            100,
            100,
            1,
            0.2,
            0.05,
            payoff="up-and-out",
            barrier=130,
            samples=20000,
            steps=256,
        )
        self.assertGreater(discrete.value + 4 * discrete.standard_error, continuous)
        self.assertLess(continuous, price(100, 100, 1, 0.2, 0.05))
        self.assertEqual(continuous_up_out_call(100, 100, 1, 0.2, 100), 0)

    def test_hedging(self):
        coarse = delta_hedge(100, 100, 1, 0.2, 0.2, paths=6000, steps=12)
        fine = delta_hedge(100, 100, 1, 0.2, 0.2, paths=6000, steps=252)
        costly = delta_hedge(100, 100, 1, 0.2, 0.2, paths=6000, steps=252, cost_bps=10)
        self.assertLess(fine.std(), coarse.std())
        self.assertLess(abs(fine.mean()), 0.06)
        self.assertTrue(np.all(costly <= fine + 1e-12))

    def test_heston_limits(self):
        model = Heston(0.04, 2, 0.04, 0.001, -0.5)
        self.assertAlmostEqual(
            heston_price(100, 100, 1, model, 0.03),
            price(100, 100, 1, 0.2, 0.03),
            delta=0.002,
        )
        m = Heston()
        call = heston_price(100, 100, 1, m, 0.03)
        put = heston_price(100, 100, 1, m, 0.03, kind="put")
        self.assertAlmostEqual(call - put, 100 - 100 * np.exp(-0.03), places=9)
        self.assertAlmostEqual(
            call, heston_price(100, 100, 1, m, 0.03, cutoff=250), delta=1e-6
        )

    def test_calibration(self):
        m = Heston()
        k = np.tile([85, 100, 115], 3)
        t = np.repeat([0.5, 1, 2], 3)
        prices = np.array([heston_price(100, a, b, m, 0.03) for a, b in zip(k, t)])
        fit, result = calibrate(100, k, t, prices, rate=0.03, max_nfev=100)
        self.assertTrue(result.success)
        self.assertLess(np.sqrt(np.mean(result.fun**2)), 1e-5)
        self.assertLess(
            abs(
                heston_price(100, 105, 1.5, fit, 0.03)
                - heston_price(100, 105, 1.5, m, 0.03)
            ),
            1e-4,
        )

    def test_surface(self):
        k = np.linspace(70, 130, 13)
        t = np.array([0.5, 1, 2])
        c = price(100, k[None, :], t[:, None], 0.2)
        self.assertTrue(all(surface_diagnostics(k, t, c, 100).values()))
        c[1, 5] = c[1, 4] + 10
        self.assertFalse(surface_diagnostics(k, t, c, 100)["decreasing_strike"])
