import unittest
import numpy as np
from quantlab.risk import (
    simulate_garch,
    features,
    forecast,
    run_research,
    var_backtest,
    qlike,
)


class RiskTests(unittest.TestCase):
    def test_loss_and_window_validation(self):
        for target, prediction in [([-1], [1]), ([np.nan], [1]), ([1], [0])]:
            with self.assertRaises(ValueError):
                qlike(target, prediction)
        with self.assertRaises(ValueError):
            features(np.ones(100), window=20.5)
        with self.assertRaises(ValueError):
            run_research(n=100)

    def test_simulation(self):
        r, h = simulate_garch(1000)
        self.assertTrue(np.all(h > 0))
        np.testing.assert_array_equal(r, simulate_garch(1000)[0])
        with self.assertRaises(ValueError):
            simulate_garch(alpha=0.5, beta=0.6)

    def test_feature_causality(self):
        r, _ = simulate_garch(500)
        a = features(r)
        edited = r.copy()
        edited[300:] += 100
        b = features(edited)
        np.testing.assert_array_equal(a.x[a.origins < 300], b.x[b.origins < 300])
        np.testing.assert_array_equal(a.y, r[a.origins + 1] ** 2)
        self.assertAlmostEqual(a.x[0, 4], np.mean(r[40:60] ** 2))

    def test_forecast_causality(self):
        r, _ = simulate_garch(600)
        a = features(r)
        r[500:] *= 3
        b = features(r)
        for name in ("ridge", "boosting"):
            p = forecast(a, 300, 10.0, name, block=50)
            q = forecast(b, 300, 10.0, name, block=50)
            np.testing.assert_allclose(
                p[a.origins[300:] < 500], q[a.origins[300:] < 500], atol=1e-12
            )
            self.assertTrue(np.all(p > 0))

    def test_backtest(self):
        result = var_backtest(np.zeros(100), np.ones(100))
        self.assertEqual(result["exceptions"], 0)
        self.assertTrue(np.isfinite(result["kupiec_lr"]))
        result = var_backtest(np.full(100, -10.0), np.ones(100))
        self.assertEqual(result["exceptions"], 100)
        self.assertTrue(np.isfinite(result["kupiec_lr"]))
        with self.assertRaises(ValueError):
            var_backtest([0], [-1])

    def test_end_to_end(self):
        result, data, pred, test = run_research(n=1200)
        self.assertEqual(result["test_rows"], len(data.y) - test)
        for p in pred.values():
            self.assertEqual(p.shape, data.y[test:].shape)
            self.assertTrue(np.isfinite(qlike(data.y[test:], p)))
