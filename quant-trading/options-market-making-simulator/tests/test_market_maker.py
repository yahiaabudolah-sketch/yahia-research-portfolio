"""Run with: python -m unittest discover -s tests -v"""
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from market_maker import (Settings, black_scholes_call, generate_market,
                          max_drawdown, run_strategy, run_experiment)


class PricingTests(unittest.TestCase):
    def test_at_the_money_call_price_and_delta(self):
        price, delta = black_scholes_call(100, 100, 1, .2)
        self.assertAlmostEqual(price, 7.9655674554, places=6)
        self.assertAlmostEqual(delta, .5398278373, places=6)

    def test_delta_matches_finite_difference(self):
        x, k, t, vol = 103.0, 100.0, .35, .27
        h = 1e-3
        _, d = black_scholes_call(x, k, t, vol, .01)
        bump = (black_scholes_call(x+h, k, t, vol, .01)[0] -
                black_scholes_call(x-h, k, t, vol, .01)[0]) / (2*h)
        self.assertAlmostEqual(d, bump, places=5)

    def test_expired_call_and_inputs(self):
        self.assertEqual(black_scholes_call(105, 100, 0, .2), (5, 1))
        self.assertEqual(black_scholes_call(95, 100, 0, .2), (0, 0))
        with self.assertRaises(ValueError):
            black_scholes_call(-100, 100, 1, .2)

    def test_max_drawdown(self):
        self.assertEqual(max_drawdown([0, 3, 1, -4, 2]), 7)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = Settings(seed=17, steps=250, max_contracts=5)

    def test_reproducible_market_and_results(self):
        a, b = generate_market(self.config), generate_market(self.config)
        self.assertEqual(a, b)
        self.assertEqual(run_strategy(self.config, a)["pnl_usd"],
                         run_strategy(self.config, b)["pnl_usd"])

    def test_inventory_limit_and_flattening(self):
        result = run_strategy(self.config, generate_market(self.config))
        self.assertLessEqual(result["max_abs_option_inventory"], self.config.max_contracts)
        self.assertEqual(result["marks"][-1][2:4], (0, 0))

    def test_quote_control_changes_trade_statistics(self):
        cfg = Settings(seed=42, steps=1000, skew_per_contract=.5)
        market = generate_market(cfg)
        a = run_strategy(cfg, market, "inventory_aware")
        b = run_strategy(cfg, market, "neutral_quotes")
        self.assertNotEqual([(x["pnl_usd"], x["fill_count"]) for x in [a]],
                            [(x["pnl_usd"], x["fill_count"]) for x in [b]])

    def test_sqlite_rows_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = run_experiment(self.config, Path(tmp))
            self.assertEqual(len(data["results"]), 2)
            self.assertTrue((Path(tmp) / "summary.json").exists())
            with sqlite3.connect(Path(tmp) / "trades.sqlite3") as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 2)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM equity_marks").fetchone()[0],
                                 2 * (self.config.steps + 1))
                self.assertGreater(con.execute("SELECT COUNT(*) FROM option_fills").fetchone()[0], 0)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            generate_market(Settings(steps=1000000))
        with self.assertRaises(ValueError):
            generate_market(Settings(max_contracts=0))


if __name__ == "__main__":
    unittest.main()

class SweepTests(unittest.TestCase):
    def test_paired_sweep_reproducible(self):
        from seed_sweep import evaluate_seeds
        a, summary_a = evaluate_seeds(count=3, steps=100)
        b, summary_b = evaluate_seeds(count=3, steps=100)
        self.assertEqual(a, b)
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(summary_a["n_seed_pairs"], 3)
        self.assertGreaterEqual(summary_a["fraction_aware_lower_drawdown"], 0)
        self.assertLessEqual(summary_a["fraction_aware_lower_drawdown"], 1)
