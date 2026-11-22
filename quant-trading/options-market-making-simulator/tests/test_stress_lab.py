"""Stress-suite checks: regime construction, paired methodology and reproducibility."""
import math
from pathlib import Path
import tempfile
import unittest

from market_maker import Settings, generate_market
from stress_lab import (
    REGIMES, SENSITIVITIES, market_in_regime, mean_fill_markout_usd,
    paired_trials, bootstrap_mean_ci, summarize, run_lab,
)

class StressLabTests(unittest.TestCase):
    def test_default_regime_matches_existing_generator_exactly(self):
        c = Settings(seed=9, steps=80)
        self.assertEqual(market_in_regime(c, "baseline"), generate_market(c))

    def test_all_scenario_paths_share_original_random_uniforms(self):
        c = Settings(seed=7, steps=90)
        a = market_in_regime(c, "baseline")
        for name in REGIMES:
            b = market_in_regime(c, name)
            self.assertEqual([s.arrival_uniform for s in a], [s.arrival_uniform for s in b])
            self.assertEqual([s.fill_uniform for s in a], [s.fill_uniform for s in b])
            self.assertTrue(all(math.isfinite(s.spot) and s.spot > 0 for s in b))

    def test_volatile_scenario_more_extreme_than_baseline(self):
        c = Settings(seed=12, steps=80)
        x = market_in_regime(c, "baseline")
        y = market_in_regime(c, "higher_volatility")
        self.assertGreater(y[0].fair_volatility, x[0].fair_volatility)
        self.assertGreater(abs(math.log(y[-1].spot / c.initial_spot)),
                           abs(math.log(x[-1].spot / c.initial_spot)))

    def test_informed_stress_changes_some_customer_directions(self):
        c = Settings(seed=3, steps=100)
        a = market_in_regime(c, "baseline")
        b = market_in_regime(c, "informed_flow")
        self.assertTrue(any(x.direction != y.direction for x, y in zip(a, b)))

    def test_paired_results_are_reproducible(self):
        a = paired_trials(3, 85, "informed_flow")
        b = paired_trials(3, 85, "informed_flow")
        self.assertEqual(a, b)
        self.assertEqual(summarize(a), summarize(b))
        self.assertEqual(len(a), 3)

    def test_ci_constant_values(self):
        self.assertEqual(bootstrap_mean_ci([4] * 20), (4, 4))
        with self.assertRaises(ValueError):
            bootstrap_mean_ci([])

    def test_invalid_regime_and_markout(self):
        with self.assertRaises(ValueError):
            market_in_regime(Settings(steps=50), "never")
        with self.assertRaises(ValueError):
            mean_fill_markout_usd({"fills": []}, [], Settings(), horizon=0)

    def test_breakeven_risk_aversion_equation(self):
        stats = summarize(paired_trials(6, 100, "baseline"))
        dd = stats["paired_mean_drawdown_delta_usd"]
        pnl = stats["paired_mean_pnl_delta_usd"]
        if dd != 0:
            self.assertAlmostEqual(stats["break_even_drawdown_penalty_lambda"], pnl / dd, delta=0.002)

    def test_aggregate_report_contains_all_regimes_and_variants(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_lab(stress_seeds=2, sensitivity_seeds=2, steps=60, output_dir=Path(d))
            self.assertEqual(set(r["regimes"]), set(REGIMES))
            self.assertEqual(set(r["sensitivity"]), set(SENSITIVITIES))
            self.assertTrue((Path(d) / "stress_lab_seed_detail.csv").exists())
            self.assertTrue((Path(d) / "stress_lab_summary.json").exists())

if __name__ == "__main__":
    unittest.main()
