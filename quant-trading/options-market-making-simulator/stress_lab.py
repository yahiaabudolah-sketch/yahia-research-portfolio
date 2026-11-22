"""Synthetic stress, sensitivity and fill mark-out experiments.

No historical prices or order books are used. All market regimes and fills are invented.
Run the same two strategies on matched paths; vary exactly one dimension at a time.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any

from market_maker import (
    CONTRACT_MULTIPLIER, Settings, MarketStep, black_scholes_call,
    generate_market, run_strategy,
)

REGIMES = ("baseline", "higher_volatility", "uptrend", "informed_flow", "volatile_informed_flow")
SENSITIVITIES = {
    "baseline": {},
    "spread_tight_0.12": {"half_spread": 0.12},
    "spread_wide_0.32": {"half_spread": 0.32},
    "skew_low_0.015": {"skew_per_contract": 0.015},
    "skew_high_0.090": {"skew_per_contract": 0.090},
    "hedge_every_step": {"hedge_every": 1},
    "hedge_every_20": {"hedge_every": 20},
    "inventory_limit_6": {"max_contracts": 6},
    "inventory_limit_18": {"max_contracts": 18},
}


def market_in_regime(settings: Settings, regime: str) -> list[MarketStep]:
    """Stylise the SAME base random path, keeping exogenous order/fill uniforms.

    'informed_flow' deliberately uses NEXT-step simulated return as a proxy for
    an informed counterparty's private signal. This is a stress assumption, not
    a historically estimated market microstructure model.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    base = generate_market(settings)
    if regime == "baseline":
        return base  # Preserve original baseline results bit-for-bit.
    volatile = regime in ("higher_volatility", "volatile_informed_flow")
    trending = regime == "uptrend"
    informed = regime in ("informed_flow", "volatile_informed_flow")
    out: list[MarketStep] = []
    for step in base:
        # Scale cumulative log returns and the volatility surface consistently.
        ret = math.log(step.spot / settings.initial_spot)
        spot = settings.initial_spot * math.exp((1.65 if volatile else 1.0) * ret)
        if trending:
            spot *= math.exp(0.65 * step.idx / settings.steps_per_year)
        scale = 1.65 if volatile else 1.0
        out.append(replace(step,
            spot=spot,
            fair_volatility=min(0.95, step.fair_volatility * scale),
            estimated_volatility=min(0.95, step.estimated_volatility * scale)))
    if informed:
        rng = random.Random(8_700_000 + settings.seed)
        # 80% of incoming orders point in the direction of the *next* return.
        # Intentional adverse selection: stress test only, NOT a forecast.
        for i in range(len(out) - 1):
            if rng.random() < 0.80:
                future_return = out[i+1].spot - out[i].spot
                out[i] = replace(out[i], direction=(1 if future_return >= 0 else -1))
    return out


def mean_fill_markout_usd(results: dict[str, Any], market: list[MarketStep],
                          settings: Settings, horizon: int = 5) -> float | None:
    """Mean market-maker option trade mark-out, ignoring hedges/fees.

    Positive: a maker's execution beat the future synthetic fair mid; this is
    NOT full portfolio P&L or a reliably estimated fill-quality statistic.
    """
    if horizon < 1:
        raise ValueError("mark-out horizon must be positive")
    markouts: list[float] = []
    for idx, side, traded_price, _ in results["fills"]:
        if idx + horizon >= len(market):
            continue
        step = market[idx + horizon]
        tau = settings.maturity_years - step.idx / settings.steps_per_year
        future_fair, _ = black_scholes_call(step.spot, settings.strike, tau,
                                           step.fair_volatility, settings.rate)
        sign = -1 if side == "sell_call" else 1
        markouts.append(sign * (future_fair - traded_price) * CONTRACT_MULTIPLIER)
    return round(mean(markouts), 2) if markouts else None


def paired_trials(seeds: int, steps: int, regime: str,
                  overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not 1 <= seeds <= 10_000:
        raise ValueError("seeds must be in 1..10000")
    if regime not in REGIMES:
        raise ValueError("unrecognised regime")
    rows: list[dict[str, Any]] = []
    for seed in range(seeds):
        config = Settings(seed=seed, steps=steps, **(overrides or {}))
        market = market_in_regime(config, regime)
        aware = run_strategy(config, market, "inventory_aware")
        neutral = run_strategy(config, market, "neutral_quotes")
        rows.append({
            "seed": seed,
            "aware_pnl_usd": aware["pnl_usd"],
            "neutral_pnl_usd": neutral["pnl_usd"],
            "paired_pnl_difference_usd": round(aware["pnl_usd"] - neutral["pnl_usd"], 2),
            "aware_drawdown_usd": aware["max_drawdown_usd"],
            "neutral_drawdown_usd": neutral["max_drawdown_usd"],
            "paired_drawdown_difference_usd": round(aware["max_drawdown_usd"] - neutral["max_drawdown_usd"], 2),
            "aware_fill_count": aware["fill_count"],
            "neutral_fill_count": neutral["fill_count"],
            "aware_mean_abs_inventory": aware["mean_abs_option_inventory"],
            "neutral_mean_abs_inventory": neutral["mean_abs_option_inventory"],
            "aware_mean_5step_markout_usd": mean_fill_markout_usd(aware, market, config),
            "neutral_mean_5step_markout_usd": mean_fill_markout_usd(neutral, market, config),
        })
    return rows


def bootstrap_mean_ci(values: list[float], seed: int = 6027,
                      draws: int = 1500) -> tuple[float, float]:
    """Percentile CI for seed-level mean; conditional on stylised DGP."""
    if not values or draws < 100:
        raise ValueError("need values and >=100 resamples")
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return (round(boot[int(0.025 * (draws - 1))], 2),
            round(boot[int(0.975 * (draws - 1))], 2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty results")
    pnl_diffs = [x["paired_pnl_difference_usd"] for x in rows]
    dd_diffs = [x["paired_drawdown_difference_usd"] for x in rows]
    aware_marks = [x["aware_mean_5step_markout_usd"] for x in rows
                   if x["aware_mean_5step_markout_usd"] is not None]
    neutral_marks = [x["neutral_mean_5step_markout_usd"] for x in rows
                     if x["neutral_mean_5step_markout_usd"] is not None]
    pnl_delta = mean(pnl_diffs)
    dd_delta = mean(dd_diffs)
    # If an analyst uses U=mean(P&L)-lambda*mean(max_drawdown),
    # the break-even lambda for inventory-aware quoting is -pnl_delta/-dd_delta.
    # This is a hypothetical preference threshold, not a calibrated risk metric.
    risk_threshold = round(pnl_delta / dd_delta, 3) if dd_delta != 0 else None
    return {
        "n_paired_paths": len(rows),
        "break_even_drawdown_penalty_lambda": risk_threshold,
        "aware_mean_pnl_usd": round(mean(x["aware_pnl_usd"] for x in rows), 2),
        "neutral_mean_pnl_usd": round(mean(x["neutral_pnl_usd"] for x in rows), 2),
        "paired_mean_pnl_delta_usd": round(mean(pnl_diffs), 2),
        "pnl_delta_bootstrap_95ci_usd": bootstrap_mean_ci(pnl_diffs),
        "paired_mean_drawdown_delta_usd": round(mean(dd_diffs), 2),
        "drawdown_delta_bootstrap_95ci_usd": bootstrap_mean_ci(dd_diffs, seed=6901),
        "fraction_aware_lower_drawdown": round(sum(x < 0 for x in dd_diffs) / len(dd_diffs), 3),
        "fraction_aware_higher_pnl": round(sum(x > 0 for x in pnl_diffs) / len(pnl_diffs), 3),
        "aware_mean_5step_fill_markout_usd": round(mean(aware_marks), 2) if aware_marks else None,
        "neutral_mean_5step_fill_markout_usd": round(mean(neutral_marks), 2) if neutral_marks else None,
        "aware_mean_abs_contracts": round(mean(x["aware_mean_abs_inventory"] for x in rows), 3),
        "neutral_mean_abs_contracts": round(mean(x["neutral_mean_abs_inventory"] for x in rows), 3),
    }


def run_lab(stress_seeds: int = 50, sensitivity_seeds: int = 30,
            steps: int = 1000, output_dir: Path = Path("outputs")) -> dict[str, Any]:
    report: dict[str, Any] = {
        "experiment": "Synthetic options market-making stress and one-variable-at-a-time sensitivity",
        "synthetic_only": True,
        "steps_per_path": steps,
        "stress_seeds_per_regime": stress_seeds,
        "sensitivity_seeds_per_variant": sensitivity_seeds,
        "paired_comparison": True,
        "confidence_intervals": "Seed-level percentile bootstrap; conditional on simulated model, not live data",
        "regimes": {},
        "sensitivity": {},
    }
    rows_csv = []
    for regime in REGIMES:
        rows = paired_trials(stress_seeds, steps, regime)
        report["regimes"][regime] = summarize(rows)
        rows_csv.extend({"group": "regime", "variant": regime, **x} for x in rows)
    for name, overrides in SENSITIVITIES.items():
        rows = paired_trials(sensitivity_seeds, steps, "baseline", overrides)
        report["sensitivity"][name] = {"parameter_overrides": overrides, **summarize(rows)}
        rows_csv.extend({"group": "sensitivity", "variant": name, **x} for x in rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stress_lab_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "stress_lab_seed_detail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stress-seeds", type=int, default=50)
    p.add_argument("--sensitivity-seeds", type=int, default=30)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()
    report = run_lab(args.stress_seeds, args.sensitivity_seeds, args.steps, args.output_dir)
    for name, outcome in report["regimes"].items():
        print(f"{name:<24} P&L delta={outcome['paired_mean_pnl_delta_usd']:>9.2f} "
              f"drawdown delta={outcome['paired_mean_drawdown_delta_usd']:>9.2f} "
              f"aware lower drawdown={outcome['fraction_aware_lower_drawdown']:.1%}")
    print(f"Full report: {args.output_dir / 'stress_lab_summary.json'}")

if __name__ == "__main__":
    main()
