"""Paired synthetic backtests across many random seeds (no live market data)."""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean, median
from market_maker import Settings, generate_market, run_strategy


def evaluate_seeds(count: int = 100, steps: int = 1000) -> tuple[list[dict], dict]:
    if not 1 <= count <= 10000:
        raise ValueError("count must be from 1 to 10000")
    rows = []
    for seed in range(count):
        config = Settings(seed=seed, steps=steps)
        market = generate_market(config)
        aware = run_strategy(config, market, "inventory_aware")
        neutral = run_strategy(config, market, "neutral_quotes")
        rows.append({
            "seed": seed,
            "aware_pnl_usd": aware["pnl_usd"],
            "neutral_pnl_usd": neutral["pnl_usd"],
            "pnl_difference_usd": round(aware["pnl_usd"] - neutral["pnl_usd"], 2),
            "aware_drawdown_usd": aware["max_drawdown_usd"],
            "neutral_drawdown_usd": neutral["max_drawdown_usd"],
            "drawdown_difference_usd": round(aware["max_drawdown_usd"] - neutral["max_drawdown_usd"], 2),
            "aware_max_inventory": aware["max_abs_option_inventory"],
            "neutral_max_inventory": neutral["max_abs_option_inventory"],
        })
    report = {
        "synthetic_only": True,
        "n_seed_pairs": count,
        "steps_per_seed": steps,
        "paired_pnl_difference_mean_usd": round(mean(x["pnl_difference_usd"] for x in rows), 2),
        "paired_pnl_difference_median_usd": round(median(x["pnl_difference_usd"] for x in rows), 2),
        "paired_drawdown_difference_mean_usd": round(mean(x["drawdown_difference_usd"] for x in rows), 2),
        "fraction_aware_lower_drawdown": round(sum(x["drawdown_difference_usd"] < 0 for x in rows) / count, 3),
        "fraction_aware_higher_pnl": round(sum(x["pnl_difference_usd"] > 0 for x in rows) / count, 3),
        "mean_aware_pnl_usd": round(mean(x["aware_pnl_usd"] for x in rows), 2),
        "mean_neutral_pnl_usd": round(mean(x["neutral_pnl_usd"] for x in rows), 2),
        "method": "paired common-random-number comparison, stylized execution, no real option data",
    }
    return rows, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()
    rows, summary = evaluate_seeds(args.seeds, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed_sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "seed_sweep_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
