"""Reproducible, synthetic option market-making experiment (educational research)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sqlite3
from statistics import mean, pstdev
from typing import Any

TRADING_DAYS = 252
CONTRACT_MULTIPLIER = 100


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, maturity: float,
                       volatility: float, rate: float = 0.0) -> tuple[float, float]:
    """European call premium and delta per underlying share; no dividends."""
    if spot <= 0 or strike <= 0 or maturity < 0 or volatility < 0:
        raise ValueError("spot and strike must be positive; time and volatility nonnegative")
    if maturity == 0 or volatility == 0:
        forward_intrinsic = max(spot - strike * math.exp(-rate * maturity), 0.0)
        delta = 1.0 if spot > strike * math.exp(-rate * maturity) else 0.0
        return forward_intrinsic, delta
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    price = spot * normal_cdf(d1) - strike * math.exp(-rate * maturity) * normal_cdf(d2)
    return price, normal_cdf(d1)


@dataclass(frozen=True)
class Settings:
    steps: int = 1000
    seed: int = 42
    initial_spot: float = 100.0
    strike: float = 100.0
    initial_volatility: float = 0.24
    maturity_years: float = 0.5
    rate: float = 0.02
    steps_per_year: int = 2520
    half_spread: float = 0.20
    skew_per_contract: float = 0.045
    max_contracts: int = 12
    hedge_every: int = 5
    hedge_threshold_shares: int = 20
    option_fee_per_contract: float = 0.45
    stock_fee_per_share: float = 0.005
    stock_slippage_per_share: float = 0.015
    option_exit_slippage: float = 0.12
    arrival_probability: float = 0.22
    fill_distance_sensitivity: float = 3.5

    def validate(self) -> None:
        if self.steps < 2 or self.steps_per_year <= 0 or self.steps >= self.maturity_years * self.steps_per_year:
            raise ValueError("steps must be >=2 and finish before option maturity")
        if self.max_contracts < 1 or self.hedge_every < 1 or self.hedge_threshold_shares < 0:
            raise ValueError("invalid risk or hedge controls")
        for name in ("initial_spot", "strike", "initial_volatility", "maturity_years"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("half_spread", "skew_per_contract", "option_fee_per_contract", "stock_fee_per_share", "stock_slippage_per_share", "option_exit_slippage", "fill_distance_sensitivity"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.arrival_probability <= 1:
            raise ValueError("arrival_probability must be in [0,1]")


@dataclass(frozen=True)
class MarketStep:
    idx: int
    spot: float
    fair_volatility: float
    estimated_volatility: float
    direction: int  # +1 = incoming customer buys calls; -1 = incoming customer sells
    arrival_uniform: float
    fill_uniform: float


def generate_market(config: Settings) -> list[MarketStep]:
    """Synthetic stochastic-volatility path and common random order-flow draws."""
    config.validate()
    rng = random.Random(config.seed)
    dt = 1.0 / config.steps_per_year
    spot = config.initial_spot
    logvol = math.log(config.initial_volatility)
    estimate = config.initial_volatility
    steps: list[MarketStep] = []
    shock_prev = 0.0
    for idx in range(config.steps):
        shock = rng.gauss(0.0, 1.0)
        vol_noise = rng.gauss(0.0, 1.0)
        # Mean-reverting latent log-volatility, not an observed exchange quotation.
        logvol += 8.0 * (math.log(config.initial_volatility) - logvol) * dt + 0.7 * math.sqrt(dt) * vol_noise
        fair_vol = min(0.65, max(0.08, math.exp(logvol)))
        spot *= math.exp(-0.5 * fair_vol**2 * dt + fair_vol * math.sqrt(dt) * shock)
        # Deliberately imperfect estimator: observable proxy reacts with a lag.
        observed_vol_proxy = min(0.65, max(0.08, fair_vol * (1.0 + 0.06 * shock_prev)))
        estimate = 0.985 * estimate + 0.015 * observed_vol_proxy
        # More customer call buying after positive moves => mild adverse selection.
        p_buy = 1.0 / (1.0 + math.exp(-0.65 * shock))
        direction = 1 if rng.random() < p_buy else -1
        steps.append(MarketStep(idx, spot, fair_vol, estimate, direction,
                                rng.random(), rng.random()))
        shock_prev = shock
    return steps


def max_drawdown(equity: list[float]) -> float:
    peak = float('-inf')
    largest = 0.0
    for value in equity:
        peak = max(peak, value)
        largest = max(largest, peak - value)
    return largest


def _fill_probability(quote: float, fair_mid: float, direction: int,
                      sensitivity: float) -> float:
    # A customer buying lifts the ask; customer selling hits the bid.
    # Attractive prices improve fill odds; both informed flow and quote distance matter.
    disadvantage = (quote - fair_mid) if direction == 1 else (fair_mid - quote)
    return min(0.92, 0.42 * math.exp(-sensitivity * max(-0.16, disadvantage)))


def run_strategy(config: Settings, market: list[MarketStep],
                 strategy: str = "inventory_aware") -> dict[str, Any]:
    if strategy not in ("inventory_aware", "neutral_quotes"):
        raise ValueError("strategy must be inventory_aware or neutral_quotes")
    cash = 0.0
    contracts = 0
    shares = 0
    option_fees = 0.0
    stock_costs = 0.0
    fills: list[tuple[int, str, float, int]] = []
    hedges: list[tuple[int, int, float]] = []
    marks: list[tuple[int, float, int, int, float]] = []
    max_abs_inventory = 0
    inventory_samples: list[int] = []
    for step in market:
        tau = config.maturity_years - step.idx / config.steps_per_year
        model_mid, model_delta = black_scholes_call(step.spot, config.strike, tau,
                                                     step.estimated_volatility, config.rate)
        fair_mid, fair_delta = black_scholes_call(step.spot, config.strike, tau,
                                                   step.fair_volatility, config.rate)
        skew = (-config.skew_per_contract * contracts) if strategy == "inventory_aware" else 0.0
        bid = max(0.01, model_mid + skew - config.half_spread)
        ask = max(bid + 0.01, model_mid + skew + config.half_spread)
        if step.arrival_uniform < config.arrival_probability:
            direction = step.direction
            quote = ask if direction == 1 else bid
            probability = _fill_probability(quote, fair_mid, direction,
                                            config.fill_distance_sensitivity)
            target_inventory = contracts - direction
            if abs(target_inventory) <= config.max_contracts and step.fill_uniform < probability:
                contracts = target_inventory
                cash += direction * quote * CONTRACT_MULTIPLIER
                cash -= config.option_fee_per_contract
                option_fees += config.option_fee_per_contract
                fills.append((step.idx, "sell_call" if direction == 1 else "buy_call", quote, contracts))
        if step.idx % config.hedge_every == 0:
            target_shares = round(-contracts * model_delta * CONTRACT_MULTIPLIER)
            size = target_shares - shares
            if abs(size) >= config.hedge_threshold_shares:
                cash -= size * step.spot
                cost = abs(size) * (config.stock_fee_per_share + config.stock_slippage_per_share)
                cash -= cost
                stock_costs += cost
                shares = target_shares
                hedges.append((step.idx, size, step.spot))
        equity = cash + shares * step.spot + contracts * fair_mid * CONTRACT_MULTIPLIER
        marks.append((step.idx, equity, contracts, shares, fair_delta))
        inventory_samples.append(abs(contracts))
        max_abs_inventory = max(max_abs_inventory, abs(contracts))

    # Liquidate both legs at the end; crossing spread is a real cash cost.
    last = market[-1]
    tau = config.maturity_years - last.idx / config.steps_per_year
    fair_mid, _ = black_scholes_call(last.spot, config.strike, tau,
                                      last.fair_volatility, config.rate)
    cash += contracts * fair_mid * CONTRACT_MULTIPLIER
    cash -= abs(contracts) * (config.option_exit_slippage * CONTRACT_MULTIPLIER + config.option_fee_per_contract)
    cash += shares * last.spot
    cash -= abs(shares) * (config.stock_fee_per_share + config.stock_slippage_per_share)
    terminal_pnl = cash
    marks.append((len(market), terminal_pnl, 0, 0, 0.0))
    values = [v[1] for v in marks]
    mark_changes = [b - a for a, b in zip(values, values[1:])]
    return {
        "strategy": strategy,
        "pnl_usd": round(terminal_pnl, 2),
        "max_drawdown_usd": round(max_drawdown(values), 2),
        "mark_change_std_usd": round(pstdev(mark_changes), 2),
        "fill_count": len(fills),
        "hedge_count": len(hedges),
        "max_abs_option_inventory": max_abs_inventory,
        "mean_abs_option_inventory": round(mean(inventory_samples), 3),
        "option_fees_usd": round(option_fees, 2),
        "stock_execution_cost_usd": round(stock_costs, 2),
        "fills": fills,
        "hedges": hedges,
        "marks": marks,
    }


SQL_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    seed INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    steps INTEGER NOT NULL,
    pnl_usd REAL NOT NULL,
    max_drawdown_usd REAL NOT NULL,
    fill_count INTEGER NOT NULL,
    hedge_count INTEGER NOT NULL,
    max_abs_option_inventory INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS option_fills (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    step INTEGER NOT NULL,
    side TEXT NOT NULL,
    option_price REAL NOT NULL,
    inventory_after INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stock_hedges (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    step INTEGER NOT NULL,
    share_change INTEGER NOT NULL,
    spot REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS equity_marks (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    step INTEGER NOT NULL,
    equity_usd REAL NOT NULL,
    option_inventory INTEGER NOT NULL,
    stock_inventory INTEGER NOT NULL,
    fair_delta REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_option_fills_run ON option_fills(run_id);
CREATE INDEX IF NOT EXISTS idx_equity_marks_run_step ON equity_marks(run_id, step);
"""


def write_sqlite(path: Path, config: Settings, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # Re-running with the same seed regenerates the database.
    with sqlite3.connect(path) as con:
        con.executescript(SQL_SCHEMA)
        for result in results:
            run_id = con.execute("""INSERT INTO runs
                (seed, strategy, steps, pnl_usd, max_drawdown_usd, fill_count,
                 hedge_count, max_abs_option_inventory) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (config.seed, result["strategy"], config.steps, result["pnl_usd"],
                 result["max_drawdown_usd"], result["fill_count"],
                 result["hedge_count"], result["max_abs_option_inventory"])).lastrowid
            con.executemany("INSERT INTO option_fills VALUES (?, ?, ?, ?, ?)",
                            [(run_id, *fill) for fill in result["fills"]])
            con.executemany("INSERT INTO stock_hedges VALUES (?, ?, ?, ?)",
                            [(run_id, *hedge) for hedge in result["hedges"]])
            con.executemany("INSERT INTO equity_marks VALUES (?, ?, ?, ?, ?, ?)",
                            [(run_id, *mark) for mark in result["marks"]])


def run_experiment(config: Settings, output_dir: Path) -> dict[str, Any]:
    market = generate_market(config)
    results = [run_strategy(config, market, mode)
               for mode in ("inventory_aware", "neutral_quotes")]
    # Same synthetic market and the same random arrivals/fill draws (common random numbers).
    clean = [{k: v for k, v in r.items() if k not in ("fills", "hedges", "marks")}
             for r in results]
    payload = {"settings": asdict(config), "synthetic_data": True,
               "comparison_method": "paired common random numbers, not a trading forecast",
               "results": clean}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_sqlite(output_dir / "trades.sqlite3", config, results)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    result = run_experiment(Settings(seed=args.seed, steps=args.steps), args.output_dir)
    print(json.dumps(result["results"], indent=2))
    print(f"SQLite event log: {args.output_dir / 'trades.sqlite3'}")


if __name__ == "__main__":
    main()
