-- Run these statements against outputs/trades.sqlite3, e.g.:
-- sqlite3 -header -column outputs/trades.sqlite3 < sql/report_queries.sql

-- Compare simulated end-of-run P&L and worst peak-to-trough drawdown.
SELECT strategy, ROUND(pnl_usd, 2) AS pnl_usd,
       ROUND(max_drawdown_usd, 2) AS max_drawdown_usd,
       fill_count, hedge_count, max_abs_option_inventory
FROM runs ORDER BY strategy;

-- Summarise simulated option fills by desk side.
SELECT r.strategy, f.side, COUNT(*) AS trades,
       ROUND(AVG(f.option_price), 3) AS average_fill_price
FROM option_fills f JOIN runs r ON r.run_id = f.run_id
GROUP BY r.strategy, f.side ORDER BY r.strategy, f.side;

-- Inventory exposure (events are marked to market every simulation step).
SELECT r.strategy, ROUND(AVG(ABS(m.option_inventory)), 3) AS mean_abs_contracts,
       ROUND(MAX(ABS(m.stock_inventory)), 0) AS peak_abs_stock_shares
FROM equity_marks m JOIN runs r ON r.run_id = m.run_id
GROUP BY r.strategy;
