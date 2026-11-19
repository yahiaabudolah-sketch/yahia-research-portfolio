# The Economics of the New Space Race

AI-assisted analytical prototype based on annual 10-K statements. Open `Space_Economics_Model.xlsx` and start at Summary. Edit Assumptions to see nine scenarios recalculate. Blue numbers are inputs; cash gap is hypothetical. `python analyze.py` reproduces the original assumptions and charts (Python 3, pandas, matplotlib). Input CSVs contain source URLs. Figures in USD millions.

## Source and scope
Rocket Lab 2025 10-K, consolidated statements and cash/securities note: https://www.sec.gov/Archives/edgar/data/1819994/000181999426000013/rklb-20251231.htm
Planet Labs FY2026 10-K, consolidated statements, CapEx definition and cash flow/deferred revenue changes: https://www.sec.gov/Archives/edgar/data/1836833/000119312526119957/pl-20260131.htm
Intuitive Machines 2025 10-K: https://investors.intuitivemachines.com/static-files/d756f897-e69a-456c-b18f-e51788effa55
Intuitive Machines 2024 10-K: https://investors.intuitivemachines.com/static-files/5535cf83-9896-4c04-bab9-fd4dc74699a2
Lanteris transaction 8-K (January 2026): https://www.sec.gov/Archives/edgar/data/1844452/000119312526011200/d89967d8k.htm
All sources consulted 19 November 2026. Scope is annual FY2023–25 (Planet FY2024–26), not a current valuation or live credit rating. Reported OCF and CapEx are cash measures. Planet CapEx includes internal-use software; LUNR excludes additional intangible purchases and acquisitions. LUNR FY2023 financing inflow rounded to $53.9m. Forecasts use illustrative assumptions, not management guidance; no debt schedule, tax model, financing costs, acquired earnings or dilution. LUNR's $450m pre-adjustment acquisition cash commitment is a deliberately simplified stress overlay, not post-close cash or consolidated forecast.

## Interview prep
Re-run calculations: Rocket Lab 2025 FCF = -165.521 - 156.285 = -321.806m. Planet FCF = 134.362 - 81.497 = 52.865m. If the $151.054m deferred revenue increase is removed as a collection-timing sensitivity, FCF = -98.189m; this is not GAAP and does not mean cash was fake. Changing OCF margin/capex/revenue assumptions affects reserves and external funding needs. Independently change one parameter and explain why the result changes; distinguish cash flow from profit. Portfolio should disclose AI assistance and your own verified contribution.
