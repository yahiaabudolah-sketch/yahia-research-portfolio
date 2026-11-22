"""Create a dependency-free SVG comparing mean synthetic P&L and drawdown changes."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

SCENARIOS = [
    ("baseline", "Baseline"),
    ("higher_volatility", "Higher volatility"),
    ("uptrend", "Upward trend"),
    ("informed_flow", "Informed flow"),
    ("volatile_informed_flow", "Volatile + informed"),
]


def render(report_path: Path, output: Path) -> None:
    data = json.loads(report_path.read_text())
    regimes = data["regimes"]
    chart_width = 1050
    bar_w = 265
    bar_start = 260
    right_bar = 650
    scale = bar_w / 800.0
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" height="430" viewBox="0 0 {chart_width} 430">',
        '<rect width="1050" height="430" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#173446} .desc{font-size:14px;fill:#5f7180} .name{font-size:17px;font-weight:600} .val{font-size:14px;font-weight:600}</style>',
        '<text x="28" y="36" font-size="26" font-weight="700">Synthetic scenario results</text>',
        '<text x="28" y="60" class="desc">Mean change: inventory-aware minus neutral quoting, 50 paired seeds per scenario</text>',
        '<text x="260" y="97" font-size="15" font-weight="700">P&amp;L sacrificed ($)</text>',
        '<text x="650" y="97" font-size="15" font-weight="700">Drawdown reduced ($)</text>',
    ]
    for i, (slug, label) in enumerate(SCENARIOS):
        y = 134 + i * 51
        pnl = regimes[slug]["paired_mean_pnl_delta_usd"]
        dd = regimes[slug]["paired_mean_drawdown_delta_usd"]
        s.extend([
            f'<text x="28" y="{y+15}" class="name">{escape(label)}</text>',
            f'<rect x="{bar_start}" y="{y}" width="{abs(pnl)*scale:.1f}" height="24" rx="4" fill="#b87b60"/>',
            f'<text x="{bar_start+abs(pnl)*scale+8:.1f}" y="{y+17}" class="val">${abs(pnl):.0f}</text>',
            f'<rect x="{right_bar}" y="{y}" width="{abs(dd)*scale:.1f}" height="24" rx="4" fill="#2d7d73"/>',
            f'<text x="{right_bar+abs(dd)*scale+8:.1f}" y="{y+17}" class="val">${abs(dd):.0f}</text>',
        ])
    s.append('<text x="28" y="413" class="desc">Illustrative model only - not historical market data, live execution or investment performance.</text>')
    s.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(s) + "\n", encoding="utf-8")

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    render(base / "reports" / "stress_lab_summary.json", base / "reports" / "stress_lab_comparison.svg")
