"""Portable reconstruction of the historical LAMOST FGK fitting workflow.
The original dataset was not recovered. This script generates fresh results only
from user-supplied observations; archived numbers are not hard-coded into fits.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fits", type=Path)
    p.add_argument("--mass", default="Mass")
    p.add_argument("--lum", default="Lum")
    p.add_argument("--teff", default="Teff")
    p.add_argument("--logg", default="logg")
    p.add_argument("--logg-min", type=float, default=4.1)
    p.add_argument("--prefiltered", action="store_true")
    p.add_argument("--out", type=Path, default=Path("output"))
    a = p.parse_args()
    t = Table.read(a.fits)
    names = {n.lower(): n for n in t.colnames}
    def read_col(name):
        return np.ma.asarray(t[names[name.lower()]], dtype=float).filled(np.nan)
    m, l = read_col(a.mass), read_col(a.lum)
    valid = np.isfinite(m) & np.isfinite(l) & (m > 0) & (m < 2.5) & (l > 0) & (l < 100)
    for name, cut in [(a.teff, lambda v: (v >= 3700) & (v <= 7500)),
                      (a.logg, lambda v: v > a.logg_min)]:
        if name.lower() in names:
            v = read_col(name)
            valid &= np.isfinite(v) & cut(v)
        elif not a.prefiltered:
            p.error(f"Missing {name}; specify column name or documented --prefiltered input")
    x, y = np.log10(m[valid]), np.log10(l[valid])
    if len(x) < 3 or np.ptp(x) == 0:
        p.error("Insufficient nondegenerate data after cuts")
    slope, intercept = np.polyfit(x, y, 1)
    yh = slope * x + intercept
    sst = float(np.sum((y-y.mean())**2))
    r2 = 1-float(np.sum((y-yh)**2))/sst if sst else None
    a.out.mkdir(parents=True, exist_ok=True)
    result = dict(input_rows=int(len(t)), selected_rows=int(len(x)), slope=float(slope),
                  intercept=float(intercept), r_squared=r2, logg_min=a.logg_min,
                  input_prefiltered=a.prefiltered)
    (a.out/"fit_results.json").write_text(json.dumps(result, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(8,6), layout="constrained")
    hb = ax.hexbin(x, y, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    xx = np.linspace(x.min(), x.max(), 150)
    ax.plot(xx, slope*xx+intercept, "#eea13b", lw=2.2, label=f"Fit exponent = {slope:.3f}")
    ax.plot(xx, 3.5*(xx-x.mean())+y.mean(), "--", color="#dbe8eb", lw=1.7, label="Reference exponent 3.5 (centred)")
    ax.set(xlabel="log₁₀(M / M☉)", ylabel="log₁₀(L / L☉)", title="FGK candidates: stellar mass–luminosity relation")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="Count per bin (log scale)")
    fig.savefig(a.out/"mass_luminosity.png", dpi=180)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
