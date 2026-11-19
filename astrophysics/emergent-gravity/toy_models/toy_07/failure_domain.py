import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 7e: FAILURE / BREAKDOWN BOUNDARY TEST
# ==============================================================
#
# Formal:
#     epsilon = max |2*Phi/c^2|
#     g_00 = -(1 + 2*Phi/c^2)
#     weak-field regime requires epsilon << 1 and -g_00 > 0
#
#     d_tau/dt_exact = sqrt(1 + 2*Phi/c^2)
#     d_tau/dt_weak  = 1 + Phi/c^2
#
#     z_exact = sqrt((-g00_obs)/(-g00_emit)) - 1
#     z_weak  = (Phi_obs - Phi_emit)/c^2
#
#     Delta t_exact = (1/c) integral [sqrt(gii/(-g00)) - 1] dl
#     Delta t_weak  = -((1 + gamma_PPN)/c^3) integral Phi dl
#
# Microsoft/plain:
#     weakness = max(abs(2*phi/c^2))
#     g00 = -(1 + 2*phi/c^2)
#     weak_field_valid means weakness << 1 and -g00 > 0
#     clock_exact = sqrt(1 + 2*phi/c^2)
#     clock_weak = 1 + phi/c^2
#     redshift_weak = (phi_observer - phi_emitter)/c^2
#     delay_exact = integral((sqrt(gii/(-g00)) - 1)*dl/c)
#     delay_weak = -(1 + gamma_PPN)*integral(phi dl)/c^3
#
# Normal:
#     Toy 7a-7d tested the weak-field linearized bridge.
#     Toy 7e deliberately pushes the model outside the weak-field regime.
#     A successful failure test should show:
#         1. weak cases behave well,
#         2. strong/compact cases break the weak-field approximations,
#         3. very strong cases can break the metric signature itself.

EPS = 1e-12
GAMMA_PPN = 1.0
GAMMA_SOURCE = 1.0

BOOST_VALUES = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160]
C_LIGHT_VALUES = [12.0, 10.0, 8.0, 6.0]
B_VALUES = [20.0, 35.0, 60.0]

X_MAX = 350.0
N_INTEGRAL = 5001

VALID_WEAKNESS_LIMIT = 0.05
BREAKDOWN_WEAKNESS_LIMIT = 0.35
ERROR_LIMIT = 0.05
SIGNATURE_MARGIN = 0.0

CASE_CSV = "toy_model_7e_failure_case_summary.csv"
DELAY_CSV = "toy_model_7e_failure_delay_samples.csv"
SUMMARY_CSV = "toy_model_7e_failure_summary.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

SOURCE_LIBRARY = {
    "weak_r6": {"alpha": 2.0, "beta": 0.60, "radius": 6.0},
    "medium_r6": {"alpha": 4.0, "beta": 0.35, "radius": 6.0},
    "strong_r6": {"alpha": 8.0, "beta": 0.25, "radius": 6.0},
    "medium_r8": {"alpha": 4.0, "beta": 0.35, "radius": 8.0},
    "strong_r8": {"alpha": 8.0, "beta": 0.25, "radius": 8.0},
}

SOURCE_CONFIGS = [
    {
        "label": "single_strong",
        "sources": [
            {"source_type": "strong_r6", "position": (0.0, 0.0, 0.0)},
        ],
    },
    {
        "label": "four_source_cluster",
        "sources": [
            {"source_type": "strong_r6", "position": (-18.0, -16.0, 0.0)},
            {"source_type": "medium_r8", "position": (20.0, -12.0, 4.0)},
            {"source_type": "medium_r6", "position": (-8.0, 24.0, -5.0)},
            {"source_type": "weak_r6", "position": (24.0, 20.0, 5.0)},
        ],
    },
]


def delta_K_excess(alpha, beta, radius):
    r_int = int(radius)
    if r_int not in COEFFS:
        raise ValueError("radius must be one of 4, 5, 6, 7, 8")

    c_beta, c_inv_beta, c_ab, c_ba = COEFFS[r_int]

    return (
        c_beta * (math.sqrt(beta) - 1.0)
        + c_inv_beta * (1.0 / math.sqrt(beta) - 1.0)
        + c_ab * (math.sqrt(alpha / beta) - 1.0)
        + c_ba * (math.sqrt(beta / alpha) - 1.0)
    )


def mu_eff(alpha, beta, radius, gamma_source=1.0):
    return gamma_source * abs(delta_K_excess(alpha, beta, radius))


def softening_from_radius(radius):
    return max(1.8, radius / 3.0)


def build_sources(config, boost):
    sources = []

    for item in config["sources"]:
        base = SOURCE_LIBRARY[item["source_type"]]
        alpha = base["alpha"]
        beta = base["beta"]
        radius = base["radius"]
        dK = delta_K_excess(alpha, beta, radius)
        mu_base = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
        soft = softening_from_radius(radius)

        sources.append({
            "source_type": item["source_type"],
            "alpha": alpha,
            "beta": beta,
            "radius": radius,
            "delta_K_excess": dK,
            "mu_eff_base": mu_base,
            "mu_eff": boost * mu_base,
            "softening": soft,
            "position": np.array(item["position"], dtype=float),
        })

    return sources


def phi_single(points, source):
    pos = source["position"]
    mu = source["mu_eff"]
    s = source["softening"]

    dx = points[:, 0] - pos[0]
    dy = points[:, 1] - pos[1]
    dz = points[:, 2] - pos[2]

    r2 = dx*dx + dy*dy + dz*dz
    return -mu / np.sqrt(r2 + s*s)


def phi_total(points, sources):
    total = np.zeros(len(points), dtype=float)
    for src in sources:
        total += phi_single(points, src)
    return total


def g00_minus(phi, c_light):
    # This is -g00 = 1 + 2*Phi/c^2.
    return 1.0 + 2.0*phi/(c_light*c_light)


def gii(phi, c_light, gamma_ppn):
    return 1.0 - 2.0*gamma_ppn*phi/(c_light*c_light)


def relative_error(exact, approx):
    return abs(exact - approx) / (abs(exact) + EPS)


def safe_sqrt(x):
    if x <= 0:
        return np.nan
    return math.sqrt(x)


def clock_and_redshift_errors(sources, c_light):
    observer = np.array([[180.0, 180.0, 180.0]], dtype=float)
    phi_obs = float(phi_total(observer, sources)[0])
    minus_g_obs = float(g00_minus(phi_obs, c_light))

    rows = []
    tau_errors = []
    redshift_errors = []
    signature_ok = minus_g_obs > 0

    for src in sources:
        # Emitter slightly outside the softened core.
        r_emit = max(1.6 * src["radius"], 2.2 * src["softening"])
        emit_point = src["position"] + np.array([r_emit, 0.0, 0.0])
        emit_point = emit_point.reshape(1, 3)

        phi_emit = float(phi_total(emit_point, sources)[0])
        minus_g_emit = float(g00_minus(phi_emit, c_light))
        signature_ok = signature_ok and (minus_g_emit > 0)

        tau_exact = safe_sqrt(minus_g_emit)
        tau_weak = 1.0 + phi_emit/(c_light*c_light)

        tau_err = np.nan
        if np.isfinite(tau_exact):
            tau_err = relative_error(tau_exact - 1.0, tau_weak - 1.0)
            tau_errors.append(tau_err)

        z_exact = np.nan
        z_weak = (phi_obs - phi_emit)/(c_light*c_light)
        z_err = np.nan

        if minus_g_emit > 0 and minus_g_obs > 0:
            z_exact = math.sqrt(minus_g_obs/minus_g_emit) - 1.0
            z_err = relative_error(z_exact, z_weak)
            redshift_errors.append(z_err)

        rows.append({
            "source_type": src["source_type"],
            "r_emit": r_emit,
            "phi_emit": phi_emit,
            "minus_g_emit": minus_g_emit,
            "tau_exact": tau_exact,
            "tau_weak": tau_weak,
            "tau_rel_error": tau_err,
            "phi_obs": phi_obs,
            "minus_g_obs": minus_g_obs,
            "redshift_exact": z_exact,
            "redshift_weak": z_weak,
            "redshift_rel_error": z_err,
        })

    max_tau_error = float(np.nanmax(tau_errors)) if tau_errors else np.inf
    max_redshift_error = float(np.nanmax(redshift_errors)) if redshift_errors else np.inf

    return {
        "signature_ok_clock_redshift": bool(signature_ok),
        "max_tau_error": max_tau_error,
        "max_redshift_error": max_redshift_error,
        "rows": rows,
    }


def delay_errors(sources, c_light, gamma_ppn):
    x = np.linspace(-X_MAX, X_MAX, N_INTEGRAL)
    rows = []
    errors = []
    signature_ok_all = True

    for b in B_VALUES:
        points = np.column_stack([
            x,
            np.full_like(x, b),
            np.zeros_like(x),
        ])

        phi = phi_total(points, sources)
        minus_g00 = g00_minus(phi, c_light)
        gii_val = gii(phi, c_light, gamma_ppn)

        signature_ok = bool(np.all(minus_g00 > 0) and np.all(gii_val > 0))
        signature_ok_all = signature_ok_all and signature_ok

        if not signature_ok:
            rows.append({
                "b": b,
                "delay_exact": np.nan,
                "delay_weak": np.nan,
                "delay_rel_error": np.inf,
                "path_signature_ok": 0,
                "min_minus_g00_path": float(np.min(minus_g00)),
            })
            errors.append(np.inf)
            continue

        integrand_exact = np.sqrt(gii_val/minus_g00) - 1.0
        delay_exact = np.trapezoid(integrand_exact, x) / c_light
        integral_phi = np.trapezoid(phi, x)
        delay_weak = -((1.0 + gamma_ppn)/(c_light**3))*integral_phi
        err = relative_error(delay_exact, delay_weak)
        errors.append(err)

        rows.append({
            "b": b,
            "delay_exact": float(delay_exact),
            "delay_weak": float(delay_weak),
            "delay_rel_error": float(err),
            "path_signature_ok": 1,
            "min_minus_g00_path": float(np.min(minus_g00)),
        })

    return {
        "signature_ok_delay": bool(signature_ok_all),
        "max_delay_error": float(np.nanmax(errors)) if errors else np.inf,
        "rows": rows,
    }


def weakness_diagnostic(sources, c_light):
    # Include exact source centers plus emitter/path-relevant sample points.
    points = []
    for src in sources:
        points.append(src["position"])
        r_emit = max(1.6 * src["radius"], 2.2 * src["softening"])
        points.append(src["position"] + np.array([r_emit, 0.0, 0.0]))

    points.append(np.array([180.0, 180.0, 180.0]))
    points = np.array(points, dtype=float)

    phi = phi_total(points, sources)
    weakness = float(np.max(np.abs(2.0*phi/(c_light*c_light))))
    min_minus_g00 = float(np.min(g00_minus(phi, c_light)))

    return weakness, min_minus_g00


def classify_regime(weakness, signature_ok):
    if not signature_ok:
        return "signature_break"
    if weakness < VALID_WEAKNESS_LIMIT:
        return "weak_valid"
    if weakness < BREAKDOWN_WEAKNESS_LIMIT:
        return "transition"
    return "breakdown"


def run_scan():
    case_rows = []
    delay_rows = []

    total = len(SOURCE_CONFIGS) * len(C_LIGHT_VALUES) * len(BOOST_VALUES)
    counter = 0

    for config in SOURCE_CONFIGS:
        for c_light in C_LIGHT_VALUES:
            for boost in BOOST_VALUES:
                counter += 1
                print("Running failure scan %d/%d: config=%s, c=%.1f, boost=%.2f" % (
                    counter, total, config["label"], c_light, boost
                ))

                sources = build_sources(config, boost)
                weakness, min_minus_g00_centers = weakness_diagnostic(sources, c_light)

                cr = clock_and_redshift_errors(sources, c_light)
                de = delay_errors(sources, c_light, GAMMA_PPN)

                signature_ok = (
                    min_minus_g00_centers > SIGNATURE_MARGIN
                    and cr["signature_ok_clock_redshift"]
                    and de["signature_ok_delay"]
                )

                max_approx_error = float(np.nanmax([
                    cr["max_tau_error"],
                    cr["max_redshift_error"],
                    de["max_delay_error"],
                ]))

                approximation_ok = bool(signature_ok and max_approx_error < ERROR_LIMIT)
                regime = classify_regime(weakness, signature_ok)

                # A successful failure test wants weak cases to stay valid,
                # and high-weakness cases to fail the weak-field approximation/signature checks.
                if regime == "weak_valid":
                    expected_behavior_pass = approximation_ok
                elif regime in ["breakdown", "signature_break"]:
                    expected_behavior_pass = not approximation_ok
                else:
                    # Transitional cases are diagnostics, not strict pass/fail.
                    expected_behavior_pass = True

                row = {
                    "config": config["label"],
                    "c_light": c_light,
                    "source_boost": boost,
                    "n_sources": len(sources),
                    "total_mu_eff": float(sum(src["mu_eff"] for src in sources)),
                    "weakness": weakness,
                    "min_minus_g00_centers": min_minus_g00_centers,
                    "signature_ok": int(signature_ok),
                    "max_tau_error": cr["max_tau_error"],
                    "max_redshift_error": cr["max_redshift_error"],
                    "max_delay_error": de["max_delay_error"],
                    "max_approximation_error": max_approx_error,
                    "approximation_ok": int(approximation_ok),
                    "regime": regime,
                    "expected_behavior_pass": int(expected_behavior_pass),
                }
                case_rows.append(row)

                for drow in de["rows"]:
                    delay_rows.append({
                        "config": config["label"],
                        "c_light": c_light,
                        "source_boost": boost,
                        "weakness": weakness,
                        "regime": regime,
                        **drow,
                    })

    return case_rows, delay_rows


def save_csv(filename, rows):
    if not rows:
        print("No rows to save:", filename)
        return

    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filename)


def make_plots(case_rows):
    weak = np.array([r["weakness"] for r in case_rows], dtype=float)
    err = np.array([r["max_approximation_error"] for r in case_rows], dtype=float)
    min_g = np.array([r["min_minus_g00_centers"] for r in case_rows], dtype=float)

    finite_err = np.where(np.isfinite(err), err, 10.0)

    plt.figure(figsize=(9, 6))
    plt.scatter(weak, finite_err, s=35)
    plt.axvline(VALID_WEAKNESS_LIMIT, linestyle="--", label="weak-field limit")
    plt.axvline(BREAKDOWN_WEAKNESS_LIMIT, linestyle="--", label="breakdown test limit")
    plt.axhline(ERROR_LIMIT, linestyle="--", label="5% approximation error")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("weakness = max |2*Phi/c^2|")
    plt.ylabel("max approximation error")
    plt.title("Toy Model 7e: weak-field approximations fail outside regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7e_error_vs_weakness.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.scatter(weak, min_g, s=35)
    plt.axvline(VALID_WEAKNESS_LIMIT, linestyle="--", label="weak-field limit")
    plt.axhline(0.0, linestyle="--", label="signature boundary: -g00=0")
    plt.xscale("log")
    plt.xlabel("weakness = max |2*Phi/c^2|")
    plt.ylabel("minimum -g00 sampled")
    plt.title("Toy Model 7e: metric signature boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7e_signature_boundary.png", dpi=200)
    plt.show()

    regimes = [r["regime"] for r in case_rows]
    labels = ["weak_valid", "transition", "breakdown", "signature_break"]
    counts = [regimes.count(label) for label in labels]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.ylabel("case count")
    plt.title("Toy Model 7e: detected regimes")
    plt.tight_layout()
    plt.savefig("toy_model_7e_regime_counts.png", dpi=200)
    plt.show()

    # Representative line plot for single strong source at c=12.
    subset = [r for r in case_rows if r["config"] == "single_strong" and abs(r["c_light"] - 12.0) < 1e-12]
    subset = sorted(subset, key=lambda row: row["source_boost"])

    boosts = [r["source_boost"] for r in subset]
    errors = [r["max_approximation_error"] if np.isfinite(r["max_approximation_error"]) else 10.0 for r in subset]
    weaknesses = [r["weakness"] for r in subset]

    plt.figure(figsize=(9, 6))
    plt.plot(boosts, errors, marker="o", label="max approximation error")
    plt.axhline(ERROR_LIMIT, linestyle="--", label="5% error threshold")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("source boost")
    plt.ylabel("max approximation error")
    plt.title("Toy Model 7e: single-source breakdown as source strength grows")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7e_single_source_breakdown.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(boosts, weaknesses, marker="o", label="weakness")
    plt.axhline(VALID_WEAKNESS_LIMIT, linestyle="--", label="weak-field limit")
    plt.axhline(BREAKDOWN_WEAKNESS_LIMIT, linestyle="--", label="breakdown test limit")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("source boost")
    plt.ylabel("weakness = max |2*Phi/c^2|")
    plt.title("Toy Model 7e: source boost drives weak-field breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7e_boost_vs_weakness.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_7e_error_vs_weakness.png")
    print("toy_model_7e_signature_boundary.png")
    print("toy_model_7e_regime_counts.png")
    print("toy_model_7e_single_source_breakdown.png")
    print("toy_model_7e_boost_vs_weakness.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 7e: FAILURE / BREAKDOWN BOUNDARY TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    epsilon = max |2*Phi/c^2|")
    print("    weak-field regime requires epsilon << 1 and -g00 = 1 + 2*Phi/c^2 > 0")
    print("    compare exact metric effects against weak-field approximations")

    print("\nMicrosoft/plain:")
    print("    weakness = max(abs(2*phi/c^2))")
    print("    minus_g00 = 1 + 2*phi/c^2")
    print("    valid if weakness < 0.05 and minus_g00 > 0")
    print("    failure if approximations exceed 5% or metric signature breaks")

    print("\nNormal:")
    print("    We deliberately push the Toy 7 bridge outside its weak-field domain.")
    print("    A good result is not that everything passes; a good result is that the model knows where it fails.")
    print("==============================================================")

    case_rows, delay_rows = run_scan()

    save_csv(CASE_CSV, case_rows)
    save_csv(DELAY_CSV, delay_rows)

    weak_cases = [r for r in case_rows if r["regime"] == "weak_valid"]
    transition_cases = [r for r in case_rows if r["regime"] == "transition"]
    breakdown_cases = [r for r in case_rows if r["regime"] == "breakdown"]
    signature_break_cases = [r for r in case_rows if r["regime"] == "signature_break"]

    weak_valid_pass = sum(r["expected_behavior_pass"] for r in weak_cases)
    weak_valid_total = len(weak_cases)

    breakdown_detect_pass = sum(r["expected_behavior_pass"] for r in breakdown_cases + signature_break_cases)
    breakdown_detect_total = len(breakdown_cases) + len(signature_break_cases)

    overall_pass = sum(r["expected_behavior_pass"] for r in case_rows)
    overall_total = len(case_rows)

    max_weakness = float(max(r["weakness"] for r in case_rows))
    max_error_finite = float(max(
        r["max_approximation_error"] for r in case_rows
        if np.isfinite(r["max_approximation_error"])
    ))
    min_minus_g00 = float(min(r["min_minus_g00_centers"] for r in case_rows))

    first_failures = [
        r for r in case_rows
        if r["regime"] in ["breakdown", "signature_break"]
    ]
    first_failures = sorted(first_failures, key=lambda row: row["weakness"])
    first_failure = first_failures[0] if first_failures else None

    final = {
        "total_cases": len(case_rows),
        "weak_valid_cases": weak_valid_total,
        "transition_cases": len(transition_cases),
        "breakdown_cases": len(breakdown_cases),
        "signature_break_cases": len(signature_break_cases),
        "weak_valid_pass": weak_valid_pass,
        "breakdown_detect_pass": breakdown_detect_pass,
        "breakdown_detect_total": breakdown_detect_total,
        "overall_expected_behavior_pass": overall_pass,
        "overall_total": overall_total,
        "max_weakness": max_weakness,
        "max_finite_approximation_error": max_error_finite,
        "min_minus_g00_sampled": min_minus_g00,
        "first_failure_config": first_failure["config"] if first_failure else "none",
        "first_failure_c_light": first_failure["c_light"] if first_failure else np.nan,
        "first_failure_boost": first_failure["source_boost"] if first_failure else np.nan,
        "first_failure_weakness": first_failure["weakness"] if first_failure else np.nan,
        "first_failure_regime": first_failure["regime"] if first_failure else "none",
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 7e SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    epsilon = max |2*Phi/c^2|")
    print("    weak-field bridge valid only when epsilon << 1")
    print("    g00 signature condition: 1 + 2*Phi/c^2 > 0")
    print("    weak approximations should fail as epsilon grows")

    print("\nMicrosoft/plain:")
    print("    weakness = max(abs(2*phi/c^2))")
    print("    weak_valid if weakness < 0.05")
    print("    breakdown if weakness >= 0.35 or minus_g00 <= 0")
    print("    failure_detected if exact-vs-weak error > 5% or signature breaks")

    print("\nNormal:")
    print("    This is a controlled failure test. We want the model to behave well in-domain")
    print("    and visibly fail outside the weak-field regime.")

    print("\nCases:")
    print("    total cases =", len(case_rows))
    print("    weak_valid cases =", weak_valid_total)
    print("    transition cases =", len(transition_cases))
    print("    breakdown cases =", len(breakdown_cases))
    print("    signature_break cases =", len(signature_break_cases))

    print("\nExpected behavior:")
    print("    weak_valid pass = %d/%d" % (weak_valid_pass, weak_valid_total))
    print("    breakdown detected = %d/%d" % (breakdown_detect_pass, breakdown_detect_total))
    print("    overall expected behavior = %d/%d" % (overall_pass, overall_total))

    print("\nExtremes:")
    print("    max weakness =", max_weakness)
    print("    max finite approximation error =", max_error_finite)
    print("    min sampled -g00 =", min_minus_g00)

    if first_failure is not None:
        print("\nFirst detected breakdown/signature case:")
        print("    config =", first_failure["config"])
        print("    c_light =", first_failure["c_light"])
        print("    source_boost =", first_failure["source_boost"])
        print("    weakness =", first_failure["weakness"])
        print("    regime =", first_failure["regime"])
        print("    max approximation error =", first_failure["max_approximation_error"])

    weak_pass = (weak_valid_total > 0 and weak_valid_pass == weak_valid_total)
    fail_pass = (breakdown_detect_total > 0 and breakdown_detect_pass == breakdown_detect_total)
    signature_boundary_seen = len(signature_break_cases) > 0

    print("\nWEAK-REGIME BEHAVIOR:", "PASS" if weak_pass else "CHECK")
    print("BREAKDOWN DETECTION:", "PASS" if fail_pass else "CHECK")
    print("SIGNATURE BOUNDARY DETECTED:", "PASS" if signature_boundary_seen else "CHECK")

    if weak_pass and fail_pass and signature_boundary_seen:
        print("\nRESULT: PASS")
        print("Toy Model 7e successfully identifies the failure boundary of the weak-field bridge:")
        print("in-domain cases work, strong-field cases break approximations, and extreme cases break metric signature.")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics and thresholds.")

    print("==============================================================")

    make_plots(case_rows)


if __name__ == "__main__":
    main()
