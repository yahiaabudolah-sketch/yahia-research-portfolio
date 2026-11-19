import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 6c: REDSHIFT, CLOCK RATE, AND SHAPIRO-LIKE DELAY
# ==============================================================
#
# Formal:
#     mu_eff = gamma_source*|Delta K_excess|
#     Phi(r) = -mu_eff/sqrt(r^2 + s^2)
#
#     g_00 = -(1 + 2*Phi/c^2)
#     g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij
#
#     proper-time rate:
#         d_tau/dt = sqrt(-g_00)
#                  = sqrt(1 + 2*Phi/c^2)
#                  approx 1 + Phi/c^2
#
#     gravitational redshift:
#         1 + z = sqrt(-g00_observer)/sqrt(-g00_emitter)
#         z approx (Phi_observer - Phi_emitter)/c^2
#
#     null travel-time delay:
#         Delta t = (1/c) integral [sqrt(gii/(-g00)) - 1] dl
#         Delta t approx -((1 + gamma_PPN)/c^3) integral Phi dl
#
# Microsoft/plain:
#     mu_eff = gamma_source*abs(delta_K_excess)
#     phi = -mu_eff/sqrt(r^2 + softening^2)
#
#     g00 = -(1 + 2*phi/c^2)
#     gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij
#
#     proper_time_rate = sqrt(-g00)
#     proper_time_rate ≈ 1 + phi/c^2
#
#     redshift ≈ (phi_observer - phi_emitter)/c^2
#
#     shapiro_delay_exact = integral((sqrt(gii/(-g00)) - 1)*dl/c)
#     shapiro_delay_weak ≈ -(1 + gamma_PPN)*integral(phi dl)/c^3
#
# Normal:
#     Toy Model 6a/6b showed that the weak-field metric bends light correctly.
#     Toy Model 6c checks whether the same metric also gives consistent clock
#     slowing, gravitational redshift, and Shapiro-like light-travel delay.

EPS = 1e-12


# --------------------------------------------------------------
# USER CONTROLS
# --------------------------------------------------------------

GAMMA_SOURCE = 1.0

NETWORK_CASES = [
    {"label": "r4_a2_b060", "alpha": 2.0, "beta": 0.60, "radius": 4.0},
    {"label": "r4_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 4.0},
    {"label": "r6_a2_b060", "alpha": 2.0, "beta": 0.60, "radius": 6.0},
    {"label": "r6_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 6.0},
    {"label": "r6_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 6.0},
    {"label": "r8_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 8.0},
    {"label": "r8_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 8.0},
]

C_LIGHT_VALUES = [10.0, 12.0, 16.0]
GAMMA_PPN_VALUES = [0.0, 0.5, 1.0]

EMITTER_RADIUS_FACTORS = [1.6, 2.2, 3.2, 5.0]
OBSERVER_RADII = [80.0, 120.0, 180.0]

B_VALUES = [10, 12, 15, 18, 22, 28, 35, 44, 56]
X_MAX = 650.0
N_INTEGRAL = 6001

CLOCK_CSV = "toy_model_6c_clock_redshift_data.csv"
DELAY_CSV = "toy_model_6c_time_delay_data.csv"
SUMMARY_CSV = "toy_model_6c_summary.csv"
CASE_CSV = "toy_model_6c_case_summary.csv"


# --------------------------------------------------------------
# TOY MODEL 1 CLOSED-FORM NETWORK SOURCE
# --------------------------------------------------------------

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}


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


# --------------------------------------------------------------
# METRIC FUNCTIONS
# --------------------------------------------------------------

def softening_from_radius(radius):
    return max(1.8, radius / 3.0)


def phi_r(r, mu, softening):
    return -mu / math.sqrt(r*r + softening*softening)


def phi_xy(x, y, mu, softening):
    return -mu / math.sqrt(x*x + y*y + softening*softening)


def g00_from_phi(phi, c_light):
    return -(1.0 + 2.0*phi/(c_light*c_light))


def gii_from_phi(phi, c_light, gamma_ppn):
    return 1.0 - 2.0*gamma_ppn*phi/(c_light*c_light)


def proper_time_rate(phi, c_light):
    val = 1.0 + 2.0*phi/(c_light*c_light)
    if val <= 0:
        return np.nan
    return math.sqrt(val)


def proper_time_rate_weak(phi, c_light):
    return 1.0 + phi/(c_light*c_light)


def redshift_exact(phi_emit, phi_obs, c_light):
    g_emit = -g00_from_phi(phi_emit, c_light)
    g_obs = -g00_from_phi(phi_obs, c_light)

    if g_emit <= 0 or g_obs <= 0:
        return np.nan

    return math.sqrt(g_obs/g_emit) - 1.0


def redshift_weak(phi_emit, phi_obs, c_light):
    return (phi_obs - phi_emit)/(c_light*c_light)


def weak_field_parameter(mu, radius, c_light):
    soft = softening_from_radius(radius)
    return 2.0*mu/(soft*c_light*c_light)


# --------------------------------------------------------------
# NUMERICAL TIME DELAY INTEGRALS
# --------------------------------------------------------------

def time_delay_integrals(mu, radius, b, c_light, gamma_ppn):
    """
    Straight-line path y=b from x=-X_MAX to +X_MAX.

    Formal:
        Delta t_exact = (1/c) integral [sqrt(gii/(-g00)) - 1] dl
        Delta t_weak = -((1 + gamma_PPN)/c^3) integral Phi dl

    Microsoft/plain:
        delay_exact = integral((sqrt(gii/(-g00)) - 1)*dl/c)
        delay_weak = -(1 + gamma_PPN)*integral(phi dl)/c^3
    """
    soft = softening_from_radius(radius)
    x = np.linspace(-X_MAX, X_MAX, N_INTEGRAL)

    r2 = x*x + b*b + soft*soft
    phi = -mu / np.sqrt(r2)

    minus_g00 = 1.0 + 2.0*phi/(c_light*c_light)
    gii = 1.0 - 2.0*gamma_ppn*phi/(c_light*c_light)

    if np.any(minus_g00 <= 0):
        return {
            "delay_exact": np.nan,
            "delay_weak": np.nan,
            "flat_time": 2.0*X_MAX/c_light,
            "integral_phi": np.nan,
            "integral_unit": np.nan,
        }

    integrand_exact = np.sqrt(gii/minus_g00) - 1.0

    # Fixed for newer NumPy versions: use np.trapezoid, not np.trapz.
    delay_exact = np.trapezoid(integrand_exact, x) / c_light
    integral_phi = np.trapezoid(phi, x)
    integral_unit = np.trapezoid(1.0/np.sqrt(r2), x)

    delay_weak = -((1.0 + gamma_ppn)/(c_light**3)) * integral_phi

    return {
        "delay_exact": float(delay_exact),
        "delay_weak": float(delay_weak),
        "flat_time": float(2.0*X_MAX/c_light),
        "integral_phi": float(integral_phi),
        "integral_unit": float(integral_unit),
    }


# --------------------------------------------------------------
# FITTING TOOLS
# --------------------------------------------------------------

def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if len(y) < 2:
        return np.nan

    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot <= EPS:
        return np.nan

    return float(1.0 - ss_res/ss_tot)


def fit_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    denom = np.sum(x*x)

    if denom <= EPS:
        return np.nan, np.full_like(y, np.nan), np.nan

    q = float(np.sum(x*y)/denom)
    pred = q*x

    return q, pred, r2_score(y, pred)


def relative_error(exact, approx):
    return abs(exact - approx)/(abs(exact) + EPS)


# --------------------------------------------------------------
# SCANS
# --------------------------------------------------------------

def run_clock_redshift_scan():
    rows = []

    for case in NETWORK_CASES:
        label = case["label"]
        alpha = case["alpha"]
        beta = case["beta"]
        radius = case["radius"]

        dK = delta_K_excess(alpha, beta, radius)
        mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
        soft = softening_from_radius(radius)

        emitter_radii = [max(f*radius, soft*2.2) for f in EMITTER_RADIUS_FACTORS]

        for c_light in C_LIGHT_VALUES:
            weakness = weak_field_parameter(mu, radius, c_light)

            for r_emit in emitter_radii:
                phi_emit = phi_r(r_emit, mu, soft)
                tau_exact = proper_time_rate(phi_emit, c_light)
                tau_weak = proper_time_rate_weak(phi_emit, c_light)
                tau_rel_err = relative_error(tau_exact - 1.0, tau_weak - 1.0)

                for r_obs in OBSERVER_RADII:
                    phi_obs = phi_r(r_obs, mu, soft)
                    z_exact = redshift_exact(phi_emit, phi_obs, c_light)
                    z_weak = redshift_weak(phi_emit, phi_obs, c_light)
                    z_rel_err = relative_error(z_exact, z_weak)

                    rows.append({
                        "case": label,
                        "alpha": alpha,
                        "beta": beta,
                        "radius": radius,
                        "delta_K_excess": dK,
                        "mu_eff": mu,
                        "softening": soft,
                        "c_light": c_light,
                        "weak_field_parameter": weakness,
                        "r_emit": r_emit,
                        "r_obs": r_obs,
                        "phi_emit": phi_emit,
                        "phi_obs": phi_obs,
                        "tau_rate_exact": tau_exact,
                        "tau_rate_weak": tau_weak,
                        "tau_rate_rel_error": tau_rel_err,
                        "redshift_exact": z_exact,
                        "redshift_weak": z_weak,
                        "redshift_rel_error": z_rel_err,
                    })

    return rows


def run_delay_scan():
    rows = []
    total = len(NETWORK_CASES)*len(C_LIGHT_VALUES)*len(GAMMA_PPN_VALUES)
    counter = 0

    for case in NETWORK_CASES:
        label = case["label"]
        alpha = case["alpha"]
        beta = case["beta"]
        radius = case["radius"]

        dK = delta_K_excess(alpha, beta, radius)
        mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
        soft = softening_from_radius(radius)

        for c_light in C_LIGHT_VALUES:
            weakness = weak_field_parameter(mu, radius, c_light)

            for gamma_ppn in GAMMA_PPN_VALUES:
                counter += 1

                print("Running delay set %d/%d: case=%s, c=%.1f, gamma_PPN=%.1f" % (
                    counter, total, label, c_light, gamma_ppn
                ))

                for b in B_VALUES:
                    out = time_delay_integrals(mu, radius, b, c_light, gamma_ppn)

                    delay_exact = out["delay_exact"]
                    delay_weak = out["delay_weak"]
                    rel_err = relative_error(delay_exact, delay_weak)

                    expected_scale = (1.0 + gamma_ppn)*mu*out["integral_unit"]/(c_light**3)

                    rows.append({
                        "case": label,
                        "alpha": alpha,
                        "beta": beta,
                        "radius": radius,
                        "delta_K_excess": dK,
                        "mu_eff": mu,
                        "softening": soft,
                        "c_light": c_light,
                        "gamma_PPN": gamma_ppn,
                        "weak_field_parameter": weakness,
                        "b": float(b),
                        "flat_time": out["flat_time"],
                        "integral_phi": out["integral_phi"],
                        "integral_unit": out["integral_unit"],
                        "delay_exact": delay_exact,
                        "delay_weak": delay_weak,
                        "delay_expected_scale": expected_scale,
                        "delay_ratio_exact_to_weak": delay_exact/(delay_weak + EPS),
                        "delay_rel_error": rel_err,
                    })

    return rows


# --------------------------------------------------------------
# SUMMARIES
# --------------------------------------------------------------

def summarize(clock_rows, delay_rows):
    case_rows = []

    for case in sorted(set(r["case"] for r in clock_rows)):
        for c_light in C_LIGHT_VALUES:
            subset = [
                r for r in clock_rows
                if r["case"] == case and abs(r["c_light"] - c_light) < 1e-12
            ]

            if not subset:
                continue

            tau_errors = np.array([r["tau_rate_rel_error"] for r in subset], dtype=float)
            z_errors = np.array([r["redshift_rel_error"] for r in subset], dtype=float)
            weakness = subset[0]["weak_field_parameter"]

            xz = np.array([r["redshift_weak"] for r in subset], dtype=float)
            yz = np.array([r["redshift_exact"] for r in subset], dtype=float)

            Qz, predz, R2z = fit_origin(xz, yz)

            clock_pass = (
                weakness < 0.05
                and float(np.mean(tau_errors)) < 0.02
                and float(np.mean(z_errors)) < 0.02
                and R2z > 0.999
                and abs(Qz - 1.0) < 0.03
            )

            case_rows.append({
                "type": "clock_redshift",
                "case": case,
                "c_light": c_light,
                "gamma_PPN": "none",
                "weak_field_parameter": weakness,
                "mean_tau_rel_error": float(np.mean(tau_errors)),
                "max_tau_rel_error": float(np.max(tau_errors)),
                "mean_redshift_rel_error": float(np.mean(z_errors)),
                "max_redshift_rel_error": float(np.max(z_errors)),
                "redshift_Q_fit": Qz,
                "redshift_R2": R2z,
                "mean_delay_rel_error": np.nan,
                "max_delay_rel_error": np.nan,
                "delay_Q_fit": np.nan,
                "delay_R2": np.nan,
                "delay_ratio_mean": np.nan,
                "case_pass": int(clock_pass),
            })

    for case in sorted(set(r["case"] for r in delay_rows)):
        for c_light in C_LIGHT_VALUES:
            for gamma_ppn in GAMMA_PPN_VALUES:
                subset = [
                    r for r in delay_rows
                    if r["case"] == case
                    and abs(r["c_light"] - c_light) < 1e-12
                    and abs(r["gamma_PPN"] - gamma_ppn) < 1e-12
                ]

                if not subset:
                    continue

                errors = np.array([r["delay_rel_error"] for r in subset], dtype=float)
                ratios = np.array([r["delay_ratio_exact_to_weak"] for r in subset], dtype=float)
                weakness = subset[0]["weak_field_parameter"]

                x = np.array([r["delay_weak"] for r in subset], dtype=float)
                y = np.array([r["delay_exact"] for r in subset], dtype=float)

                Qd, predd, R2d = fit_origin(x, y)

                delay_pass = (
                    weakness < 0.05
                    and float(np.mean(errors)) < 0.02
                    and R2d > 0.999
                    and abs(Qd - 1.0) < 0.03
                )

                case_rows.append({
                    "type": "time_delay",
                    "case": case,
                    "c_light": c_light,
                    "gamma_PPN": gamma_ppn,
                    "weak_field_parameter": weakness,
                    "mean_tau_rel_error": np.nan,
                    "max_tau_rel_error": np.nan,
                    "mean_redshift_rel_error": np.nan,
                    "max_redshift_rel_error": np.nan,
                    "redshift_Q_fit": np.nan,
                    "redshift_R2": np.nan,
                    "mean_delay_rel_error": float(np.mean(errors)),
                    "max_delay_rel_error": float(np.max(errors)),
                    "delay_Q_fit": Qd,
                    "delay_R2": R2d,
                    "delay_ratio_mean": float(np.mean(ratios)),
                    "case_pass": int(delay_pass),
                })

    tau_x = np.array([r["tau_rate_weak"] - 1.0 for r in clock_rows], dtype=float)
    tau_y = np.array([r["tau_rate_exact"] - 1.0 for r in clock_rows], dtype=float)

    Qtau, predtau, R2tau = fit_origin(tau_x, tau_y)

    z_x = np.array([r["redshift_weak"] for r in clock_rows], dtype=float)
    z_y = np.array([r["redshift_exact"] for r in clock_rows], dtype=float)

    Qz, predz, R2z = fit_origin(z_x, z_y)

    d_x = np.array([r["delay_weak"] for r in delay_rows], dtype=float)
    d_y = np.array([r["delay_exact"] for r in delay_rows], dtype=float)

    Qd, predd, R2d = fit_origin(d_x, d_y)

    tau_errors = np.array([r["tau_rate_rel_error"] for r in clock_rows], dtype=float)
    z_errors = np.array([r["redshift_rel_error"] for r in clock_rows], dtype=float)
    delay_errors = np.array([r["delay_rel_error"] for r in delay_rows], dtype=float)

    ratio05 = []
    ratio1 = []

    for r0 in delay_rows:
        if abs(r0["gamma_PPN"] - 0.0) > 1e-12:
            continue

        matches05 = [
            r for r in delay_rows
            if r["case"] == r0["case"]
            and abs(r["c_light"] - r0["c_light"]) < 1e-12
            and abs(r["b"] - r0["b"]) < 1e-12
            and abs(r["gamma_PPN"] - 0.5) < 1e-12
        ]

        matches1 = [
            r for r in delay_rows
            if r["case"] == r0["case"]
            and abs(r["c_light"] - r0["c_light"]) < 1e-12
            and abs(r["b"] - r0["b"]) < 1e-12
            and abs(r["gamma_PPN"] - 1.0) < 1e-12
        ]

        if matches05:
            ratio05.append(matches05[0]["delay_exact"]/(r0["delay_exact"] + EPS))

        if matches1:
            ratio1.append(matches1[0]["delay_exact"]/(r0["delay_exact"] + EPS))

    ratio05 = np.array(ratio05, dtype=float)
    ratio1 = np.array(ratio1, dtype=float)

    final = {
        "clock_rows": len(clock_rows),
        "delay_rows": len(delay_rows),
        "case_groups_passed": int(sum(r["case_pass"] for r in case_rows)),
        "case_groups_total": len(case_rows),
        "max_weak_field_parameter": float(max([r["weak_field_parameter"] for r in clock_rows + delay_rows])),
        "time_dilation_Q_fit": Qtau,
        "time_dilation_R2": R2tau,
        "mean_tau_rel_error": float(np.mean(tau_errors)),
        "max_tau_rel_error": float(np.max(tau_errors)),
        "redshift_Q_fit": Qz,
        "redshift_R2": R2z,
        "mean_redshift_rel_error": float(np.mean(z_errors)),
        "max_redshift_rel_error": float(np.max(z_errors)),
        "delay_Q_fit": Qd,
        "delay_R2": R2d,
        "mean_delay_rel_error": float(np.mean(delay_errors)),
        "max_delay_rel_error": float(np.max(delay_errors)),
        "gamma05_to_gamma0_delay_mean": float(np.mean(ratio05)),
        "gamma05_to_gamma0_delay_std": float(np.std(ratio05)),
        "gamma1_to_gamma0_delay_mean": float(np.mean(ratio1)),
        "gamma1_to_gamma0_delay_std": float(np.std(ratio1)),
    }

    return case_rows, final


# --------------------------------------------------------------
# CSV AND PLOTS
# --------------------------------------------------------------

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


def make_plots(clock_rows, delay_rows):
    x_tau = np.array([r["tau_rate_weak"] - 1.0 for r in clock_rows], dtype=float)
    y_tau = np.array([r["tau_rate_exact"] - 1.0 for r in clock_rows], dtype=float)

    plt.figure(figsize=(8, 7))
    plt.scatter(x_tau, y_tau, s=20)

    lo = min(np.min(x_tau), np.min(y_tau))
    hi = max(np.max(x_tau), np.max(y_tau))

    plt.plot([lo, hi], [lo, hi], label="perfect weak-field relation")
    plt.xlabel("weak clock-rate shift")
    plt.ylabel("exact clock-rate shift")
    plt.title("Toy Model 6c: time-dilation consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6c_time_dilation_consistency.png", dpi=200)
    plt.show()

    x_z = np.array([r["redshift_weak"] for r in clock_rows], dtype=float)
    y_z = np.array([r["redshift_exact"] for r in clock_rows], dtype=float)

    plt.figure(figsize=(8, 7))
    plt.scatter(x_z, y_z, s=20)

    lo = min(np.min(x_z), np.min(y_z))
    hi = max(np.max(x_z), np.max(y_z))

    plt.plot([lo, hi], [lo, hi], label="perfect weak-field relation")
    plt.xlabel("weak redshift")
    plt.ylabel("exact redshift")
    plt.title("Toy Model 6c: redshift consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6c_redshift_consistency.png", dpi=200)
    plt.show()

    x_d = np.array([r["delay_weak"] for r in delay_rows], dtype=float)
    y_d = np.array([r["delay_exact"] for r in delay_rows], dtype=float)

    plt.figure(figsize=(8, 7))
    plt.scatter(x_d, y_d, s=20)

    lo = min(np.min(x_d), np.min(y_d))
    hi = max(np.max(x_d), np.max(y_d))

    plt.plot([lo, hi], [lo, hi], label="perfect weak-field relation")
    plt.xlabel("weak Shapiro-like delay")
    plt.ylabel("exact metric travel-time delay")
    plt.title("Toy Model 6c: time-delay consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6c_delay_consistency.png", dpi=200)
    plt.show()

    representative = "r6_a8_b025"
    c_light = 12.0

    plt.figure(figsize=(10, 6))

    for gamma_ppn in GAMMA_PPN_VALUES:
        subset = [
            r for r in delay_rows
            if r["case"] == representative
            and abs(r["c_light"] - c_light) < 1e-12
            and abs(r["gamma_PPN"] - gamma_ppn) < 1e-12
        ]

        subset = sorted(subset, key=lambda row: row["b"])

        b = [r["b"] for r in subset]
        delay = [r["delay_exact"] for r in subset]

        plt.plot(b, delay, marker="o", label="gamma=%.1f" % gamma_ppn)

    plt.xlabel("impact parameter b")
    plt.ylabel("exact time delay")
    plt.title("Toy Model 6c: Shapiro-like delay by gamma_PPN")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6c_delay_vs_b_by_gamma.png", dpi=200)
    plt.show()

    subset0 = [
        r for r in delay_rows
        if r["case"] == representative
        and abs(r["c_light"] - c_light) < 1e-12
        and abs(r["gamma_PPN"] - 0.0) < 1e-12
    ]

    subset05 = [
        r for r in delay_rows
        if r["case"] == representative
        and abs(r["c_light"] - c_light) < 1e-12
        and abs(r["gamma_PPN"] - 0.5) < 1e-12
    ]

    subset1 = [
        r for r in delay_rows
        if r["case"] == representative
        and abs(r["c_light"] - c_light) < 1e-12
        and abs(r["gamma_PPN"] - 1.0) < 1e-12
    ]

    subset0 = sorted(subset0, key=lambda row: row["b"])
    subset05 = sorted(subset05, key=lambda row: row["b"])
    subset1 = sorted(subset1, key=lambda row: row["b"])

    b = [r["b"] for r in subset0]

    ratio05 = [
        r05["delay_exact"]/(r0["delay_exact"] + EPS)
        for r0, r05 in zip(subset0, subset05)
    ]

    ratio1 = [
        r1["delay_exact"]/(r0["delay_exact"] + EPS)
        for r0, r1 in zip(subset0, subset1)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(b, ratio05, marker="o", label="delay(gamma=0.5)/delay(gamma=0)")
    plt.plot(b, ratio1, marker="o", label="delay(gamma=1)/delay(gamma=0)")
    plt.axhline(1.5, linestyle="--", label="expected 1.5")
    plt.axhline(2.0, linestyle="--", label="expected 2.0")
    plt.xlabel("impact parameter b")
    plt.ylabel("delay ratio")
    plt.title("Toy Model 6c: gamma_PPN scaling of time delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6c_delay_gamma_ratios.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_6c_time_dilation_consistency.png")
    print("toy_model_6c_redshift_consistency.png")
    print("toy_model_6c_delay_consistency.png")
    print("toy_model_6c_delay_vs_b_by_gamma.png")
    print("toy_model_6c_delay_gamma_ratios.png")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main():
    print("\n==============================================================")
    print("TOY MODEL 6c: REDSHIFT, CLOCK RATE, AND SHAPIRO-LIKE DELAY")
    print("==============================================================")

    print("\nFormal:")
    print("    mu_eff = gamma_source*|Delta K_excess|")
    print("    g_00 = -(1 + 2*Phi/c^2)")
    print("    g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij")
    print("    d_tau/dt = sqrt(-g_00) approx 1 + Phi/c^2")
    print("    z approx (Phi_observer - Phi_emitter)/c^2")
    print("    Delta t approx -((1+gamma_PPN)/c^3) integral Phi dl")

    print("\nMicrosoft/plain:")
    print("    mu_eff = gamma_source*abs(delta_K_excess)")
    print("    g00 = -(1 + 2*phi/c^2)")
    print("    gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij")
    print("    proper_time_rate = sqrt(-g00)")
    print("    redshift ≈ (phi_observer - phi_emitter)/c^2")
    print("    delay ≈ -(1+gamma_PPN)*integral(phi dl)/c^3")

    print("\nNormal:")
    print("    We test whether the same weak-field metric that bends light")
    print("    also gives consistent clock slowing, redshift, and signal delay.")
    print("==============================================================")

    print("\nNetwork cases:")

    for case in NETWORK_CASES:
        dK = delta_K_excess(case["alpha"], case["beta"], case["radius"])
        mu = mu_eff(case["alpha"], case["beta"], case["radius"], GAMMA_SOURCE)

        print("    %s: alpha=%.1f beta=%.2f radius=%.1f dK=%.6f mu=%.6f" % (
            case["label"], case["alpha"], case["beta"], case["radius"], dK, mu
        ))

    print("\nc_light values:", C_LIGHT_VALUES)
    print("gamma_PPN values:", GAMMA_PPN_VALUES)
    print("b values:", B_VALUES)

    print("\nRunning clock/redshift scan...")
    clock_rows = run_clock_redshift_scan()

    print("\nRunning time-delay scan...")
    delay_rows = run_delay_scan()

    case_rows, final = summarize(clock_rows, delay_rows)

    save_csv(CLOCK_CSV, clock_rows)
    save_csv(DELAY_CSV, delay_rows)
    save_csv(CASE_CSV, case_rows)
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 6c SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Phi -> g_mu_nu")
    print("    d_tau/dt = sqrt(-g_00)")
    print("    z = sqrt(-g00_obs/g00_emit) - 1")
    print("    Delta t = (1/c) integral [sqrt(gii/(-g00)) - 1] dl")

    print("\nMicrosoft/plain:")
    print("    phi -> metric")
    print("    clock_rate = sqrt(-g00)")
    print("    redshift = sqrt(-g00_obs/-g00_emit) - 1")
    print("    delay = integral((sqrt(gii/(-g00)) - 1)*dl/c)")

    print("\nNormal:")
    print("    This checks whether the metric wrapper is consistent beyond bending.")

    print("\nCases:")
    print("    clock/redshift rows =", final["clock_rows"])
    print("    delay rows =", final["delay_rows"])
    print("    case groups passed =", str(final["case_groups_passed"]) + "/" + str(final["case_groups_total"]))

    print("\nWeak-field condition:")
    print("    max weak_field_parameter =", final["max_weak_field_parameter"])
    print("    target < 0.05")

    print("\nClock-rate consistency:")
    print("    time_dilation_Q_fit =", final["time_dilation_Q_fit"])
    print("    time_dilation_R2 =", final["time_dilation_R2"])
    print("    mean_tau_rel_error =", final["mean_tau_rel_error"])

    print("\nRedshift consistency:")
    print("    redshift_Q_fit =", final["redshift_Q_fit"])
    print("    redshift_R2 =", final["redshift_R2"])
    print("    mean_redshift_rel_error =", final["mean_redshift_rel_error"])

    print("\nTime-delay consistency:")
    print("    delay_Q_fit =", final["delay_Q_fit"])
    print("    delay_R2 =", final["delay_R2"])
    print("    mean_delay_rel_error =", final["mean_delay_rel_error"])

    print("\nGamma_PPN delay scaling:")
    print("    mean delay(gamma=0.5)/delay(gamma=0) =", final["gamma05_to_gamma0_delay_mean"])
    print("    std  delay(gamma=0.5)/delay(gamma=0) =", final["gamma05_to_gamma0_delay_std"])
    print("    expected = 1.5")
    print("    mean delay(gamma=1)/delay(gamma=0) =", final["gamma1_to_gamma0_delay_mean"])
    print("    std  delay(gamma=1)/delay(gamma=0) =", final["gamma1_to_gamma0_delay_std"])
    print("    expected = 2")

    weak_pass = final["max_weak_field_parameter"] < 0.05
    cases_pass = final["case_groups_passed"] == final["case_groups_total"]
    tau_pass = abs(final["time_dilation_Q_fit"] - 1.0) < 0.03 and final["time_dilation_R2"] > 0.999
    redshift_pass = abs(final["redshift_Q_fit"] - 1.0) < 0.03 and final["redshift_R2"] > 0.999
    delay_pass = abs(final["delay_Q_fit"] - 1.0) < 0.03 and final["delay_R2"] > 0.999
    gamma_pass = (
        abs(final["gamma05_to_gamma0_delay_mean"] - 1.5) < 0.04
        and abs(final["gamma1_to_gamma0_delay_mean"] - 2.0) < 0.05
    )

    print("\nWEAK-FIELD CONDITION:", "PASS" if weak_pass else "CHECK")
    print("CLOCK/REDSHIFT/TIME-DELAY CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")
    print("CLOCK-RATE CONSISTENCY:", "PASS" if tau_pass else "CHECK")
    print("REDSHIFT CONSISTENCY:", "PASS" if redshift_pass else "CHECK")
    print("TIME-DELAY CONSISTENCY:", "PASS" if delay_pass else "CHECK")
    print("GAMMA_PPN DELAY SCALING:", "PASS" if gamma_pass else "CHECK")

    if weak_pass and cases_pass and tau_pass and redshift_pass and delay_pass and gamma_pass:
        print("\nRESULT: PASS")
        print("Toy Model 6c supports metric consistency beyond light bending:")
        print("network source -> metric -> clock slowing + redshift + Shapiro-like delay")
    else:
        print("\nRESULT: MIXED")
        print("The metric wrapper is useful, but inspect the CHECK diagnostics.")

    print("==============================================================")

    make_plots(clock_rows, delay_rows)


if __name__ == "__main__":
    main()
