import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 5b: BINARY / TWO-SOURCE ROBUSTNESS SCAN
# ==============================================================
#
# Formal:
#     mu_i = gamma*|Delta K_excess_i|
#     Phi_total = Phi_1 + Phi_2
#     F_total = F_1 + F_2
#     theta_total approximately theta_1 + theta_2
#     far away: |F| ~ (mu_1 + mu_2)/r^2
#
# Microsoft/plain:
#     mu_i = gamma*abs(delta_K_excess_i)
#     phi_total = phi_1 + phi_2
#     force_total = force_1 + force_2
#     theta_total ~ theta_1 + theta_2
#     far away: force ~ (mu_1 + mu_2)/r^2
#
# Normal:
#     Toy Model 5a showed one two-source case works.
#     Toy Model 5b checks whether that result survives changes in
#     mass ratio, source separation, impact parameter, and velocity.
#
# Important:
#     This version uses analytic softened Poisson sources instead of a
#     full 3D grid solve for every binary. This makes the robustness scan fast.
#     Each source has potential Phi = -mu/sqrt(r^2 + s^2), which is a smooth
#     Poisson-field source. Outside the core it behaves like -mu/r.

EPS = 1e-12


# --------------------------------------------------------------
# USER CONTROLS
# --------------------------------------------------------------

GAMMA = 1.0

# Binary separations to test.
SEPARATIONS = [14.0, 25.0, 45.0]

# Speeds to test.
V0_VALUES = [2.4, 3.0]

# Impact parameters are generated from these multiples of separation.
B_FACTORS = [0.9, 1.2, 1.7, 2.4, 3.3, 4.5]
MIN_B = 16.0

# Trajectory integration controls.
X_START = -180.0
X_END = 180.0
DT = 0.08
MAX_STEPS = 15000

# Softening controls.
SOFTENING_FACTOR = 0.45
MIN_SOFTENING = 1.8

# Far-field fit controls.
FAR_R_MAX = 400.0
FAR_POINTS = 80

SUMMARY_CSV = "toy_model_5b_binary_summary.csv"
SCATTER_CSV = "toy_model_5b_binary_scattering_data.csv"


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
    """
    Formal:
        Delta K_excess =
        c_beta*(sqrt(beta)-1)
        + c_inv_beta*(1/sqrt(beta)-1)
        + c_ab*(sqrt(alpha/beta)-1)
        + c_ba*(sqrt(beta/alpha)-1)

    Microsoft/plain:
        delta_K_excess =
            c_beta*(sqrt(beta)-1)
          + c_inv_beta*(1/sqrt(beta)-1)
          + c_ab*(sqrt(alpha/beta)-1)
          + c_ba*(sqrt(beta/alpha)-1)

    Normal:
        This is the exact Toy Model 1 curvature response.
    """
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


def mu_eff_from_network(alpha, beta, radius, gamma=1.0):
    """
    Formal:
        mu_eff = gamma*|Delta K_excess|

    Microsoft/plain:
        mu_eff = gamma*abs(delta_K_excess)

    Normal:
        Network curvature response becomes the external source strength.
    """
    return gamma * abs(delta_K_excess(alpha, beta, radius))


# --------------------------------------------------------------
# SOURCE DEFINITIONS
# --------------------------------------------------------------

SOURCE_LIBRARY = {
    "strong_r6": {"alpha": 8.0, "beta": 0.25, "radius": 6.0},
    "medium_r6": {"alpha": 4.0, "beta": 0.35, "radius": 6.0},
    "weak_r6": {"alpha": 2.0, "beta": 0.60, "radius": 6.0},
    "strong_r8": {"alpha": 8.0, "beta": 0.25, "radius": 8.0},
}

PAIR_CASES = [
    ("equal_strong", "strong_r6", "strong_r6"),
    ("strong_medium", "strong_r6", "medium_r6"),
    ("strong_weak", "strong_r6", "weak_r6"),
    ("mixed_radius", "strong_r6", "strong_r8"),
]


def make_source_from_label(label, x_pos):
    base = SOURCE_LIBRARY[label]
    alpha = base["alpha"]
    beta = base["beta"]
    radius = base["radius"]
    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff_from_network(alpha, beta, radius, GAMMA)
    softening = max(MIN_SOFTENING, SOFTENING_FACTOR * radius)

    return {
        "label": label,
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "delta_K_excess": dK,
        "mu": mu,
        "softening": softening,
        "x": float(x_pos),
        "y": 0.0,
    }


# --------------------------------------------------------------
# ANALYTIC POISSON FIELD FOR SOFTENED SOURCES
# --------------------------------------------------------------


def potential_from_sources(x, y, sources):
    """
    Formal:
        Phi(x,y) = sum_i -mu_i/sqrt(|r-r_i|^2 + s_i^2)

    Microsoft/plain:
        phi_total = sum(-mu_i/sqrt(distance_i^2 + softening_i^2))

    Normal:
        This is the smooth Poisson potential of the binary.
    """
    phi = 0.0
    for src in sources:
        dx = x - src["x"]
        dy = y - src["y"]
        s = src["softening"]
        phi += -src["mu"] / math.sqrt(dx*dx + dy*dy + s*s)
    return phi


def acceleration_from_sources(x, y, sources):
    """
    Formal:
        a = F = -grad(Phi)

    Microsoft/plain:
        acceleration = sum source pulls

    Normal:
        The force is additive because the potential is additive.
    """
    ax = 0.0
    ay = 0.0

    for src in sources:
        dx = x - src["x"]
        dy = y - src["y"]
        s = src["softening"]
        r2 = dx*dx + dy*dy + s*s
        r = math.sqrt(r2)
        fac = -src["mu"] / (r2 * r)
        ax += fac * dx
        ay += fac * dy

    return ax, ay


def force_magnitude_at_point(x, y, sources):
    ax, ay = acceleration_from_sources(x, y, sources)
    return math.sqrt(ax*ax + ay*ay)


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
    denom = np.sum(x*x)
    if denom <= EPS:
        return np.nan, np.full_like(y, np.nan), np.nan
    q = float(np.sum(x*y)/denom)
    pred = q*x
    return q, pred, r2_score(y, pred)


def fit_power_law(x, y):
    """
    Formal:
        y = A/x^p

    Microsoft/plain:
        y ~ 1/x^p
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, np.nan

    lx = np.log(x[mask])
    ly = np.log(y[mask])
    X = np.vstack([np.ones_like(lx), -lx]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)
    logA, p = coeffs
    pred_log = logA - p*lx
    R2_log = r2_score(ly, pred_log)
    return float(math.exp(logA)), float(p), pred_log, float(R2_log)


def far_field_test(sources, separation):
    """
    Formal:
        For r >> separation:
        |F(r)| ~ (mu_1+mu_2)/r^2

    Microsoft/plain:
        far away, binary force should look like one combined source

    Normal:
        We measure along the positive y-axis from the binary centre.
    """
    r_min = max(3.0*separation, 35.0)
    r_values = np.linspace(r_min, FAR_R_MAX, FAR_POINTS)
    F_values = np.array([force_magnitude_at_point(0.0, r, sources) for r in r_values], dtype=float)
    Phi_values = np.array([potential_from_sources(0.0, r, sources) for r in r_values], dtype=float)

    A_force, p_force, pred_log_F, R2_force = fit_power_law(r_values, F_values)

    X = np.vstack([1.0/r_values, np.ones_like(r_values)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, Phi_values, rcond=None)
    A_phi, B_phi = coeffs
    phi_pred = A_phi/r_values + B_phi
    R2_phi = r2_score(Phi_values, phi_pred)

    return {
        "r_values": r_values,
        "F_values": F_values,
        "Phi_values": Phi_values,
        "A_force": A_force,
        "p_force": p_force,
        "R2_force": R2_force,
        "A_phi": float(A_phi),
        "B_phi": float(B_phi),
        "R2_phi": R2_phi,
        "Phi_pred": phi_pred,
        "F_pred": A_force/(r_values**p_force),
    }


# --------------------------------------------------------------
# PARTICLE DYNAMICS
# --------------------------------------------------------------


def verlet_step(state, dt, accel_func):
    # Velocity-Verlet style step: faster than RK4 and accurate enough here.
    x, y, vx, vy = state
    ax, ay = accel_func(x, y)
    x_new = x + vx*dt + 0.5*ax*dt*dt
    y_new = y + vy*dt + 0.5*ay*dt*dt
    ax_new, ay_new = accel_func(x_new, y_new)
    vx_new = vx + 0.5*(ax + ax_new)*dt
    vy_new = vy + 0.5*(ay + ay_new)*dt
    return np.array([x_new, y_new, vx_new, vy_new], dtype=float)


def simulate_scattering(sources, b, v0):
    """
    Formal:
        d2x/dt2 = F_x
        d2y/dt2 = F_y

    Microsoft/plain:
        acceleration = binary_source_force

    Normal:
        A test particle passes the binary and its final angle is measured.
    """
    def accel_func(x, y):
        return acceleration_from_sources(x, y, sources)

    state = np.array([X_START, float(b), float(v0), 0.0], dtype=float)

    reached = False
    captured = False
    escaped = False
    min_distance_to_source = 1e99

    for _ in range(MAX_STEPS):
        x, y, vx, vy = state

        for src in sources:
            d = math.sqrt((x-src["x"])**2 + (y-src["y"])**2)
            min_distance_to_source = min(min_distance_to_source, d)
            if d < src["radius"]:
                captured = True

        if x >= X_END and vx > 0:
            reached = True
            break

        if abs(x) > 1.5*abs(X_START) or abs(y) > 1.5*abs(X_START):
            escaped = True
            break

        state = verlet_step(state, DT, accel_func)

    x, y, vx, vy = state
    theta_signed = math.atan2(vy, vx)
    theta_abs = abs(theta_signed)

    usable = reached and (not captured) and theta_abs > 1e-12

    return {
        "theta_signed_rad": float(theta_signed),
        "theta_abs_rad": float(theta_abs),
        "theta_signed_deg": float(math.degrees(theta_signed)),
        "theta_abs_deg": float(math.degrees(theta_abs)),
        "min_distance_to_source": float(min_distance_to_source),
        "reached": int(reached),
        "captured": int(captured),
        "escaped": int(escaped),
        "usable": int(usable),
    }


def unique_b_values(separation):
    vals = []
    for f in B_FACTORS:
        b = max(MIN_B, f*separation)
        vals.append(round(float(b), 6))
    return sorted(list(set(vals)))


# --------------------------------------------------------------
# CASE SUMMARIES
# --------------------------------------------------------------


def summarize_case(scatter_rows):
    usable = [r for r in scatter_rows if r["usable_total"] == 1 and r["usable_1"] == 1 and r["usable_2"] == 1]

    if len(usable) < 3:
        return {
            "usable_count": len(usable),
            "R2_total_vs_sum": np.nan,
            "slope_total_vs_sum": np.nan,
            "mean_frac_add_error": np.nan,
            "max_frac_add_error": np.nan,
            "R2_total_1_over_b": np.nan,
            "p_scatter_total": np.nan,
            "C_mean": np.nan,
            "C_cv": np.nan,
            "case_pass": 0,
        }

    b = np.array([r["b"] for r in usable], dtype=float)
    theta_total = np.array([r["theta_total_abs_rad"] for r in usable], dtype=float)
    theta_sum = np.array([abs(r["theta_1_signed_rad"] + r["theta_2_signed_rad"]) for r in usable], dtype=float)
    frac_errors = np.array([r["theta_addition_frac_error"] for r in usable], dtype=float)
    v0 = np.array([r["v0"] for r in usable], dtype=float)
    mu_total = np.array([r["mu_total"] for r in usable], dtype=float)

    slope_add, pred_add, R2_add = fit_origin(theta_sum, theta_total)
    Q_inv, pred_inv, R2_inv = fit_origin(1.0/b, theta_total)
    A_pow, p_scatter, pred_log, R2_pow = fit_power_law(b, theta_total)

    C = theta_total*b*v0*v0/(mu_total + EPS)
    C_mean = float(np.mean(C))
    C_std = float(np.std(C))
    C_cv = C_std/C_mean if abs(C_mean) > EPS else np.nan

    case_pass = (
        R2_add > 0.995
        and float(np.mean(frac_errors)) < 0.03
        and R2_inv > 0.95
        and 0.75 <= p_scatter <= 1.35
    )

    return {
        "usable_count": len(usable),
        "R2_total_vs_sum": float(R2_add),
        "slope_total_vs_sum": float(slope_add),
        "mean_frac_add_error": float(np.mean(frac_errors)),
        "max_frac_add_error": float(np.max(frac_errors)),
        "R2_total_1_over_b": float(R2_inv),
        "p_scatter_total": float(p_scatter),
        "C_mean": C_mean,
        "C_cv": float(C_cv),
        "case_pass": int(case_pass),
    }


# --------------------------------------------------------------
# CSV AND PLOTS
# --------------------------------------------------------------


def save_csv(filename, rows):
    if not rows:
        print("No rows to save:", filename)
        return
    keys = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved CSV:", filename)


def make_plots(summary_rows, scatter_rows):
    case_labels = [r["case_id"] for r in summary_rows]
    idx = np.arange(len(case_labels))

    p_force = np.array([r["far_p_force"] for r in summary_rows], dtype=float)
    p_scatter = np.array([r["p_scatter_total"] for r in summary_rows], dtype=float)
    add_err = np.array([r["mean_frac_add_error"] for r in summary_rows], dtype=float)
    R2_add = np.array([r["R2_total_vs_sum"] for r in summary_rows], dtype=float)

    plt.figure(figsize=(14, 6))
    plt.bar(idx, p_force)
    plt.axhline(2.0, linestyle="--", label="target p=2")
    plt.xticks(idx, case_labels, rotation=70, ha="right")
    plt.ylabel("far-field force power p")
    plt.title("Toy Model 5b: far-field force power by binary case")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5b_far_force_power_by_case.png", dpi=200)
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(idx, p_scatter)
    plt.axhline(1.0, linestyle="--", label="target p=1")
    plt.xticks(idx, case_labels, rotation=70, ha="right")
    plt.ylabel("scattering power p")
    plt.title("Toy Model 5b: scattering power by binary case")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5b_scattering_power_by_case.png", dpi=200)
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(idx, add_err)
    plt.axhline(0.03, linestyle="--", label="3 percent")
    plt.xticks(idx, case_labels, rotation=70, ha="right")
    plt.ylabel("mean fractional addition error")
    plt.title("Toy Model 5b: trajectory-addition error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5b_addition_error_by_case.png", dpi=200)
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(idx, R2_add)
    plt.axhline(0.995, linestyle="--", label="pass threshold")
    plt.xticks(idx, case_labels, rotation=70, ha="right")
    plt.ylabel("R2: theta_total vs theta_1 + theta_2")
    plt.title("Toy Model 5b: deflection-addition fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5b_R2_addition_by_case.png", dpi=200)
    plt.show()

    # Global collapse plot.
    usable = [r for r in scatter_rows if r["usable_total"] == 1 and r["usable_1"] == 1 and r["usable_2"] == 1]
    x_collapse = np.array([r["mu_total"]/(r["b"]*r["v0"]**2) for r in usable], dtype=float)
    theta = np.array([r["theta_total_abs_rad"] for r in usable], dtype=float)
    Q, pred, R2 = fit_origin(x_collapse, theta)

    plt.figure(figsize=(10, 7))
    plt.scatter(x_collapse, theta, s=24)
    if len(x_collapse) > 0:
        xx = np.linspace(0.0, np.max(x_collapse)*1.05, 200)
        plt.plot(xx, Q*xx, label="fit: theta = %.4f*mu_total/(b*v0^2), R2=%.4f" % (Q, R2))
    plt.xlabel("mu_total/(b*v0^2)")
    plt.ylabel("|theta_total| in radians")
    plt.title("Toy Model 5b: global binary scaling collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5b_global_collapse.png", dpi=200)
    plt.show()

    # Selected deflection curves.
    selected_case_ids = []
    for r in summary_rows:
        if r["pair_name"] in ["equal_strong", "strong_medium"] and r["v0"] == V0_VALUES[0]:
            if r["separation"] in [25.0, 45.0]:
                selected_case_ids.append(r["case_id"])

    plt.figure(figsize=(11, 7))
    for cid in selected_case_ids:
        subset = [r for r in scatter_rows if r["case_id"] == cid and r["usable_total"] == 1]
        subset = sorted(subset, key=lambda row: row["b"])
        if not subset:
            continue
        b = np.array([r["b"] for r in subset], dtype=float)
        theta_deg = np.array([r["theta_total_abs_deg"] for r in subset], dtype=float)
        plt.plot(b, theta_deg, marker="o", label=cid)

    plt.xlabel("impact parameter b")
    plt.ylabel("|total deflection| in degrees")
    plt.title("Toy Model 5b: selected binary deflection curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5b_selected_deflection_curves.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_5b_far_force_power_by_case.png")
    print("toy_model_5b_scattering_power_by_case.png")
    print("toy_model_5b_addition_error_by_case.png")
    print("toy_model_5b_R2_addition_by_case.png")
    print("toy_model_5b_global_collapse.png")
    print("toy_model_5b_selected_deflection_curves.png")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------


def main():
    print("\n==============================================================")
    print("TOY MODEL 5b: BINARY / TWO-SOURCE ROBUSTNESS SCAN")
    print("==============================================================")
    print("\nFormal:")
    print("    mu_i = gamma*|Delta K_excess_i|")
    print("    Phi_total = Phi_1 + Phi_2")
    print("    F_total = F_1 + F_2")
    print("    theta_total approximately theta_1 + theta_2")
    print("    far away: |F| ~ (mu_1 + mu_2)/r^2")
    print("\nMicrosoft/plain:")
    print("    mu_i = gamma*abs(delta_K_excess_i)")
    print("    phi_total = phi_1 + phi_2")
    print("    force_total = force_1 + force_2")
    print("    theta_total ~ theta_1 + theta_2")
    print("    far away: force ~ (mu_1 + mu_2)/r^2")
    print("\nNormal:")
    print("    We now test whether the two-source result survives")
    print("    different source pairs, separations, impact parameters, and speeds.")
    print("==============================================================")

    summary_rows = []
    scatter_rows = []

    case_counter = 0
    total_cases = len(PAIR_CASES)*len(SEPARATIONS)*len(V0_VALUES)

    for pair_name, label1, label2 in PAIR_CASES:
        for sep in SEPARATIONS:
            src1 = make_source_from_label(label1, -0.5*sep)
            src2 = make_source_from_label(label2, 0.5*sep)
            sources_total = [src1, src2]
            mu_total = src1["mu"] + src2["mu"]
            mass_ratio = max(src1["mu"], src2["mu"])/(min(src1["mu"], src2["mu"]) + EPS)
            far = far_field_test(sources_total, sep)

            for v0 in V0_VALUES:
                case_counter += 1
                case_id = "%s_d%.0f_v%.1f" % (pair_name, sep, v0)

                print("\n--------------------------------------------------------------")
                print("Case %d/%d: %s" % (case_counter, total_cases, case_id))
                print("    source 1:", label1, "mu=", src1["mu"], "dK=", src1["delta_K_excess"])
                print("    source 2:", label2, "mu=", src2["mu"], "dK=", src2["delta_K_excess"])
                print("    separation =", sep)
                print("    mass_ratio =", mass_ratio)
                print("    v0 =", v0)
                print("    far p_force = %.6f, R2_force = %.6f" % (far["p_force"], far["R2_force"]))

                b_vals = unique_b_values(sep)
                case_scatter = []

                for b in b_vals:
                    t1 = simulate_scattering([src1], b, v0)
                    t2 = simulate_scattering([src2], b, v0)
                    tt = simulate_scattering(sources_total, b, v0)

                    theta_sum_signed = t1["theta_signed_rad"] + t2["theta_signed_rad"]
                    theta_total_signed = tt["theta_signed_rad"]
                    add_error = abs(theta_total_signed - theta_sum_signed)
                    add_frac_error = add_error/(abs(theta_total_signed) + EPS)

                    row = {
                        "case_id": case_id,
                        "pair_name": pair_name,
                        "source_1_label": label1,
                        "source_2_label": label2,
                        "separation": float(sep),
                        "mass_ratio": float(mass_ratio),
                        "v0": float(v0),
                        "b": float(b),
                        "alpha_1": src1["alpha"],
                        "beta_1": src1["beta"],
                        "radius_1": src1["radius"],
                        "delta_K_1": src1["delta_K_excess"],
                        "mu_1": src1["mu"],
                        "alpha_2": src2["alpha"],
                        "beta_2": src2["beta"],
                        "radius_2": src2["radius"],
                        "delta_K_2": src2["delta_K_excess"],
                        "mu_2": src2["mu"],
                        "mu_total": mu_total,
                        "theta_1_signed_rad": t1["theta_signed_rad"],
                        "theta_2_signed_rad": t2["theta_signed_rad"],
                        "theta_sum_signed_rad": theta_sum_signed,
                        "theta_total_signed_rad": theta_total_signed,
                        "theta_total_abs_rad": tt["theta_abs_rad"],
                        "theta_total_abs_deg": tt["theta_abs_deg"],
                        "theta_sum_abs_deg": abs(theta_sum_signed)*180.0/math.pi,
                        "theta_addition_error_rad": add_error,
                        "theta_addition_frac_error": add_frac_error,
                        "usable_1": t1["usable"],
                        "usable_2": t2["usable"],
                        "usable_total": tt["usable"],
                        "min_distance_total": tt["min_distance_to_source"],
                    }

                    scatter_rows.append(row)
                    case_scatter.append(row)

                    print(
                        "    b=%6.2f | theta_total=%8.5f deg | theta_sum=%8.5f deg | frac_err=%.5f"
                        % (b, row["theta_total_abs_deg"], row["theta_sum_abs_deg"], add_frac_error)
                    )

                sc = summarize_case(case_scatter)

                far_pass = far["R2_force"] > 0.995 and abs(far["p_force"] - 2.0) < 0.15
                total_pass = bool(far_pass and sc["case_pass"])

                summary = {
                    "case_id": case_id,
                    "pair_name": pair_name,
                    "source_1_label": label1,
                    "source_2_label": label2,
                    "separation": float(sep),
                    "mass_ratio": float(mass_ratio),
                    "v0": float(v0),
                    "mu_1": src1["mu"],
                    "mu_2": src2["mu"],
                    "mu_total": mu_total,
                    "delta_K_1": src1["delta_K_excess"],
                    "delta_K_2": src2["delta_K_excess"],
                    "far_R2_phi": far["R2_phi"],
                    "far_p_force": far["p_force"],
                    "far_R2_force": far["R2_force"],
                    "far_pass": int(far_pass),
                    "usable_count": sc["usable_count"],
                    "R2_total_vs_sum": sc["R2_total_vs_sum"],
                    "slope_total_vs_sum": sc["slope_total_vs_sum"],
                    "mean_frac_add_error": sc["mean_frac_add_error"],
                    "max_frac_add_error": sc["max_frac_add_error"],
                    "R2_total_1_over_b": sc["R2_total_1_over_b"],
                    "p_scatter_total": sc["p_scatter_total"],
                    "C_mean": sc["C_mean"],
                    "C_cv": sc["C_cv"],
                    "scatter_pass": sc["case_pass"],
                    "case_pass": int(total_pass),
                }
                summary_rows.append(summary)

                print("    Summary:")
                print("        R2_total_vs_sum =", sc["R2_total_vs_sum"])
                print("        mean_frac_add_error =", sc["mean_frac_add_error"])
                print("        R2_total_1_over_b =", sc["R2_total_1_over_b"])
                print("        p_scatter_total =", sc["p_scatter_total"])
                print("        C_mean =", sc["C_mean"])
                print("        Result:", "PASS" if total_pass else "CHECK")

    save_csv(SUMMARY_CSV, summary_rows)
    save_csv(SCATTER_CSV, scatter_rows)

    pass_count = sum(r["case_pass"] for r in summary_rows)
    far_pass_count = sum(r["far_pass"] for r in summary_rows)
    scatter_pass_count = sum(r["scatter_pass"] for r in summary_rows)
    total_count = len(summary_rows)

    p_force_vals = np.array([r["far_p_force"] for r in summary_rows], dtype=float)
    p_scatter_vals = np.array([r["p_scatter_total"] for r in summary_rows], dtype=float)
    add_err_vals = np.array([r["mean_frac_add_error"] for r in summary_rows], dtype=float)
    R2_add_vals = np.array([r["R2_total_vs_sum"] for r in summary_rows], dtype=float)

    usable_global = [r for r in scatter_rows if r["usable_total"] == 1 and r["usable_1"] == 1 and r["usable_2"] == 1]
    x_global = np.array([r["mu_total"]/(r["b"]*r["v0"]**2) for r in usable_global], dtype=float)
    theta_global = np.array([r["theta_total_abs_rad"] for r in usable_global], dtype=float)
    Q_global, pred_global, R2_global = fit_origin(x_global, theta_global)

    C_global = np.array(
        [r["theta_total_abs_rad"]*r["b"]*r["v0"]**2/(r["mu_total"] + EPS) for r in usable_global],
        dtype=float,
    )

    print("\n==============================================================")
    print("FINAL TOY MODEL 5b SUMMARY")
    print("==============================================================")
    print("\nFormal:")
    print("    Test binary robustness under separation, mass ratio, b, and v0")
    print("    Phi_total = Phi_1 + Phi_2")
    print("    F_total = F_1 + F_2")
    print("    theta_total approximately theta_1 + theta_2")
    print("    far away: |F| ~ (mu_1 + mu_2)/r^2")
    print("\nMicrosoft/plain:")
    print("    change separation/mass_ratio/impact_parameter/velocity")
    print("    check force_total = force_1 + force_2")
    print("    check theta_total ~ theta_1 + theta_2")
    print("    check far force ~ 1/r^2")
    print("\nNormal:")
    print("    This checks whether Toy Model 5a was a one-off or robust.")

    print("\nCases:")
    print("    total binary cases =", total_count)
    print("    far-field pass =", str(far_pass_count) + "/" + str(total_count))
    print("    scattering-addition pass =", str(scatter_pass_count) + "/" + str(total_count))
    print("    full case pass =", str(pass_count) + "/" + str(total_count))

    print("\nFar-field robustness:")
    print("    mean p_force =", float(np.mean(p_force_vals)))
    print("    std  p_force =", float(np.std(p_force_vals)))
    print("    expected p_force close to 2")

    print("\nScattering robustness:")
    print("    mean p_scatter =", float(np.mean(p_scatter_vals)))
    print("    std  p_scatter =", float(np.std(p_scatter_vals)))
    print("    expected p_scatter close to 1")

    print("\nTrajectory-addition robustness:")
    print("    mean R2_total_vs_sum =", float(np.mean(R2_add_vals)))
    print("    mean fractional addition error =", float(np.mean(add_err_vals)))
    print("    max  fractional addition error =", float(np.max(add_err_vals)))

    print("\nGlobal binary scaling collapse:")
    print("    theta_total = Q*mu_total/(b*v0^2)")
    print("    Q_global =", Q_global)
    print("    R2_global =", R2_global)
    print("    C_mean =", float(np.mean(C_global)))
    print("    C_std =", float(np.std(C_global)))
    print("    C_cv =", float(np.std(C_global)/(np.mean(C_global) + EPS)))

    if pass_count == total_count:
        print("\nRESULT: PASS")
        print("Toy Model 5b supports robust binary weak-field behaviour:")
        print("network sources -> additive field -> additive bending -> far-field inverse-square gravity")
    elif pass_count >= int(0.8*total_count):
        print("\nRESULT: MOSTLY PASS / CHECK EDGE CASES")
        print("Most binary configurations behave correctly, but inspect the CHECK cases.")
    else:
        print("\nRESULT: MIXED")
        print("Binary behaviour depends strongly on configuration; inspect diagnostics.")

    print("==============================================================")

    make_plots(summary_rows, scatter_rows)


if __name__ == "__main__":
    main()
