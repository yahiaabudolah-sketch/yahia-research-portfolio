import numpy as np
import matplotlib.pyplot as plt
import csv
import math


# ==============================================================
# TOY MODEL 4b: ROBUSTNESS SCAN
# Network source -> Poisson field -> dynamic bending
# ==============================================================


# -----------------------------
# USER SETTINGS
# -----------------------------

RADII = [4, 6, 8]
ALPHAS = [2.0, 4.0, 8.0]
BETAS = [1.0, 0.6, 0.35, 0.25]
V0_VALUES = [2.0, 2.8]
B_VALUES = [14, 18, 22, 26, 30, 36, 42, 50, 58]

GAMMA = 1.0

# Radial Poisson solver settings
RMAX = 180.0
DR = 0.15

# Trajectory settings
X_START = -100.0
X_END = 100.0
DT = 0.04

# Output filenames
PATH_CSV = "toy_model_4b_poisson_path_data.csv"
SUMMARY_CSV = "toy_model_4b_poisson_summary.csv"


# -----------------------------
# EXACT BASELINE-CORRECTED CURVATURE COEFFICIENTS
# From Toy Model 1 closed-form law
# -----------------------------

DELTAK_COEFFS = {
    4: {
        "c_beta": -0.371681415929,
        "c_ratio": -0.300884955752,
    },
    6: {
        "c_beta": -0.314720812183,
        "c_ratio": -0.274111675127,
    },
    8: {
        "c_beta": -0.258675078864,
        "c_ratio": -0.233438485804,
    },
}


def deltaK_excess(alpha, beta, radius):
    """
    Formal:
        delta_K_excess(alpha,beta,r)
        = c_beta(r)*(sqrt(beta)-1)
        + c_beta(r)*(1/sqrt(beta)-1)
        + c_ratio(r)*(sqrt(alpha/beta)-1)
        + c_ratio(r)*(sqrt(beta/alpha)-1)

    Microsoft/plain:
        delta_K_excess =
            c_beta*(sqrt(beta)-1)
          + c_beta*(1/sqrt(beta)-1)
          + c_ratio*(sqrt(alpha/beta)-1)
          + c_ratio*(sqrt(beta/alpha)-1)

    Normal:
        This is the exact network curvature response from Toy Model 1,
        with the empty-space background removed.
    """

    radius_int = int(radius)

    if radius_int not in DELTAK_COEFFS:
        raise ValueError("No closed-form coefficients for radius = " + str(radius))

    c_beta = DELTAK_COEFFS[radius_int]["c_beta"]
    c_ratio = DELTAK_COEFFS[radius_int]["c_ratio"]

    term_beta = c_beta * (math.sqrt(beta) - 1.0)
    term_inv_beta = c_beta * (1.0 / math.sqrt(beta) - 1.0)

    term_alpha_beta = c_ratio * (math.sqrt(alpha / beta) - 1.0)
    term_beta_alpha = c_ratio * (math.sqrt(beta / alpha) - 1.0)

    return term_beta + term_inv_beta + term_alpha_beta + term_beta_alpha


# -----------------------------
# BASIC FITTING TOOLS
# -----------------------------

def r2_score(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y) < 2:
        return float("nan")

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return float("nan")

    return 1.0 - ss_res / ss_tot


def fit_through_origin(x, y):
    """
    Fit:
        y = q*x
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = np.sum(x * x)

    if denom == 0:
        return float("nan"), np.full_like(y, np.nan), float("nan")

    q = np.sum(x * y) / denom
    y_pred = q * x
    r2 = r2_score(y, y_pred)

    return q, y_pred, r2


def fit_power_law(x, y):
    """
    Fit:
        y = A*x^slope

    For inverse laws:
        y ~ 1/x^p
        p = -slope
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0)

    if np.sum(mask) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    lx = np.log(x[mask])
    ly = np.log(y[mask])

    X = np.column_stack([np.ones_like(lx), lx])
    coeffs, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)

    logA = coeffs[0]
    slope = coeffs[1]

    ly_pred = X @ coeffs
    r2_log = r2_score(ly, ly_pred)

    A = math.exp(logA)
    p = -slope

    return A, slope, p, r2_log


def fit_potential_inverse_r(r, phi):
    """
    Fit:
        phi(r) = A/r + B
    """

    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)

    X = np.column_stack([1.0 / r, np.ones_like(r)])
    coeffs, _, _, _ = np.linalg.lstsq(X, phi, rcond=None)

    A = coeffs[0]
    B = coeffs[1]

    phi_pred = X @ coeffs
    r2 = r2_score(phi, phi_pred)

    return A, B, phi_pred, r2


# -----------------------------
# CSV HELPERS
# -----------------------------

def save_csv(filename, rows):
    if len(rows) == 0:
        print("No rows to save for", filename)
        return

    keys = list(rows[0].keys())

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filename)


# -----------------------------
# RADIAL POISSON SOLVER
# -----------------------------

def make_radial_network_density(mu_eff, source_radius, r_grid):
    """
    Formal:
        integral 4*pi*r^2*rho(r) dr = mu_eff

    Microsoft/plain:
        total_source = integral(4*pi*r^2*rho dr) = mu_eff

    Normal:
        We turn the network source strength into a smooth spherical density.
    """

    sigma = max(1.5, 0.55 * source_radius)

    shape = np.exp(-0.5 * (r_grid / sigma) ** 2)

    norm = 4.0 * math.pi * np.sum(shape * r_grid ** 2) * DR

    rho = mu_eff * shape / norm

    return rho, sigma


def solve_radial_poisson(mu_eff, source_radius):
    """
    Formal:
        (1/r^2) d/dr [r^2 dPhi/dr] = 4*pi*rho(r)

        M(r) = integral_0^r 4*pi*s^2*rho(s) ds

        dPhi/dr = M(r)/r^2

        F(r) = -dPhi/dr

    Microsoft/plain:
        enclosed_mass = integral(4*pi*r^2*rho dr)
        force_strength = enclosed_mass/r^2
        force = -force_strength * radial_direction

    Normal:
        This is the spherical Poisson field generated by the network source.
    """

    r = np.arange(DR, RMAX + 0.5 * DR, DR)

    rho, sigma = make_radial_network_density(mu_eff, source_radius, r)

    enclosed = 4.0 * math.pi * np.cumsum(rho * r ** 2) * DR

    force_mag = enclosed / (r ** 2)

    total_source = enclosed[-1]

    phi = np.zeros_like(r)

    # Boundary condition:
    # phi(Rmax) approx -M/Rmax
    phi[-1] = -total_source / r[-1]

    # Integrate inward:
    # dPhi/dr = force_mag
    for i in range(len(r) - 2, -1, -1):
        phi[i] = phi[i + 1] - 0.5 * (force_mag[i + 1] + force_mag[i]) * DR

    field = {
        "r": r,
        "rho": rho,
        "phi": phi,
        "force_mag": force_mag,
        "enclosed": enclosed,
        "sigma": sigma,
        "total_source": total_source,
        "source_radius": source_radius,
        "mu_eff": mu_eff,
    }

    return field


def interpolate_force(field, rr):
    r = field["r"]
    f = field["force_mag"]
    total_source = field["total_source"]

    if rr <= 0:
        return 0.0

    if rr <= r[0]:
        return f[0] * rr / r[0]

    if rr >= r[-1]:
        return total_source / (rr ** 2)

    return np.interp(rr, r, f)


def acceleration_from_field(field, x, y):
    rr = math.sqrt(x * x + y * y)

    if rr < 1e-12:
        return 0.0, 0.0

    fmag = interpolate_force(field, rr)

    ax = -fmag * x / rr
    ay = -fmag * y / rr

    return ax, ay


# -----------------------------
# FIELD DIAGNOSTICS
# -----------------------------

def evaluate_field_shape(field):
    r = field["r"]
    phi = field["phi"]
    force_mag = field["force_mag"]
    source_radius = field["source_radius"]

    r_min_fit = max(4.0 * source_radius, 18.0)
    r_max_fit = 0.65 * RMAX

    mask = (r >= r_min_fit) & (r <= r_max_fit)

    r_fit = r[mask]
    phi_fit_data = phi[mask]
    f_fit_data = force_mag[mask]

    A_phi, B_phi, phi_pred, R2_phi = fit_potential_inverse_r(r_fit, phi_fit_data)

    A_force, slope_force, p_force, R2_force_log = fit_power_law(r_fit, f_fit_data)

    result = {
        "R2_phi_1_over_r": R2_phi,
        "A_phi": A_phi,
        "B_phi": B_phi,
        "p_force": p_force,
        "R2_force_power_log": R2_force_log,
    }

    return result


# -----------------------------
# PARTICLE DYNAMICS
# -----------------------------

def integrate_particle(field, b, v0):
    """
    Formal:
        d2x/dt2 = F_x
        d2y/dt2 = F_y

    Microsoft/plain:
        acceleration = force_from_poisson_field

    Normal:
        A test particle moves through the solved field.
    """

    source_radius = field["source_radius"]

    x = X_START
    y = float(b)

    vx = float(v0)
    vy = 0.0

    ax, ay = acceleration_from_field(field, x, y)

    min_r = math.sqrt(x * x + y * y)

    captured = False
    reached = False

    max_steps = int(4.0 * (X_END - X_START) / max(v0, 1e-9) / DT)

    for _ in range(max_steps):
        rr = math.sqrt(x * x + y * y)

        if rr < min_r:
            min_r = rr

        if rr < 0.75 * source_radius:
            captured = True
            break

        if x >= X_END:
            reached = True
            break

        x_new = x + vx * DT + 0.5 * ax * DT * DT
        y_new = y + vy * DT + 0.5 * ay * DT * DT

        ax_new, ay_new = acceleration_from_field(field, x_new, y_new)

        vx_new = vx + 0.5 * (ax + ax_new) * DT
        vy_new = vy + 0.5 * (ay + ay_new) * DT

        x, y = x_new, y_new
        vx, vy = vx_new, vy_new
        ax, ay = ax_new, ay_new

    theta_rad = abs(math.atan2(vy, vx))
    theta_deg = math.degrees(theta_rad)

    usable = 1 if reached and (not captured) and theta_rad > 0 else 0

    return {
        "theta_rad": theta_rad,
        "theta_deg": theta_deg,
        "min_r": min_r,
        "captured": int(captured),
        "reached": int(reached),
        "usable": usable,
        "final_x": x,
        "final_y": y,
        "final_vx": vx,
        "final_vy": vy,
    }


# -----------------------------
# CASE ANALYSIS
# -----------------------------

def summarize_scattering(rows):
    usable_rows = [row for row in rows if row["usable"] == 1]

    if len(usable_rows) < 3:
        return {
            "usable_count": len(usable_rows),
            "R2_1_over_b": float("nan"),
            "p_scatter": float("nan"),
            "R2_power_log": float("nan"),
            "Q_case": float("nan"),
            "C_mean": float("nan"),
            "C_std": float("nan"),
            "C_cv": float("nan"),
            "max_theta_deg": float("nan"),
            "regime": "not_enough_data",
        }

    b_arr = np.array([row["b"] for row in usable_rows], dtype=float)
    theta_arr = np.array([row["theta_rad"] for row in usable_rows], dtype=float)

    x_1_over_b = 1.0 / b_arr

    Q_case, theta_pred, R2_1_over_b = fit_through_origin(x_1_over_b, theta_arr)

    A_power, slope_power, p_scatter, R2_power_log = fit_power_law(b_arr, theta_arr)

    C_arr = np.array(
        [
            row["theta_rad"] * row["b"] * row["v0"] ** 2 / row["mu_eff"]
            for row in usable_rows
            if row["mu_eff"] > 0
        ],
        dtype=float,
    )

    C_mean = float(np.mean(C_arr))
    C_std = float(np.std(C_arr))
    C_cv = C_std / C_mean if C_mean != 0 else float("nan")

    max_theta_deg = float(np.max([row["theta_deg"] for row in usable_rows]))

    if R2_1_over_b > 0.95 and 0.75 <= p_scatter <= 1.35:
        regime = "inverse_b_pass"
    else:
        regime = "inverse_b_fail"

    return {
        "usable_count": len(usable_rows),
        "R2_1_over_b": R2_1_over_b,
        "p_scatter": p_scatter,
        "R2_power_log": R2_power_log,
        "Q_case": Q_case,
        "C_mean": C_mean,
        "C_std": C_std,
        "C_cv": C_cv,
        "max_theta_deg": max_theta_deg,
        "regime": regime,
    }


# -----------------------------
# PLOTTING
# -----------------------------

def plot_global_collapse(path_rows, Q_global):
    usable = [row for row in path_rows if row["usable"] == 1]

    x = np.array(
        [row["mu_eff"] / (row["b"] * row["v0"] ** 2) for row in usable],
        dtype=float,
    )
    y = np.array([row["theta_rad"] for row in usable], dtype=float)

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, s=25)

    if len(x) > 0:
        x_line = np.linspace(0, np.max(x) * 1.05, 200)
        y_line = Q_global * x_line
        plt.plot(x_line, y_line, label="fit: theta = %.4f*mu_eff/(b*v0^2)" % Q_global)

    plt.xlabel("mu_eff/(b*v0^2)")
    plt.ylabel("|deflection angle| in radians")
    plt.title("Toy Model 4b: global Poisson-field scaling collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4b_global_collapse.png", dpi=200)
    plt.show()

    print("Saved plot: toy_model_4b_global_collapse.png")


def plot_selected_deflection_curves(path_rows):
    selected = [
        (4.0, 2.0, 1.0, 2.0),
        (4.0, 8.0, 0.25, 2.8),
        (6.0, 4.0, 0.35, 2.8),
        (8.0, 8.0, 0.25, 2.8),
    ]

    plt.figure(figsize=(12, 7))

    for radius, alpha, beta, v0 in selected:
        subset = [
            row for row in path_rows
            if row["radius"] == radius
            and row["alpha"] == alpha
            and row["beta"] == beta
            and row["v0"] == v0
            and row["usable"] == 1
        ]

        if len(subset) == 0:
            continue

        subset = sorted(subset, key=lambda row: row["b"])

        b_arr = [row["b"] for row in subset]
        theta_deg = [row["theta_deg"] for row in subset]

        label = "r=%.0f, a=%.1f, beta=%.2f, v0=%.1f" % (radius, alpha, beta, v0)

        plt.plot(b_arr, theta_deg, marker="o", label=label)

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 4b: selected Poisson-field deflection curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4b_selected_deflection_curves.png", dpi=200)
    plt.show()

    print("Saved plot: toy_model_4b_selected_deflection_curves.png")


def plot_C_vs_b(path_rows):
    selected = [
        (4.0, 8.0, 0.25, 2.8),
        (6.0, 8.0, 0.25, 2.8),
        (8.0, 8.0, 0.25, 2.8),
    ]

    plt.figure(figsize=(12, 7))

    for radius, alpha, beta, v0 in selected:
        subset = [
            row for row in path_rows
            if row["radius"] == radius
            and row["alpha"] == alpha
            and row["beta"] == beta
            and row["v0"] == v0
            and row["usable"] == 1
        ]

        if len(subset) == 0:
            continue

        subset = sorted(subset, key=lambda row: row["b"])

        b_arr = np.array([row["b"] for row in subset], dtype=float)
        C_arr = np.array(
            [
                row["theta_rad"] * row["b"] * row["v0"] ** 2 / row["mu_eff"]
                for row in subset
            ],
            dtype=float,
        )

        label = "r=%.0f, a=%.1f, beta=%.2f, v0=%.1f" % (radius, alpha, beta, v0)

        plt.plot(b_arr, C_arr, marker="o", label=label)

    plt.xlabel("impact parameter b")
    plt.ylabel("C = theta*b*v0^2/mu_eff")
    plt.title("Toy Model 4b: collapse constant C")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4b_C_vs_b.png", dpi=200)
    plt.show()

    print("Saved plot: toy_model_4b_C_vs_b.png")


def plot_field_power_hist(summary_rows):
    p_values = [
        row["p_force"]
        for row in summary_rows
        if not math.isnan(row["p_force"])
    ]

    plt.figure(figsize=(10, 7))
    plt.hist(p_values, bins=15)
    plt.xlabel("force power p")
    plt.ylabel("count")
    plt.title("Toy Model 4b: Poisson force power distribution")
    plt.tight_layout()
    plt.savefig("toy_model_4b_force_power_histogram.png", dpi=200)
    plt.show()

    print("Saved plot: toy_model_4b_force_power_histogram.png")


def plot_representative_field(field):
    r = field["r"]
    phi = field["phi"]
    force_mag = field["force_mag"]
    source_radius = field["source_radius"]

    mask = (r >= max(4.0 * source_radius, 18.0)) & (r <= 0.65 * RMAX)

    r_fit = r[mask]
    phi_fit = phi[mask]

    A_phi, B_phi, phi_pred, R2_phi = fit_potential_inverse_r(r_fit, phi_fit)

    plt.figure(figsize=(10, 7))
    plt.plot(r_fit, phi_fit, marker="o", label="solved phi")
    plt.plot(r_fit, phi_pred, label="fit: A/r + B")
    plt.xlabel("r")
    plt.ylabel("Phi(r)")
    plt.title("Toy Model 4b: representative potential fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4b_representative_potential_fit.png", dpi=200)
    plt.show()

    print("Saved plot: toy_model_4b_representative_potential_fit.png")

    A_force, slope_force, p_force, R2_force = fit_power_law(r_fit, force_mag[mask])

    force_pred = A_force * (r_fit ** slope_force)

    plt.figure(figsize=(10, 7))
    plt.loglog(r_fit, force_mag[mask], marker="o", label="force magnitude")
    plt.loglog(r_fit, force_pred, label="fit: A/r^%.3f" % p_force)
    plt.xlabel("r")
    plt.ylabel("force magnitude")
    plt.title("Toy Model 4b: representative force power law")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4b_representative_force_power.png", dpi=200)
    plt.show()

    print("Saved plot: toy_model_4b_representative_force_power.png")


# -----------------------------
# MAIN RUN
# -----------------------------

def main():
    print("")
    print("==============================================================")
    print("TOY MODEL 4b: ROBUSTNESS SCAN")
    print("Network source -> Poisson field -> force -> dynamic bending")
    print("==============================================================")
    print("")
    print("Formal:")
    print("    mu_eff = gamma*|delta_K_excess|")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    F = -grad(Phi)")
    print("    |theta| proportional to mu_eff/(b*v0^2)")
    print("")
    print("Microsoft/plain:")
    print("    mu_eff = gamma*abs(delta_K_excess)")
    print("    laplacian(phi) = 4*pi*network_density")
    print("    force = -gradient(phi)")
    print("    abs(deflection_angle) ~ mu_eff/(b*v0^2)")
    print("")
    print("Normal:")
    print("    We test whether the Toy Model 4 bridge works")
    print("    for many different network defects.")
    print("==============================================================")
    print("")
    print("Radii:", RADII)
    print("Alpha values:", ALPHAS)
    print("Beta values:", BETAS)
    print("v0 values:", V0_VALUES)
    print("b values:", B_VALUES)
    print("")

    all_path_rows = []
    summary_rows = []

    representative_field = None

    total_network_cases = len(RADII) * len(ALPHAS) * len(BETAS)
    network_counter = 0

    for radius in RADII:
        for alpha in ALPHAS:
            for beta in BETAS:
                network_counter += 1

                dK = deltaK_excess(alpha, beta, radius)
                mu_eff = GAMMA * abs(dK)

                print("--------------------------------------------------------------")
                print(
                    "Network case %d/%d: r=%.0f, alpha=%.2f, beta=%.2f"
                    % (network_counter, total_network_cases, radius, alpha, beta)
                )
                print("delta_K_excess = %.12f" % dK)
                print("mu_eff = %.12f" % mu_eff)

                field = solve_radial_poisson(mu_eff, radius)
                field_metrics = evaluate_field_shape(field)

                if radius == 6 and alpha == 8.0 and beta == 0.25:
                    representative_field = field

                print(
                    "Field: R2_phi=%.8f, p_force=%.6f, R2_force_log=%.8f"
                    % (
                        field_metrics["R2_phi_1_over_r"],
                        field_metrics["p_force"],
                        field_metrics["R2_force_power_log"],
                    )
                )

                for v0 in V0_VALUES:
                    case_rows = []

                    print("  Running trajectories for v0 = %.2f" % v0)

                    for b in B_VALUES:
                        traj = integrate_particle(field, b, v0)

                        row = {
                            "radius": float(radius),
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "v0": float(v0),
                            "b": float(b),
                            "delta_K_excess": float(dK),
                            "mu_eff": float(mu_eff),
                            "theta_rad": float(traj["theta_rad"]),
                            "theta_deg": float(traj["theta_deg"]),
                            "min_r": float(traj["min_r"]),
                            "captured": int(traj["captured"]),
                            "reached": int(traj["reached"]),
                            "usable": int(traj["usable"]),
                            "R2_phi_1_over_r": float(field_metrics["R2_phi_1_over_r"]),
                            "p_force": float(field_metrics["p_force"]),
                            "R2_force_power_log": float(field_metrics["R2_force_power_log"]),
                        }

                        all_path_rows.append(row)
                        case_rows.append(row)

                    scatter_summary = summarize_scattering(case_rows)

                    summary = {
                        "radius": float(radius),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "v0": float(v0),
                        "delta_K_excess": float(dK),
                        "mu_eff": float(mu_eff),
                        "R2_phi_1_over_r": float(field_metrics["R2_phi_1_over_r"]),
                        "p_force": float(field_metrics["p_force"]),
                        "R2_force_power_log": float(field_metrics["R2_force_power_log"]),
                        "usable_count": int(scatter_summary["usable_count"]),
                        "R2_1_over_b": float(scatter_summary["R2_1_over_b"]),
                        "p_scatter": float(scatter_summary["p_scatter"]),
                        "R2_power_log": float(scatter_summary["R2_power_log"]),
                        "Q_case": float(scatter_summary["Q_case"]),
                        "C_mean": float(scatter_summary["C_mean"]),
                        "C_std": float(scatter_summary["C_std"]),
                        "C_cv": float(scatter_summary["C_cv"]),
                        "max_theta_deg": float(scatter_summary["max_theta_deg"]),
                        "regime": scatter_summary["regime"],
                    }

                    summary_rows.append(summary)

                    print(
                        "  v0=%.2f | regime=%s | usable=%d | R2_1/b=%.6f | p=%.6f | C_mean=%.6f | C_cv=%.6f"
                        % (
                            v0,
                            summary["regime"],
                            summary["usable_count"],
                            summary["R2_1_over_b"],
                            summary["p_scatter"],
                            summary["C_mean"],
                            summary["C_cv"],
                        )
                    )

    save_csv(PATH_CSV, all_path_rows)
    save_csv(SUMMARY_CSV, summary_rows)

    # -----------------------------
    # GLOBAL COLLAPSE
    # -----------------------------

    usable = [row for row in all_path_rows if row["usable"] == 1]

    x_global = np.array(
        [row["mu_eff"] / (row["b"] * row["v0"] ** 2) for row in usable],
        dtype=float,
    )

    theta_global = np.array(
        [row["theta_rad"] for row in usable],
        dtype=float,
    )

    Q_global, theta_pred_global, R2_global = fit_through_origin(x_global, theta_global)

    C_global = np.array(
        [
            row["theta_rad"] * row["b"] * row["v0"] ** 2 / row["mu_eff"]
            for row in usable
            if row["mu_eff"] > 0
        ],
        dtype=float,
    )

    C_mean = float(np.mean(C_global))
    C_std = float(np.std(C_global))
    C_cv = C_std / C_mean if C_mean != 0 else float("nan")

    pass_count = sum(1 for row in summary_rows if row["regime"] == "inverse_b_pass")
    total_cases = len(summary_rows)

    p_force_values = np.array([row["p_force"] for row in summary_rows], dtype=float)
    p_scatter_values = np.array(
        [row["p_scatter"] for row in summary_rows if not math.isnan(row["p_scatter"])],
        dtype=float,
    )

    print("")
    print("==============================================================")
    print("FINAL TOY MODEL 4b SUMMARY")
    print("==============================================================")
    print("")
    print("Formal:")
    print("    mu_eff = gamma*|delta_K_excess|")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    |F(r)| ~ 1/r^2")
    print("    |theta| ~ mu_eff/(b*v0^2)")
    print("")
    print("Microsoft/plain:")
    print("    mu_eff = gamma*abs(delta_K_excess)")
    print("    laplacian(phi) = 4*pi*network_density")
    print("    force ~ 1/r^2")
    print("    abs(deflection_angle) ~ mu_eff/(b*v0^2)")
    print("")
    print("Normal:")
    print("    We checked whether the network source produces")
    print("    the correct field and bending across many defects.")
    print("")
    print("Cases:")
    print("    total network+velocity cases =", total_cases)
    print("    inverse_b_pass =", str(pass_count) + "/" + str(total_cases))
    print("")
    print("Field robustness:")
    print("    mean p_force =", float(np.mean(p_force_values)))
    print("    std  p_force =", float(np.std(p_force_values)))
    print("    expected p_force close to 2")
    print("")
    print("Scattering robustness:")
    if len(p_scatter_values) > 0:
        print("    mean p_scatter =", float(np.mean(p_scatter_values)))
        print("    std  p_scatter =", float(np.std(p_scatter_values)))
        print("    expected p_scatter close to 1")
    print("")
    print("Global collapse:")
    print("    theta = Q*mu_eff/(b*v0^2)")
    print("    Q_global =", Q_global)
    print("    R2_global =", R2_global)
    print("")
    print("Collapse constant:")
    print("    C = theta*b*v0^2/mu_eff")
    print("    C_mean =", C_mean)
    print("    C_std  =", C_std)
    print("    C_cv   =", C_cv)
    print("")

    if R2_global > 0.95 and pass_count >= int(0.8 * total_cases):
        print("RESULT: PASS")
        print("Toy Model 4b supports the full bridge:")
        print("network curvature source -> Poisson field -> gravity-like bending")
    else:
        print("RESULT: MIXED")
        print("Some cases did not cleanly obey the expected scaling.")
        print("Check the CSV files and plots.")
    print("==============================================================")

    # -----------------------------
    # PLOTS
    # -----------------------------

    plot_global_collapse(all_path_rows, Q_global)
    plot_selected_deflection_curves(all_path_rows)
    plot_C_vs_b(all_path_rows)
    plot_field_power_hist(summary_rows)

    if representative_field is not None:
        plot_representative_field(representative_field)


if __name__ == "__main__":
    main()
