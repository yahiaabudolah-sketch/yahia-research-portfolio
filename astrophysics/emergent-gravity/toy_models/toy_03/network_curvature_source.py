import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# TOY MODEL 3c:
# ENTANGLEMENT-CURVATURE SOURCE -> DYNAMIC BENDING
# ============================================================
#
# We now connect Toy Model 1 and Toy Model 3.
#
# Toy Model 1 gave:
#
# Formal:
#   delta_K_excess(alpha,beta,r)
#
# Microsoft/plain:
#   delta_K_excess = network curvature response
#
# Normal:
#   The defect changes the graph curvature.
#
#
# Toy Model 3b gave:
#
# Formal:
#   |delta_theta| proportional to mu/(b*v0^2)
#
# Microsoft/plain:
#   abs(deflection_angle) ~ mu/(b*v0^2)
#
# Normal:
#   Dynamic inverse-radius scattering behaves like gravity.
#
#
# Toy Model 3c tests:
#
# Formal:
#   mu_eff = gamma * |delta_K_excess|
#
# Microsoft/plain:
#   mu_eff = gamma*abs(delta_K_excess)
#
# Normal:
#   The gravitational source strength comes from the entanglement
#   curvature response.
#
#
# Final test:
#
# Formal:
#   |delta_theta| proportional to |delta_K_excess|/(b*v0^2)
#
# Microsoft/plain:
#   abs(deflection_angle) ~ abs(delta_K_excess)/(b*v0^2)
#
# Normal:
#   If this works, the network defect source controls the dynamic bending.


EPSILON = 1e-12


# ============================================================
# 1. Closed-form Toy Model 1 curvature law
# ============================================================

# These are the baseline-corrected closed-form coefficients
# we derived from the node-motif curvature law.
#
# Formal:
#   delta_K_excess =
#       c_beta(r)*(sqrt(beta)-1)
#     + c_inv_beta(r)*(1/sqrt(beta)-1)
#     + c_ab(r)*(sqrt(alpha/beta)-1)
#     + c_ba(r)*(sqrt(beta/alpha)-1)
#
# Microsoft/plain:
#   delta_K_excess =
#       c_beta*(sqrt(beta)-1)
#     + c_inv_beta*(1/sqrt(beta)-1)
#     + c_ab*(sqrt(alpha/beta)-1)
#     + c_ba*(sqrt(beta/alpha)-1)
#
# Normal:
#   This gives the curvature change caused by the defect.
#   At alpha=1 and beta=1, delta_K_excess = 0.

CLOSED_FORM_COEFFS = {
    4: {
        "c_beta": -0.371681415929,
        "c_inv_beta": -0.371681415929,
        "c_ab": -0.300884955752,
        "c_ba": -0.300884955752
    },
    5: {
        "c_beta": -0.362416107383,
        "c_inv_beta": -0.362416107383,
        "c_ab": -0.308724832215,
        "c_ba": -0.308724832215
    },
    6: {
        "c_beta": -0.314720812183,
        "c_inv_beta": -0.314720812183,
        "c_ab": -0.274111675127,
        "c_ba": -0.274111675127
    },
    7: {
        "c_beta": -0.276679841897,
        "c_inv_beta": -0.276679841897,
        "c_ab": -0.245059288538,
        "c_ba": -0.245059288538
    },
    8: {
        "c_beta": -0.258675078864,
        "c_inv_beta": -0.258675078864,
        "c_ab": -0.233438485804,
        "c_ba": -0.233438485804
    }
}


def delta_K_excess(alpha, beta, radius):
    """
    Formal:
        delta_K_excess(alpha,beta,r)

    Microsoft/plain:
        delta_K_excess = closed_form_curvature_response(alpha,beta,radius)

    Normal:
        This is the exact Toy Model 1 curvature source term.
    """

    r_int = int(radius)

    if r_int not in CLOSED_FORM_COEFFS:
        raise ValueError("Radius must be one of: 4, 5, 6, 7, 8")

    c = CLOSED_FORM_COEFFS[r_int]

    sqrt_alpha = np.sqrt(alpha)
    sqrt_beta = np.sqrt(beta)

    term_beta = sqrt_beta - 1.0
    term_inv_beta = 1.0 / sqrt_beta - 1.0
    term_ab = np.sqrt(alpha / beta) - 1.0
    term_ba = np.sqrt(beta / alpha) - 1.0

    value = (
        c["c_beta"] * term_beta
        + c["c_inv_beta"] * term_inv_beta
        + c["c_ab"] * term_ab
        + c["c_ba"] * term_ba
    )

    return value


def mu_effective(alpha, beta, radius, gamma):
    """
    Formal:
        mu_eff = gamma * |delta_K_excess|

    Microsoft/plain:
        mu_eff = gamma*abs(delta_K_excess)

    Normal:
        The source strength is proportional to the size of the
        entanglement-curvature disturbance.
    """

    dK = delta_K_excess(alpha, beta, radius)

    return gamma * abs(dK)


# ============================================================
# 2. Dynamic inverse-radius potential
# ============================================================

def center_of_grid(n):
    return n / 2.0, n / 2.0


def radius_from_center(x, y, n):
    cx, cy = center_of_grid(n)

    dx = x - cx
    dy = y - cy

    return np.sqrt(dx*dx + dy*dy)


def potential_inverse_radius(x, y, n, mu, softening):
    """
    Formal:
        U(r) = -mu/sqrt(r^2+s^2)

    Microsoft/plain:
        U = -mu/sqrt(r^2 + softening^2)

    Normal:
        This is the dynamic potential used in Toy Model 3.
    """

    r = radius_from_center(x, y, n)

    return -mu / np.sqrt(r*r + softening*softening)


def acceleration_inverse_radius(x, y, n, mu, softening):
    """
    Formal:
        a = -grad(U)

        ax = -mu*(x-cx)/(r^2+s^2)^(3/2)
        ay = -mu*(y-cy)/(r^2+s^2)^(3/2)

    Microsoft/plain:
        ax = -mu*dx/(r^2 + softening^2)^(3/2)
        ay = -mu*dy/(r^2 + softening^2)^(3/2)

    Normal:
        The particle accelerates toward the defect center.
    """

    cx, cy = center_of_grid(n)

    dx = x - cx
    dy = y - cy

    r2 = dx*dx + dy*dy

    denom = (r2 + softening*softening)**1.5

    if denom < EPSILON:
        return 0.0, 0.0

    ax = -mu * dx / denom
    ay = -mu * dy / denom

    return ax, ay


# ============================================================
# 3. Time integration
# ============================================================

def derivatives(state, params):
    """
    state = [x, y, vx, vy]

    Formal:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = ax
        dvy/dt = ay

    Microsoft/plain:
        position changes by velocity
        velocity changes by acceleration

    Normal:
        This evolves the particle through time.
    """

    x, y, vx, vy = state

    ax, ay = acceleration_inverse_radius(
        x=x,
        y=y,
        n=params["n"],
        mu=params["mu_eff"],
        softening=params["softening"]
    )

    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state, dt, params):
    """
    Microsoft/plain:
        RK4 time step

    Normal:
        A stable time integrator.
    """

    k1 = derivatives(state, params)
    k2 = derivatives(state + 0.5*dt*k1, params)
    k3 = derivatives(state + 0.5*dt*k2, params)
    k4 = derivatives(state + dt*k3, params)

    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def energy(state, params):
    x, y, vx, vy = state

    kinetic = 0.5*(vx*vx + vy*vy)

    potential = potential_inverse_radius(
        x=x,
        y=y,
        n=params["n"],
        mu=params["mu_eff"],
        softening=params["softening"]
    )

    return kinetic + potential


# ============================================================
# 4. One trajectory
# ============================================================

def simulate_trajectory(params, b):
    """
    Formal:
        x(0) = x_start
        y(0) = y_center + b
        vx(0) = v0
        vy(0) = 0

    Microsoft/plain:
        start left
        move right
        impact parameter = b

    Normal:
        A particle flies past the entanglement defect.
    """

    n = params["n"]
    cx, cy = center_of_grid(n)

    state = np.array([
        params["x_start"],
        cy + b,
        params["v0"],
        0.0
    ], dtype=float)

    dt = params["dt"]

    xs = []
    ys = []
    radii = []

    entered_core = False
    reached_right = False
    escaped_bounds = False

    initial_energy = energy(state, params)

    for step in range(params["max_steps"]):
        x, y, vx, vy = state

        r = radius_from_center(x, y, n)

        xs.append(x)
        ys.append(y)
        radii.append(r)

        if r <= params["core_radius"]:
            entered_core = True

        if x >= params["x_end"] and vx > 0:
            reached_right = True
            break

        if x < params["x_min_escape"] or x > params["x_max_escape"]:
            escaped_bounds = True
            break

        if y < params["y_min_escape"] or y > params["y_max_escape"]:
            escaped_bounds = True
            break

        state = rk4_step(state, dt, params)

    x_final, y_final, vx_final, vy_final = state

    theta_out = np.arctan2(vy_final, vx_final)
    abs_theta = abs(theta_out)

    min_r = float(np.min(radii))

    final_energy = energy(state, params)
    energy_error = final_energy - initial_energy

    alpha = params["alpha"]
    beta = params["beta"]
    radius = params["core_radius"]
    mu_eff = params["mu_eff"]
    v0 = params["v0"]

    if mu_eff > 0 and b > 0:
        source_scaling_x = mu_eff / (b * v0*v0)
        C = abs_theta * b * v0*v0 / mu_eff
    else:
        source_scaling_x = 0.0
        C = np.nan

    row = {
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "gamma": params["gamma"],
        "delta_K_excess": params["delta_K_excess"],
        "abs_delta_K_excess": abs(params["delta_K_excess"]),
        "mu_eff": mu_eff,
        "v0": v0,
        "softening": params["softening"],
        "b": float(b),
        "xs": xs,
        "ys": ys,
        "min_r": min_r,
        "D": float(b - min_r),
        "theta_rad": theta_out,
        "abs_theta_rad": abs_theta,
        "theta_deg": float(np.degrees(theta_out)),
        "abs_theta_deg": float(abs(np.degrees(theta_out))),
        "source_scaling_x": source_scaling_x,
        "C": C,
        "entered_core": entered_core,
        "reached_right": reached_right,
        "escaped_bounds": escaped_bounds,
        "energy_error": energy_error,
        "final_x": x_final,
        "final_y": y_final,
        "final_vx": vx_final,
        "final_vy": vy_final
    }

    return row


# ============================================================
# 5. Run cases
# ============================================================

def run_case(params, b_values):
    rows = []

    for b in b_values:
        row = simulate_trajectory(params, b)
        rows.append(row)

    return rows


def is_usable(row):
    if row["mu_eff"] <= 0:
        return False

    if not row["reached_right"]:
        return False

    if row["entered_core"]:
        return False

    if row["b"] <= row["radius"]:
        return False

    if row["abs_theta_rad"] <= 1e-10:
        return False

    return True


def collect_usable(all_rows):
    usable = []

    for rows in all_rows:
        for row in rows:
            if is_usable(row):
                usable.append(row)

    return usable


# ============================================================
# 6. Fits
# ============================================================

def r2_score(y, y_pred):
    y = np.array(y, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot <= EPSILON:
        return np.nan

    return 1.0 - ss_res/ss_tot


def fit_inverse_b(rows):
    """
    Formal:
        theta = q/b

    Microsoft/plain:
        theta = q/b

    Normal:
        Per-case inverse impact parameter check.
    """

    usable = [row for row in rows if is_usable(row)]

    if len(usable) < 3:
        return np.nan, np.nan, np.nan, len(usable)

    b = np.array([row["b"] for row in usable], dtype=float)
    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    x = 1.0 / b

    q = np.sum(x*theta) / np.sum(x*x)

    pred = q*x

    # Power-law p
    log_b = np.log(b)
    log_theta = np.log(theta)

    X = np.vstack([np.ones_like(log_b), -log_b]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, log_theta, rcond=None)

    p = coeffs[1]

    return q, r2_score(theta, pred), p, len(usable)


def fit_global_source_scaling(usable):
    """
    Formal:
        theta = Q * mu_eff/(b*v0^2)

    Microsoft/plain:
        theta = Q*mu_eff/(b*v0^2)

    Normal:
        This is the main Toy Model 3c test.
    """

    if len(usable) < 3:
        return np.nan, np.nan

    x = np.array([row["source_scaling_x"] for row in usable], dtype=float)
    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    Q = np.sum(x*theta) / np.sum(x*x)

    pred = Q*x

    R2 = r2_score(theta, pred)

    return Q, R2


def fit_direct_curvature_scaling(usable):
    """
    Formal:
        theta = QK * |delta_K_excess|/(b*v0^2)

    Microsoft/plain:
        theta = QK*abs(delta_K_excess)/(b*v0^2)

    Normal:
        Same as mu_eff scaling when gamma = 1.
        This explicitly names the entanglement-curvature source.
    """

    if len(usable) < 3:
        return np.nan, np.nan

    x = np.array([
        row["abs_delta_K_excess"] / (row["b"] * row["v0"] * row["v0"])
        for row in usable
    ], dtype=float)

    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    QK = np.sum(x*theta) / np.sum(x*x)

    pred = QK*x

    R2 = r2_score(theta, pred)

    return QK, R2


# ============================================================
# 7. Summaries
# ============================================================

def summarize_case(rows):
    usable = [row for row in rows if is_usable(row)]

    q, R2_inv, p, n_fit = fit_inverse_b(rows)

    if len(usable) > 0:
        C_values = np.array([row["C"] for row in usable], dtype=float)
        theta_values = np.array([row["abs_theta_deg"] for row in usable], dtype=float)
        energy_errors = np.array([abs(row["energy_error"]) for row in usable], dtype=float)

        C_mean = float(np.mean(C_values))
        C_std = float(np.std(C_values))
        C_cv = float(C_std / abs(C_mean)) if abs(C_mean) > EPSILON else np.nan

        max_theta = float(np.max(theta_values))
        mean_theta = float(np.mean(theta_values))
        max_energy_error = float(np.max(energy_errors))
    else:
        C_mean = np.nan
        C_std = np.nan
        C_cv = np.nan
        max_theta = 0.0
        mean_theta = 0.0
        max_energy_error = np.nan

    first = rows[0]

    if len(usable) < 3:
        regime = "no_signal"
    elif R2_inv > 0.85 and abs(p - 1.0) < 0.25:
        regime = "inverse_b_pass"
    else:
        regime = "inverse_b_fail"

    return {
        "alpha": first["alpha"],
        "beta": first["beta"],
        "radius": first["radius"],
        "gamma": first["gamma"],
        "delta_K_excess": first["delta_K_excess"],
        "abs_delta_K_excess": first["abs_delta_K_excess"],
        "mu_eff": first["mu_eff"],
        "v0": first["v0"],
        "softening": first["softening"],
        "usable": len(usable),
        "R2_inverse_b": R2_inv,
        "p_power": p,
        "C_mean": C_mean,
        "C_std": C_std,
        "C_cv": C_cv,
        "max_theta_deg": max_theta,
        "mean_theta_deg": mean_theta,
        "max_energy_error": max_energy_error,
        "regime": regime
    }


def print_case_summary(summary):
    print(
        f"r={summary['radius']:3.0f}, "
        f"alpha={summary['alpha']:4.2f}, "
        f"beta={summary['beta']:4.2f}, "
        f"v0={summary['v0']:4.2f} | "
        f"dK={summary['delta_K_excess']: .6f}, "
        f"mu_eff={summary['mu_eff']: .6f} | "
        f"regime={summary['regime']:15s} | "
        f"R2_1/b={summary['R2_inverse_b']:.6f} | "
        f"p={summary['p_power']:.6f} | "
        f"C_mean={summary['C_mean']:.6f} | "
        f"C_cv={summary['C_cv']:.6f}"
    )


# ============================================================
# 8. Saving
# ============================================================

def save_path_csv(all_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "radius",
            "gamma",
            "delta_K_excess",
            "abs_delta_K_excess",
            "mu_eff",
            "v0",
            "softening",
            "b",
            "min_r",
            "D",
            "theta_rad",
            "abs_theta_rad",
            "theta_deg",
            "abs_theta_deg",
            "source_scaling_x_mu_eff_over_bv02",
            "C_theta_b_v02_over_mu_eff",
            "entered_core",
            "reached_right",
            "escaped_bounds",
            "energy_error"
        ])

        for rows in all_rows:
            for row in rows:
                writer.writerow([
                    row["alpha"],
                    row["beta"],
                    row["radius"],
                    row["gamma"],
                    row["delta_K_excess"],
                    row["abs_delta_K_excess"],
                    row["mu_eff"],
                    row["v0"],
                    row["softening"],
                    row["b"],
                    row["min_r"],
                    row["D"],
                    row["theta_rad"],
                    row["abs_theta_rad"],
                    row["theta_deg"],
                    row["abs_theta_deg"],
                    row["source_scaling_x"],
                    row["C"],
                    row["entered_core"],
                    row["reached_right"],
                    row["escaped_bounds"],
                    row["energy_error"]
                ])

    print(f"Saved path CSV: {filename}")


def save_summary_csv(summaries, global_summary, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "radius",
            "gamma",
            "delta_K_excess",
            "abs_delta_K_excess",
            "mu_eff",
            "v0",
            "softening",
            "usable",
            "R2_inverse_b",
            "p_power",
            "C_mean",
            "C_std",
            "C_cv",
            "max_theta_deg",
            "mean_theta_deg",
            "max_energy_error",
            "regime",
            "global_Q_mu_eff",
            "global_R2_mu_eff",
            "global_Q_curvature",
            "global_R2_curvature",
            "global_C_mean",
            "global_C_std",
            "global_C_cv"
        ])

        for s in summaries:
            writer.writerow([
                s["alpha"],
                s["beta"],
                s["radius"],
                s["gamma"],
                s["delta_K_excess"],
                s["abs_delta_K_excess"],
                s["mu_eff"],
                s["v0"],
                s["softening"],
                s["usable"],
                s["R2_inverse_b"],
                s["p_power"],
                s["C_mean"],
                s["C_std"],
                s["C_cv"],
                s["max_theta_deg"],
                s["mean_theta_deg"],
                s["max_energy_error"],
                s["regime"],
                global_summary["Q_mu_eff"],
                global_summary["R2_mu_eff"],
                global_summary["Q_curvature"],
                global_summary["R2_curvature"],
                global_summary["C_mean"],
                global_summary["C_std"],
                global_summary["C_cv"]
            ])

    print(f"Saved summary CSV: {filename}")


# ============================================================
# 9. Plotting
# ============================================================

def case_label(summary):
    return (
        f"r={int(summary['radius'])}, "
        f"a={summary['alpha']}, "
        f"bta={summary['beta']}, "
        f"v0={summary['v0']}"
    )


def plot_deltaK_to_mu(summaries):
    x = np.array([s["abs_delta_K_excess"] for s in summaries], dtype=float)
    y = np.array([s["mu_eff"] for s in summaries], dtype=float)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=35)

    plt.xlabel("|delta_K_excess|")
    plt.ylabel("mu_eff")
    plt.title("Toy Model 3c: source strength from curvature response")
    plt.tight_layout()
    plt.savefig("toy_model_3c_mu_eff_vs_deltaK.png", dpi=200)
    print("Saved plot: toy_model_3c_mu_eff_vs_deltaK.png")
    plt.show()


def plot_global_collapse(usable, Q):
    x = np.array([row["source_scaling_x"] for row in usable], dtype=float)
    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    plt.figure(figsize=(8, 6))

    plt.scatter(x, theta, s=25)

    x_line = np.linspace(0.0, max(x)*1.05, 200)
    theta_line = Q*x_line

    plt.plot(
        x_line,
        theta_line,
        linewidth=2,
        label=f"fit: theta = {Q:.4f}*mu_eff/(b*v0^2)"
    )

    plt.xlabel("mu_eff/(b*v0^2)")
    plt.ylabel("|deflection angle| in radians")
    plt.title("Toy Model 3c: dynamic collapse using network source")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3c_mu_eff_scaling_collapse.png", dpi=200)
    print("Saved plot: toy_model_3c_mu_eff_scaling_collapse.png")
    plt.show()


def plot_curvature_collapse(usable, QK):
    x = np.array([
        row["abs_delta_K_excess"] / (row["b"] * row["v0"] * row["v0"])
        for row in usable
    ], dtype=float)

    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    plt.figure(figsize=(8, 6))

    plt.scatter(x, theta, s=25)

    x_line = np.linspace(0.0, max(x)*1.05, 200)
    theta_line = QK*x_line

    plt.plot(
        x_line,
        theta_line,
        linewidth=2,
        label=f"fit: theta = {QK:.4f}*|deltaK|/(b*v0^2)"
    )

    plt.xlabel("|delta_K_excess|/(b*v0^2)")
    plt.ylabel("|deflection angle| in radians")
    plt.title("Toy Model 3c: direct curvature-source scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3c_curvature_scaling_collapse.png", dpi=200)
    print("Saved plot: toy_model_3c_curvature_scaling_collapse.png")
    plt.show()


def plot_selected_C_vs_b(all_rows, summaries, selected_indices):
    plt.figure(figsize=(11, 6))

    for idx in selected_indices:
        rows = all_rows[idx]
        summary = summaries[idx]

        usable = [row for row in rows if is_usable(row)]

        b = [row["b"] for row in usable]
        C = [row["C"] for row in usable]

        plt.plot(b, C, marker="o", label=case_label(summary))

    plt.xlabel("impact parameter b")
    plt.ylabel("C = theta*b*v0^2/mu_eff")
    plt.title("Toy Model 3c: collapse constant C")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3c_C_vs_b.png", dpi=200)
    print("Saved plot: toy_model_3c_C_vs_b.png")
    plt.show()


def plot_selected_deflection_vs_b(all_rows, summaries, selected_indices):
    plt.figure(figsize=(11, 6))

    for idx in selected_indices:
        rows = all_rows[idx]
        summary = summaries[idx]

        b = [row["b"] for row in rows]
        theta = [row["abs_theta_deg"] for row in rows]

        plt.plot(b, theta, marker="o", label=case_label(summary))

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 3c: selected dynamic deflection curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3c_selected_deflection_vs_b.png", dpi=200)
    print("Saved plot: toy_model_3c_selected_deflection_vs_b.png")
    plt.show()


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    # Domain.
    n = 141

    # Coupling between curvature response and effective source strength.
    #
    # Formal:
    #   mu_eff = gamma*|delta_K_excess|
    #
    # Microsoft/plain:
    #   mu_eff = gamma*abs(delta_K_excess)
    #
    # Normal:
    #   gamma sets the conversion scale between graph curvature response
    #   and dynamic source strength.
    gamma = 1.0

    base_params = {
        "n": n,
        "x_start": 0.0,
        "x_end": float(n - 1),
        "x_min_escape": -0.5*n,
        "x_max_escape": 1.5*n,
        "y_min_escape": -0.5*n,
        "y_max_escape": 1.5*n,
        "dt": 0.05,
        "max_steps": 30000,
        "gamma": gamma
    }

    # Entanglement-defect parameter scan.
    radius_values = [4, 6, 8]
    alpha_values = [2.0, 4.0, 8.0]
    beta_values = [1.0, 0.6, 0.35, 0.25]
    v0_values = [2.0, 2.8]

    # Use b values safely outside all cores.
    b_values = [14, 18, 22, 26, 30, 36, 42, 50, 58]

    parameter_cases = []

    for radius in radius_values:
        for alpha in alpha_values:
            for beta in beta_values:
                dK = delta_K_excess(alpha, beta, radius)
                mu_eff = mu_effective(alpha, beta, radius, gamma)

                for v0 in v0_values:
                    params = dict(base_params)

                    params["core_radius"] = float(radius)
                    params["softening"] = float(radius)
                    params["alpha"] = alpha
                    params["beta"] = beta
                    params["delta_K_excess"] = dK
                    params["mu_eff"] = mu_eff
                    params["v0"] = v0

                    parameter_cases.append(params)

    all_rows = []
    summaries = []

    print("\n==============================================================")
    print("TOY MODEL 3c: ENTANGLEMENT-CURVATURE SOURCE TEST")
    print("==============================================================")
    print(f"Grid/domain size: {n}")
    print(f"gamma: {gamma}")
    print(f"radii: {radius_values}")
    print(f"alpha values: {alpha_values}")
    print(f"beta values: {beta_values}")
    print(f"v0 values: {v0_values}")
    print(f"b values: {b_values}")
    print("==============================================================")

    for i, params in enumerate(parameter_cases):
        print("\n--------------------------------------------------------------")
        print(f"Running case {i + 1}/{len(parameter_cases)}")
        print(
            f"r={params['core_radius']}, "
            f"alpha={params['alpha']}, "
            f"beta={params['beta']}, "
            f"v0={params['v0']}, "
            f"dK={params['delta_K_excess']:.6f}, "
            f"mu_eff={params['mu_eff']:.6f}"
        )
        print("--------------------------------------------------------------")

        rows = run_case(params, b_values)
        summary = summarize_case(rows)

        print_case_summary(summary)

        all_rows.append(rows)
        summaries.append(summary)

    usable = collect_usable(all_rows)

    Q_mu_eff, R2_mu_eff = fit_global_source_scaling(usable)
    Q_curvature, R2_curvature = fit_direct_curvature_scaling(usable)

    C_values = np.array([row["C"] for row in usable], dtype=float)

    global_summary = {
        "Q_mu_eff": Q_mu_eff,
        "R2_mu_eff": R2_mu_eff,
        "Q_curvature": Q_curvature,
        "R2_curvature": R2_curvature,
        "C_mean": float(np.mean(C_values)),
        "C_std": float(np.std(C_values)),
        "C_cv": float(np.std(C_values) / abs(np.mean(C_values)))
    }

    save_path_csv(
        all_rows=all_rows,
        filename="toy_model_3c_entanglement_source_path_data.csv"
    )

    save_summary_csv(
        summaries=summaries,
        global_summary=global_summary,
        filename="toy_model_3c_entanglement_source_summary.csv"
    )

    # Select representative cases for plots.
    selected_indices = [
        0,
        3,
        9,
        17,
        33,
        55,
        len(summaries) - 1
    ]

    selected_indices = [idx for idx in selected_indices if idx < len(summaries)]

    plot_deltaK_to_mu(summaries)

    plot_global_collapse(
        usable=usable,
        Q=Q_mu_eff
    )

    plot_curvature_collapse(
        usable=usable,
        QK=Q_curvature
    )

    plot_selected_C_vs_b(
        all_rows=all_rows,
        summaries=summaries,
        selected_indices=selected_indices
    )

    plot_selected_deflection_vs_b(
        all_rows=all_rows,
        summaries=summaries,
        selected_indices=selected_indices
    )

    print("\n==============================================================")
    print("FINAL TOY MODEL 3c SUMMARY")
    print("==============================================================")

    print("")
    print("Formal:")
    print("    mu_eff = gamma*|delta_K_excess|")
    print("    |delta_theta| proportional to mu_eff/(b*v0^2)")
    print("")
    print("Microsoft/plain:")
    print("    mu_eff = gamma*abs(delta_K_excess)")
    print("    abs(deflection_angle) ~ mu_eff/(b*v0^2)")
    print("")
    print("Normal:")
    print("    We test whether the entanglement-network curvature response")
    print("    can act as the source strength for dynamic bending.")
    print("")

    print(f"Total cases: {len(summaries)}")
    print(f"Total usable trajectories: {len(usable)}")
    print("")

    print("Global mu_eff scaling:")
    print("    theta = Q * mu_eff/(b*v0^2)")
    print(f"    Q_mu_eff = {global_summary['Q_mu_eff']:.8f}")
    print(f"    R2_mu_eff = {global_summary['R2_mu_eff']:.8f}")
    print("")

    print("Direct curvature scaling:")
    print("    theta = QK * |delta_K_excess|/(b*v0^2)")
    print(f"    Q_curvature = {global_summary['Q_curvature']:.8f}")
    print(f"    R2_curvature = {global_summary['R2_curvature']:.8f}")
    print("")

    print("Collapse constant:")
    print("    C = theta*b*v0^2/mu_eff")
    print(f"    C_mean = {global_summary['C_mean']:.8f}")
    print(f"    C_std  = {global_summary['C_std']:.8f}")
    print(f"    C_cv   = {global_summary['C_cv']:.8f}")
    print("")

    pass_count = sum(1 for s in summaries if s["regime"] == "inverse_b_pass")

    print("Per-case pass count:")
    print(f"    inverse_b_pass = {pass_count}/{len(summaries)}")
    print("")

    print("Representative per-case summaries:")
    print("--------------------------------------------------------------")

    for idx in selected_indices:
        print_case_summary(summaries[idx])

    print("--------------------------------------------------------------")

    if global_summary["R2_mu_eff"] > 0.9 and global_summary["C_cv"] < 0.20:
        print("RESULT: PASS")
        print("The entanglement-curvature source successfully controls")
        print("the dynamic bending through")
        print("abs(deflection_angle) ~ abs(delta_K_excess)/(b*v0^2).")
    else:
        print("RESULT: PARTIAL / CHECK")
        print("The source produces bending, but the collapse is not clean enough yet.")
        print("Possible fixes:")
        print("1. increase b range so b is much larger than radius")
        print("2. use fixed softening instead of radius-dependent softening")
        print("3. fit a radius-dependent coupling gamma(r)")

    print("==============================================================")

