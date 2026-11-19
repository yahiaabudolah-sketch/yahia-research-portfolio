import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# TOY MODEL 3b:
# DYNAMIC SCALING LAW TEST
# ============================================================
#
# Formal:
#   |delta_theta| proportional to mu/(b*v0^2)
#
# Microsoft/plain:
#   abs(deflection_angle) ~ mu/(b*v0^2)
#
# Normal:
#   Stronger source bends more.
#   Larger impact parameter bends less.
#   Faster particle bends less.
#
# Main collapse quantity:
#
# Formal:
#   C = |delta_theta|*b*v0^2/mu
#
# Microsoft/plain:
#   C = abs(deflection_angle)*b*v0^2/mu
#
# Normal:
#   If the scattering is gravity-like, C should be roughly constant.


EPSILON = 1e-12


# ============================================================
# 1. Geometry
# ============================================================

def center_of_grid(n):
    return n / 2.0, n / 2.0


def radius_from_center(x, y, n):
    cx, cy = center_of_grid(n)
    dx = x - cx
    dy = y - cy
    return np.sqrt(dx*dx + dy*dy)


# ============================================================
# 2. Inverse-radius potential and force
# ============================================================

def potential_inverse_radius(x, y, n, mu, softening):
    """
    Formal:
        U(r) = -mu/sqrt(r^2 + s^2)

    Microsoft/plain:
        U = -mu/sqrt(r^2 + softening^2)

    Normal:
        This is the Newtonian-like calibration potential.
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
        mu=params["mu"],
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
        mu=params["mu"],
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
        A particle flies past the defect.
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
    energies = []

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
        energies.append(energy(state, params))

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

    row = {
        "mu": params["mu"],
        "v0": params["v0"],
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
        "entered_core": entered_core,
        "reached_right": reached_right,
        "escaped_bounds": escaped_bounds,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_error": energy_error,
        "final_x": x_final,
        "final_y": y_final,
        "final_vx": vx_final,
        "final_vy": vy_final
    }

    if params["mu"] > 0 and b > 0:
        row["collapse_x"] = params["mu"] / (b * params["v0"]**2)
        row["C"] = abs_theta * b * params["v0"]**2 / params["mu"]
    else:
        row["collapse_x"] = 0.0
        row["C"] = np.nan

    return row


# ============================================================
# 5. Run scans
# ============================================================

def run_case(params, b_values):
    rows = []

    for b in b_values:
        row = simulate_trajectory(params, b)
        rows.append(row)

    return rows


def is_usable(row, core_radius):
    if row["mu"] <= 0:
        return False

    if not row["reached_right"]:
        return False

    if row["entered_core"]:
        return False

    if row["b"] <= core_radius:
        return False

    if row["abs_theta_rad"] <= 1e-10:
        return False

    return True


def all_usable_rows(all_rows, core_radius):
    usable = []

    for rows in all_rows:
        for row in rows:
            if is_usable(row, core_radius):
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


def fit_theta_vs_inverse_b(rows, core_radius):
    """
    Formal:
        theta = q/b

    Microsoft/plain:
        theta = q/b

    Normal:
        Checks each fixed mu,v0 case.
    """

    usable = [r for r in rows if is_usable(r, core_radius)]

    if len(usable) < 3:
        return np.nan, np.nan, len(usable)

    b = np.array([r["b"] for r in usable], dtype=float)
    theta = np.array([r["abs_theta_rad"] for r in usable], dtype=float)

    x = 1.0/b

    q = np.sum(x*theta)/np.sum(x*x)

    pred = q*x

    return q, r2_score(theta, pred), len(usable)


def fit_global_scaling(usable):
    """
    Formal:
        theta = Q * mu/(b*v0^2)

    Microsoft/plain:
        deflection_angle = Q * mu/(b*v0^2)

    Normal:
        This is the full scaling-law test.
    """

    if len(usable) < 3:
        return np.nan, np.nan

    x = np.array([r["collapse_x"] for r in usable], dtype=float)
    theta = np.array([r["abs_theta_rad"] for r in usable], dtype=float)

    Q = np.sum(x*theta)/np.sum(x*x)

    pred = Q*x

    R2 = r2_score(theta, pred)

    return Q, R2


def fit_power_law_b(rows, core_radius):
    """
    Formal:
        theta = A*b^(-p)

    Microsoft/plain:
        theta = A/b^p

    Normal:
        Gravity-like means p close to 1.
    """

    usable = [r for r in rows if is_usable(r, core_radius)]

    if len(usable) < 3:
        return np.nan, np.nan, np.nan, len(usable)

    b = np.array([r["b"] for r in usable], dtype=float)
    theta = np.array([r["abs_theta_rad"] for r in usable], dtype=float)

    log_b = np.log(b)
    log_theta = np.log(theta)

    X = np.vstack([np.ones_like(log_b), -log_b]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, log_theta, rcond=None)

    log_A = coeffs[0]
    p = coeffs[1]

    pred_log = log_A - p*log_b

    R2 = r2_score(log_theta, pred_log)

    return np.exp(log_A), p, R2, len(usable)


# ============================================================
# 7. Summaries
# ============================================================

def summarize_case(rows, params):
    q, R2_inv_b, n_fit = fit_theta_vs_inverse_b(rows, params["core_radius"])
    A, p, R2_power, n_power = fit_power_law_b(rows, params["core_radius"])

    usable = [r for r in rows if is_usable(r, params["core_radius"])]

    if len(usable) > 0:
        C_values = np.array([r["C"] for r in usable], dtype=float)
        theta_values = np.array([r["abs_theta_deg"] for r in usable], dtype=float)
        energy_errors = np.array([abs(r["energy_error"]) for r in usable], dtype=float)

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

    if len(usable) < 3:
        regime = "no_signal"
    elif R2_inv_b > 0.85 and abs(p - 1.0) < 0.25:
        regime = "inverse_b_pass"
    else:
        regime = "inverse_b_fail"

    return {
        "mu": params["mu"],
        "v0": params["v0"],
        "softening": params["softening"],
        "usable": len(usable),
        "q_inverse_b": q,
        "R2_inverse_b": R2_inv_b,
        "A_power": A,
        "p_power": p,
        "R2_power": R2_power,
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
        f"mu={summary['mu']:4.2f}, "
        f"v0={summary['v0']:4.2f} | "
        f"regime={summary['regime']:15s} | "
        f"usable={summary['usable']:2d} | "
        f"R2_1/b={summary['R2_inverse_b']:.6f} | "
        f"p={summary['p_power']:.6f} | "
        f"C_mean={summary['C_mean']:.6f} | "
        f"C_cv={summary['C_cv']:.6f} | "
        f"max_theta={summary['max_theta_deg']:.4f} deg"
    )


# ============================================================
# 8. Saving
# ============================================================

def save_path_csv(all_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "mu",
            "v0",
            "softening",
            "b",
            "min_r",
            "D",
            "theta_rad",
            "abs_theta_rad",
            "theta_deg",
            "abs_theta_deg",
            "collapse_x_mu_over_bv02",
            "C_theta_b_v02_over_mu",
            "entered_core",
            "reached_right",
            "escaped_bounds",
            "energy_error"
        ])

        for rows in all_rows:
            for r in rows:
                writer.writerow([
                    r["mu"],
                    r["v0"],
                    r["softening"],
                    r["b"],
                    r["min_r"],
                    r["D"],
                    r["theta_rad"],
                    r["abs_theta_rad"],
                    r["theta_deg"],
                    r["abs_theta_deg"],
                    r["collapse_x"],
                    r["C"],
                    r["entered_core"],
                    r["reached_right"],
                    r["escaped_bounds"],
                    r["energy_error"]
                ])

    print(f"Saved path CSV: {filename}")


def save_summary_csv(summaries, global_summary, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "mu",
            "v0",
            "softening",
            "usable",
            "q_inverse_b",
            "R2_inverse_b",
            "A_power",
            "p_power",
            "R2_power",
            "C_mean",
            "C_std",
            "C_cv",
            "max_theta_deg",
            "mean_theta_deg",
            "max_energy_error",
            "regime",
            "global_Q",
            "global_R2",
            "global_C_mean",
            "global_C_std",
            "global_C_cv"
        ])

        for s in summaries:
            writer.writerow([
                s["mu"],
                s["v0"],
                s["softening"],
                s["usable"],
                s["q_inverse_b"],
                s["R2_inverse_b"],
                s["A_power"],
                s["p_power"],
                s["R2_power"],
                s["C_mean"],
                s["C_std"],
                s["C_cv"],
                s["max_theta_deg"],
                s["mean_theta_deg"],
                s["max_energy_error"],
                s["regime"],
                global_summary["Q"],
                global_summary["R2"],
                global_summary["C_mean"],
                global_summary["C_std"],
                global_summary["C_cv"]
            ])

    print(f"Saved summary CSV: {filename}")


# ============================================================
# 9. Plotting
# ============================================================

def case_label(params):
    return f"mu={params['mu']}, v0={params['v0']}"


def plot_selected_angle_vs_b(all_rows, parameter_cases, selected_indices):
    plt.figure(figsize=(11, 6))

    for idx in selected_indices:
        rows = all_rows[idx]
        params = parameter_cases[idx]

        b = [r["b"] for r in rows]
        theta = [r["abs_theta_deg"] for r in rows]

        plt.plot(b, theta, marker="o", label=case_label(params))

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 3b: selected deflection curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3b_selected_deflection_vs_b.png", dpi=200)
    print("Saved plot: toy_model_3b_selected_deflection_vs_b.png")
    plt.show()


def plot_collapse(usable, Q):
    x = np.array([r["collapse_x"] for r in usable], dtype=float)
    theta = np.array([r["abs_theta_rad"] for r in usable], dtype=float)

    plt.figure(figsize=(8, 6))

    plt.scatter(x, theta, s=35)

    x_line = np.linspace(0.0, max(x)*1.05, 200)
    theta_line = Q*x_line

    plt.plot(x_line, theta_line, linewidth=2, label=f"fit: theta = {Q:.4f} * mu/(b*v0^2)")

    plt.xlabel("mu/(b*v0^2)")
    plt.ylabel("|deflection angle| in radians")
    plt.title("Toy Model 3b: scaling collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3b_scaling_collapse.png", dpi=200)
    print("Saved plot: toy_model_3b_scaling_collapse.png")
    plt.show()


def plot_C_vs_b(all_rows, parameter_cases, selected_indices):
    plt.figure(figsize=(11, 6))

    for idx in selected_indices:
        rows = all_rows[idx]
        params = parameter_cases[idx]

        usable = [r for r in rows if is_usable(r, params["core_radius"])]

        b = [r["b"] for r in usable]
        C = [r["C"] for r in usable]

        plt.plot(b, C, marker="o", label=case_label(params))

    plt.xlabel("impact parameter b")
    plt.ylabel("C = theta*b*v0^2/mu")
    plt.title("Toy Model 3b: collapse constant C")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3b_C_vs_b.png", dpi=200)
    print("Saved plot: toy_model_3b_C_vs_b.png")
    plt.show()


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    n = 121
    core_radius = 6.0
    softening = 6.0

    base_params = {
        "n": n,
        "core_radius": core_radius,
        "softening": softening,
        "x_start": 0.0,
        "x_end": float(n - 1),
        "x_min_escape": -0.5*n,
        "x_max_escape": 1.5*n,
        "y_min_escape": -0.5*n,
        "y_max_escape": 1.5*n,
        "dt": 0.04,
        "max_steps": 30000
    }

    # Impact parameters.
    b_values = list(range(10, 51, 4))

    # Parameter scan.
    mu_values = [0.4, 0.8, 1.2, 1.6]
    v0_values = [1.5, 2.0, 2.5, 3.0]

    parameter_cases = []

    for mu in mu_values:
        for v0 in v0_values:
            params = dict(base_params)
            params["mu"] = mu
            params["v0"] = v0
            parameter_cases.append(params)

    all_rows = []
    summaries = []

    print("\n==============================================================")
    print("TOY MODEL 3b: DYNAMIC SCALING LAW TEST")
    print("==============================================================")
    print(f"Grid/domain size: {n}")
    print(f"Core radius: {core_radius}")
    print(f"Softening: {softening}")
    print(f"mu values: {mu_values}")
    print(f"v0 values: {v0_values}")
    print(f"b values: {b_values}")
    print("==============================================================")

    for i, params in enumerate(parameter_cases):
        print("\n--------------------------------------------------------------")
        print(f"Running case {i + 1}/{len(parameter_cases)}: {case_label(params)}")
        print("--------------------------------------------------------------")

        rows = run_case(params, b_values)
        summary = summarize_case(rows, params)

        print_case_summary(summary)

        all_rows.append(rows)
        summaries.append(summary)

    usable = all_usable_rows(all_rows, core_radius)

    Q, global_R2 = fit_global_scaling(usable)

    C_values = np.array([r["C"] for r in usable], dtype=float)

    global_summary = {
        "Q": Q,
        "R2": global_R2,
        "C_mean": float(np.mean(C_values)),
        "C_std": float(np.std(C_values)),
        "C_cv": float(np.std(C_values)/abs(np.mean(C_values)))
    }

    save_path_csv(
        all_rows=all_rows,
        filename="toy_model_3b_scaling_path_data.csv"
    )

    save_summary_csv(
        summaries=summaries,
        global_summary=global_summary,
        filename="toy_model_3b_scaling_summary.csv"
    )

    # Plot a few representative cases:
    # index order is mu outer loop, v0 inner loop.
    selected_indices = [
        0,   # mu=0.4, v0=1.5
        1,   # mu=0.4, v0=2.0
        5,   # mu=0.8, v0=2.0
        9,   # mu=1.2, v0=2.0
        13,  # mu=1.6, v0=2.0
        15   # mu=1.6, v0=3.0
    ]

    plot_selected_angle_vs_b(
        all_rows=all_rows,
        parameter_cases=parameter_cases,
        selected_indices=selected_indices
    )

    plot_collapse(
        usable=usable,
        Q=Q
    )

    plot_C_vs_b(
        all_rows=all_rows,
        parameter_cases=parameter_cases,
        selected_indices=selected_indices
    )

    print("\n==============================================================")
    print("FINAL TOY MODEL 3b SUMMARY")
    print("==============================================================")

    print("")
    print("Formal:")
    print("    |delta_theta| proportional to mu/(b*v0^2)")
    print("")
    print("Microsoft/plain:")
    print("    abs(deflection_angle) ~ mu/(b*v0^2)")
    print("")
    print("Normal:")
    print("    We test whether changing mu, b, and v0 collapses onto one law.")
    print("")

    print(f"Total usable trajectories: {len(usable)}")
    print("Global fit:")
    print("    theta = Q * mu/(b*v0^2)")
    print(f"    Q = {global_summary['Q']:.8f}")
    print(f"    R2_global = {global_summary['R2']:.8f}")
    print("")
    print("Collapse constant:")
    print("    C = theta*b*v0^2/mu")
    print(f"    C_mean = {global_summary['C_mean']:.8f}")
    print(f"    C_std  = {global_summary['C_std']:.8f}")
    print(f"    C_cv   = {global_summary['C_cv']:.8f}")
    print("")

    print("Per-case summary:")
    print("--------------------------------------------------------------")
    for s in summaries:
        print_case_summary(s)
    print("--------------------------------------------------------------")

    if global_summary["R2"] > 0.9 and global_summary["C_cv"] < 0.25:
        print("RESULT: PASS")
        print("The dynamic inverse-radius model approximately obeys")
        print("abs(deflection_angle) ~ mu/(b*v0^2).")
    else:
        print("RESULT: PARTIAL / CHECK")
        print("The model shows bending, but the global scaling is not clean enough yet.")

    print("==============================================================")
