import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# TOY MODEL 3a:
# TIME-DYNAMIC HALO SCATTERING
# ============================================================
#
# We are no longer using shortest spatial paths.
#
# Formal:
#   a = -grad(U)
#
# Microsoft/plain:
#   acceleration = -gradient(potential)
#
# Normal:
#   The particle moves through time under an effective halo potential.
#
# Gravity-like target:
#   abs(delta_theta(b)) approx q/b
#
# Microsoft/plain:
#   abs(deflection_angle) approx q/b
#
# Normal:
#   Farther particles should bend less, approximately like 1/b.

EPSILON = 1e-12


# ============================================================
# 1. Basic geometry
# ============================================================

def center_of_grid(n):
    return n / 2.0, n / 2.0


def radius_from_center(x, y, n):
    cx, cy = center_of_grid(n)
    dx = x - cx
    dy = y - cy
    return np.sqrt(dx * dx + dy * dy)


def unit_radial_vector(x, y, n):
    cx, cy = center_of_grid(n)

    dx = x - cx
    dy = y - cy

    r = np.sqrt(dx * dx + dy * dy)

    if r < EPSILON:
        return 0.0, 0.0, r

    return dx / r, dy / r, r


# ============================================================
# 2. Effective potentials
# ============================================================

def potential_value(x, y, n, radius, mu, profile, halo_length, softening):
    """
    Formal:
        inverse_radius:
            U(r) = -mu / sqrt(r^2 + s^2)

        exponential:
            U(r) = -mu exp(-rho/lambda)

        gaussian:
            U(r) = -mu exp(-rho^2/(2 lambda^2))

        rho = max(r - R, 0)

    Microsoft/plain:
        inverse_radius:
            U = -mu / sqrt(r^2 + softening^2)

        exponential:
            U = -mu*exp(-rho/halo_length)

        gaussian:
            U = -mu*exp(-rho^2/(2*halo_length^2))

    Normal:
        U is the effective attraction potential.
        More negative U means stronger attraction.
    """

    r = radius_from_center(x, y, n)

    if profile == "none":
        return 0.0

    if profile == "inverse_radius":
        return -mu / np.sqrt(r * r + softening * softening)

    rho = max(r - radius, 0.0)

    if profile == "exponential":
        return -mu * np.exp(-rho / halo_length)

    if profile == "gaussian":
        return -mu * np.exp(-(rho * rho) / (2.0 * halo_length * halo_length))

    raise ValueError("Unknown profile: " + str(profile))


def acceleration(x, y, n, radius, mu, profile, halo_length, softening):
    """
    Formal:
        a = -grad(U)

    Microsoft/plain:
        acceleration = -gradient(potential)

    Normal:
        This gives the force direction and strength.
        The force points toward the defect center.
    """

    ux, uy, r = unit_radial_vector(x, y, n)

    if profile == "none":
        return 0.0, 0.0

    if r < EPSILON:
        return 0.0, 0.0

    if profile == "inverse_radius":
        # Formal:
        #   U = -mu / sqrt(r^2 + s^2)
        #   a_vec = -mu r_vec / (r^2 + s^2)^(3/2)
        #
        # Microsoft/plain:
        #   ax = -mu*dx/(r^2 + softening^2)^(3/2)
        #   ay = -mu*dy/(r^2 + softening^2)^(3/2)
        #
        # Normal:
        #   This is the Newtonian-like calibration case.

        cx, cy = center_of_grid(n)

        dx = x - cx
        dy = y - cy

        denom = (r * r + softening * softening) ** 1.5

        ax = -mu * dx / denom
        ay = -mu * dy / denom

        return ax, ay

    rho = max(r - radius, 0.0)

    if profile == "exponential":
        # Formal:
        #   U = -mu exp(-rho/lambda)
        #   a_mag = mu/lambda * exp(-rho/lambda)
        #
        # Microsoft/plain:
        #   acceleration_strength = mu/halo_length * exp(-rho/halo_length)

        if r <= radius:
            return 0.0, 0.0

        a_mag = (mu / halo_length) * np.exp(-rho / halo_length)

        return -a_mag * ux, -a_mag * uy

    if profile == "gaussian":
        # Formal:
        #   U = -mu exp(-rho^2/(2 lambda^2))
        #   a_mag = mu*rho/lambda^2 * exp(-rho^2/(2 lambda^2))
        #
        # Microsoft/plain:
        #   acceleration_strength =
        #       mu*rho/halo_length^2 * exp(-rho^2/(2*halo_length^2))

        if r <= radius:
            return 0.0, 0.0

        a_mag = (
            mu * rho / (halo_length * halo_length)
            * np.exp(-(rho * rho) / (2.0 * halo_length * halo_length))
        )

        return -a_mag * ux, -a_mag * uy

    raise ValueError("Unknown profile: " + str(profile))


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
        This is the time evolution rule.
    """

    x, y, vx, vy = state

    ax, ay = acceleration(
        x=x,
        y=y,
        n=params["n"],
        radius=params["radius"],
        mu=params["mu"],
        profile=params["profile"],
        halo_length=params["halo_length"],
        softening=params["softening"]
    )

    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state, dt, params):
    """
    Formal:
        Runge-Kutta 4th order integration

    Microsoft/plain:
        RK4 time step

    Normal:
        More accurate than simple Euler stepping.
    """

    k1 = derivatives(state, params)
    k2 = derivatives(state + 0.5 * dt * k1, params)
    k3 = derivatives(state + 0.5 * dt * k2, params)
    k4 = derivatives(state + dt * k3, params)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def energy(state, params):
    x, y, vx, vy = state

    kinetic = 0.5 * (vx * vx + vy * vy)

    potential = potential_value(
        x=x,
        y=y,
        n=params["n"],
        radius=params["radius"],
        mu=params["mu"],
        profile=params["profile"],
        halo_length=params["halo_length"],
        softening=params["softening"]
    )

    return kinetic + potential


# ============================================================
# 4. Simulate one trajectory
# ============================================================

def simulate_trajectory(params, b):
    """
    Formal:
        initial state:
            x = x_start
            y = y_center + b
            vx = v0
            vy = 0

    Microsoft/plain:
        start on the left
        move to the right
        impact parameter = b

    Normal:
        A particle passes by the defect with initial horizontal velocity.
    """

    n = params["n"]
    cx, cy = center_of_grid(n)

    x_start = params["x_start"]
    x_end = params["x_end"]

    y0 = cy + b

    state = np.array([
        x_start,
        y0,
        params["v0"],
        0.0
    ], dtype=float)

    dt = params["dt"]
    max_steps = params["max_steps"]

    xs = []
    ys = []
    vxs = []
    vys = []
    times = []
    radii = []
    energies = []

    entered_core = False
    reached_right = False
    escaped_bounds = False

    initial_energy = energy(state, params)

    for step in range(max_steps):
        t = step * dt

        x, y, vx, vy = state

        r = radius_from_center(x, y, n)

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        times.append(t)
        radii.append(r)
        energies.append(energy(state, params))

        if r <= params["radius"]:
            entered_core = True

        if x >= x_end and vx > 0:
            reached_right = True
            break

        if x < -0.5 * n or x > 1.5 * n or y < -0.5 * n or y > 1.5 * n:
            escaped_bounds = True
            break

        state = rk4_step(state, dt, params)

    final_state = state

    x_final, y_final, vx_final, vy_final = final_state

    theta_in = 0.0

    theta_out = np.arctan2(vy_final, vx_final)

    delta_theta = theta_out - theta_in

    abs_delta_theta = abs(delta_theta)

    min_r = float(np.min(radii))

    final_energy = energy(final_state, params)

    energy_error = final_energy - initial_energy

    result = {
        "b": b,
        "xs": xs,
        "ys": ys,
        "vxs": vxs,
        "vys": vys,
        "times": times,
        "radii": radii,
        "energies": energies,
        "min_r": min_r,
        "D": b - min_r,
        "entered_core": entered_core,
        "reached_right": reached_right,
        "escaped_bounds": escaped_bounds,
        "theta_out_rad": theta_out,
        "delta_theta_rad": delta_theta,
        "abs_delta_theta_rad": abs_delta_theta,
        "theta_out_deg": np.degrees(theta_out),
        "abs_delta_theta_deg": abs(np.degrees(delta_theta)),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_error": energy_error,
        "final_x": x_final,
        "final_y": y_final,
        "final_vx": vx_final,
        "final_vy": vy_final
    }

    return result


# ============================================================
# 5. Run one case
# ============================================================

def run_case(params, b_values):
    rows = []

    for b in b_values:
        result = simulate_trajectory(params, b)

        result["profile"] = params["profile"]
        result["mu"] = params["mu"]
        result["halo_length"] = params["halo_length"]
        result["softening"] = params["softening"]
        result["v0"] = params["v0"]
        result["dt"] = params["dt"]

        rows.append(result)

    return rows


# ============================================================
# 6. Fits
# ============================================================

def usable_scattering_rows(rows, radius):
    usable = []

    for r in rows:
        if not r["reached_right"]:
            continue

        if r["entered_core"]:
            continue

        if r["b"] <= radius:
            continue

        if r["abs_delta_theta_rad"] <= 1e-8:
            continue

        usable.append(r)

    return usable


def fit_inverse_b(rows, radius):
    """
    Formal:
        theta(b) approx q/b

    Microsoft/plain:
        deflection_angle approx q/b

    Normal:
        This is the gravity-like test.
    """

    usable = usable_scattering_rows(rows, radius)

    if len(usable) < 3:
        return np.nan, np.nan, len(usable)

    b = np.array([r["b"] for r in usable], dtype=float)
    theta = np.array([r["abs_delta_theta_rad"] for r in usable], dtype=float)

    x = 1.0 / b

    q = np.sum(x * theta) / np.sum(x * x)

    theta_pred = q * x

    ss_res = np.sum((theta - theta_pred) ** 2)
    ss_tot = np.sum((theta - np.mean(theta)) ** 2)

    if ss_tot <= EPSILON:
        R2 = np.nan
    else:
        R2 = 1.0 - ss_res / ss_tot

    return q, R2, len(usable)


def fit_power_law(rows, radius):
    """
    Formal:
        theta(b) approx A*b^(-p)

    Microsoft/plain:
        deflection_angle approx A / b^p

    Normal:
        Gravity-like means p should be close to 1.
    """

    usable = usable_scattering_rows(rows, radius)

    if len(usable) < 3:
        return np.nan, np.nan, np.nan, len(usable)

    b = np.array([r["b"] for r in usable], dtype=float)
    theta = np.array([r["abs_delta_theta_rad"] for r in usable], dtype=float)

    log_b = np.log(b)
    log_theta = np.log(theta)

    X = np.vstack([np.ones_like(log_b), -log_b]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, log_theta, rcond=None)

    log_A = coeffs[0]
    p = coeffs[1]

    log_pred = log_A - p * log_b

    ss_res = np.sum((log_theta - log_pred) ** 2)
    ss_tot = np.sum((log_theta - np.mean(log_theta)) ** 2)

    if ss_tot <= EPSILON:
        R2 = np.nan
    else:
        R2 = 1.0 - ss_res / ss_tot

    A = np.exp(log_A)

    return A, p, R2, len(usable)


def summarize_case(rows, params):
    q, R2_inv, n_fit = fit_inverse_b(rows, params["radius"])

    A, p, R2_power, n_power = fit_power_law(rows, params["radius"])

    reached = [r for r in rows if r["reached_right"]]
    entered = [r for r in rows if r["entered_core"]]
    usable = usable_scattering_rows(rows, params["radius"])

    if len(usable) > 0:
        angles = np.array([r["abs_delta_theta_deg"] for r in usable])
        max_angle = float(np.max(angles))
        mean_angle = float(np.mean(angles))
        max_energy_error = float(np.max(np.abs([r["energy_error"] for r in usable])))
    else:
        max_angle = 0.0
        mean_angle = 0.0
        max_energy_error = np.nan

    if n_fit < 3:
        regime = "no_scattering_signal"

    elif not np.isnan(R2_inv) and R2_inv > 0.85 and not np.isnan(p) and abs(p - 1.0) < 0.25:
        regime = "inverse_like_dynamic"

    elif not np.isnan(R2_power) and R2_power > 0.85:
        regime = "power_law_not_1"

    else:
        regime = "mixed_dynamic"

    return {
        "profile": params["profile"],
        "mu": params["mu"],
        "halo_length": params["halo_length"],
        "softening": params["softening"],
        "v0": params["v0"],
        "dt": params["dt"],
        "regime": regime,
        "total": len(rows),
        "reached": len(reached),
        "entered_core": len(entered),
        "usable": len(usable),
        "q_inverse": q,
        "R2_inverse": R2_inv,
        "A_power": A,
        "p_power": p,
        "R2_power": R2_power,
        "max_angle_deg": max_angle,
        "mean_angle_deg": mean_angle,
        "max_energy_error": max_energy_error
    }


# ============================================================
# 7. Printing
# ============================================================

def print_case_table(rows, summary):
    print("\n==============================================================")
    print("CASE SUMMARY")
    print("==============================================================")
    print(f"profile         = {summary['profile']}")
    print(f"mu              = {summary['mu']}")
    print(f"halo_length     = {summary['halo_length']}")
    print(f"softening       = {summary['softening']}")
    print(f"v0              = {summary['v0']}")
    print(f"dt              = {summary['dt']}")
    print(f"regime          = {summary['regime']}")
    print("--------------------------------------------------------------")
    print(f"total trajectories       = {summary['total']}")
    print(f"reached right side       = {summary['reached']}")
    print(f"entered core             = {summary['entered_core']}")
    print(f"usable for fit           = {summary['usable']}")
    print(f"max angle                = {summary['max_angle_deg']:.6f} deg")
    print(f"mean angle               = {summary['mean_angle_deg']:.6f} deg")
    print(f"R2 for theta approx q/b  = {summary['R2_inverse']}")
    print(f"power-law p              = {summary['p_power']}")
    print(f"R2 power law             = {summary['R2_power']}")
    print(f"max energy error         = {summary['max_energy_error']}")
    print("==============================================================")

    print("")
    print(
        f"{'b':>4} | "
        f"{'min_r':>10} | "
        f"{'D=b-min_r':>10} | "
        f"{'theta_deg':>12} | "
        f"{'b*theta(rad)':>13} | "
        f"{'core':>6} | "
        f"{'reached':>8} | "
        f"{'energy_err':>12}"
    )

    print("-" * 100)

    for r in rows:
        btheta = r["b"] * r["abs_delta_theta_rad"]

        print(
            f"{r['b']:4.1f} | "
            f"{r['min_r']:10.4f} | "
            f"{r['D']:10.4f} | "
            f"{r['abs_delta_theta_deg']:12.6f} | "
            f"{btheta:13.6f} | "
            f"{str(r['entered_core']):>6} | "
            f"{str(r['reached_right']):>8} | "
            f"{r['energy_error']:12.3e}"
        )

    print("-" * 100)


# ============================================================
# 8. Saving
# ============================================================

def save_summary_csv(summaries, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "profile",
            "mu",
            "halo_length",
            "softening",
            "v0",
            "dt",
            "regime",
            "total",
            "reached",
            "entered_core",
            "usable",
            "q_inverse",
            "R2_inverse",
            "A_power",
            "p_power",
            "R2_power",
            "max_angle_deg",
            "mean_angle_deg",
            "max_energy_error"
        ])

        for s in summaries:
            writer.writerow([
                s["profile"],
                s["mu"],
                s["halo_length"],
                s["softening"],
                s["v0"],
                s["dt"],
                s["regime"],
                s["total"],
                s["reached"],
                s["entered_core"],
                s["usable"],
                s["q_inverse"],
                s["R2_inverse"],
                s["A_power"],
                s["p_power"],
                s["R2_power"],
                s["max_angle_deg"],
                s["mean_angle_deg"],
                s["max_energy_error"]
            ])

    print(f"Saved summary CSV: {filename}")


def save_path_csv(all_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "profile",
            "mu",
            "halo_length",
            "softening",
            "v0",
            "dt",
            "b",
            "min_r",
            "D",
            "theta_out_deg",
            "abs_delta_theta_deg",
            "b_times_theta_rad",
            "entered_core",
            "reached_right",
            "escaped_bounds",
            "initial_energy",
            "final_energy",
            "energy_error",
            "final_x",
            "final_y",
            "final_vx",
            "final_vy"
        ])

        for rows in all_rows:
            for r in rows:
                writer.writerow([
                    r["profile"],
                    r["mu"],
                    r["halo_length"],
                    r["softening"],
                    r["v0"],
                    r["dt"],
                    r["b"],
                    r["min_r"],
                    r["D"],
                    r["theta_out_deg"],
                    r["abs_delta_theta_deg"],
                    r["b"] * r["abs_delta_theta_rad"],
                    r["entered_core"],
                    r["reached_right"],
                    r["escaped_bounds"],
                    r["initial_energy"],
                    r["final_energy"],
                    r["energy_error"],
                    r["final_x"],
                    r["final_y"],
                    r["final_vx"],
                    r["final_vy"]
                ])

    print(f"Saved path CSV: {filename}")


# ============================================================
# 9. Plotting
# ============================================================

def case_label(params):
    return (
        f"{params['profile']}, "
        f"mu={params['mu']}, "
        f"lambda={params['halo_length']}, "
        f"v0={params['v0']}"
    )


def plot_deflection(all_rows, parameter_cases):
    plt.figure(figsize=(11, 6))

    for rows, params in zip(all_rows, parameter_cases):
        b = [r["b"] for r in rows]
        theta = [r["abs_delta_theta_deg"] for r in rows]

        plt.plot(b, theta, marker="o", label=case_label(params))

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 3a: dynamic deflection angle vs impact parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3a_deflection_vs_b.png", dpi=200)
    print("Saved plot: toy_model_3a_deflection_vs_b.png")
    plt.show()


def plot_b_times_theta(all_rows, parameter_cases):
    plt.figure(figsize=(11, 6))

    for rows, params in zip(all_rows, parameter_cases):
        b = np.array([r["b"] for r in rows], dtype=float)
        theta = np.array([r["abs_delta_theta_rad"] for r in rows], dtype=float)

        plt.plot(b, b * theta, marker="o", label=case_label(params))

    plt.xlabel("impact parameter b")
    plt.ylabel("b * |deflection angle|")
    plt.title("Toy Model 3a: check for 1/b behaviour")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3a_b_times_theta.png", dpi=200)
    print("Saved plot: toy_model_3a_b_times_theta.png")
    plt.show()


def plot_trajectories(rows, params, filename):
    n = params["n"]
    cx, cy = center_of_grid(n)

    radius = params["radius"]

    plt.figure(figsize=(7, 7))

    theta = np.linspace(0, 2 * np.pi, 400)

    x_circle = cx + radius * np.cos(theta)
    y_circle = cy + radius * np.sin(theta)

    plt.plot(x_circle, y_circle, linewidth=2, label="defect/core radius")

    b_values = [r["b"] for r in rows]

    selected_b = [
        min(b_values),
        np.percentile(b_values, 20),
        np.percentile(b_values, 40),
        np.percentile(b_values, 60),
        np.percentile(b_values, 80),
        max(b_values)
    ]

    selected_clean = []

    for b0 in selected_b:
        closest = min(b_values, key=lambda z: abs(z - b0))

        if closest not in selected_clean:
            selected_clean.append(closest)

    for b in selected_clean:
        r = [z for z in rows if z["b"] == b][0]

        plt.plot(r["xs"], r["ys"], label=f"b={b}")

    plt.xlim(params["x_start"], params["x_end"])
    plt.ylim(0, n)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 3a trajectories: " + case_label(params))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved trajectory plot: {filename}")
    plt.show()


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    n = 101
    radius = 6.0

    base_params = {
        "n": n,
        "radius": radius,
        "x_start": 0.0,
        "x_end": float(n - 1),
        "v0": 2.0,
        "dt": 0.04,
        "max_steps": 20000
    }

    # Use b values outside the core radius.
    b_values = list(range(8, 43, 2))

    # Important:
    # inverse_radius is the calibration case.
    # It should be the most likely to give 1/b-like deflection.
    #
    # exponential and gaussian are closer to our previous halo profiles,
    # and we test whether they can also produce gravity-like scattering.

    parameter_cases = [
        {
            **base_params,
            "profile": "none",
            "mu": 0.0,
            "halo_length": 20.0,
            "softening": 6.0
        },
        {
            **base_params,
            "profile": "inverse_radius",
            "mu": 0.8,
            "halo_length": 20.0,
            "softening": 6.0
        },
        {
            **base_params,
            "profile": "inverse_radius",
            "mu": 1.5,
            "halo_length": 20.0,
            "softening": 6.0
        },
        {
            **base_params,
            "profile": "exponential",
            "mu": 1.5,
            "halo_length": 24.0,
            "softening": 6.0
        },
        {
            **base_params,
            "profile": "gaussian",
            "mu": 1.5,
            "halo_length": 18.0,
            "softening": 6.0
        }
    ]

    all_rows = []
    summaries = []

    print("\n==============================================================")
    print("TOY MODEL 3a: TIME-DYNAMIC HALO SCATTERING")
    print("==============================================================")
    print(f"Grid/domain size: {n}")
    print(f"Core radius: {radius}")
    print(f"b values: {b_values}")
    print("==============================================================")

    for i, params in enumerate(parameter_cases):
        print("\n--------------------------------------------------------------")
        print(f"Running case {i + 1}/{len(parameter_cases)}")
        print(case_label(params))
        print("--------------------------------------------------------------")

        rows = run_case(params, b_values)
        summary = summarize_case(rows, params)

        print_case_table(rows, summary)

        all_rows.append(rows)
        summaries.append(summary)

    save_summary_csv(
        summaries=summaries,
        filename="toy_model_3a_dynamic_summary.csv"
    )

    save_path_csv(
        all_rows=all_rows,
        filename="toy_model_3a_dynamic_path_data.csv"
    )

    plot_deflection(all_rows, parameter_cases)

    plot_b_times_theta(all_rows, parameter_cases)

    # Plot trajectories for the calibration inverse case and one halo case.
    plot_trajectories(
        rows=all_rows[2],
        params=parameter_cases[2],
        filename="toy_model_3a_inverse_radius_trajectories.png"
    )

    plot_trajectories(
        rows=all_rows[3],
        params=parameter_cases[3],
        filename="toy_model_3a_exponential_trajectories.png"
    )

    print("\n==============================================================")
    print("FINAL TOY MODEL 3a SUMMARY")
    print("==============================================================")

    for s in summaries:
        print(
            f"{s['profile']:16s} | "
            f"mu={s['mu']:5.2f} | "
            f"regime={s['regime']:24s} | "
            f"usable={s['usable']:3d} | "
            f"R2_1/b={s['R2_inverse']} | "
            f"p={s['p_power']} | "
            f"R2_power={s['R2_power']} | "
            f"max_angle={s['max_angle_deg']:.4f} deg"
        )

    print("==============================================================")

