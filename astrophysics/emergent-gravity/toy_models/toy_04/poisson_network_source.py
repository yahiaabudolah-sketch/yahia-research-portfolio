import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# TOY MODEL 4
# Poisson field equation from entanglement-network source
# ============================================================
#
# Formal:
#     laplacian(Phi) = 4*pi*rho_network
#     F = -grad(Phi)
#     rho_network is normalized so total source mass is mu_eff
#     mu_eff = gamma*abs(delta_K_excess)
#
# Microsoft/plain:
#     laplacian(phi) = 4*pi*network_density
#     force = -gradient(phi)
#     mu_eff = gamma*abs(delta_K_excess)
#
# Normal:
#     Toy Model 3d generated the field from spherical spreading.
#     Toy Model 4 solves the field equation directly on a grid.
#
# Dependencies:
#     numpy, matplotlib, csv

EPS = 1e-12


# ------------------------------------------------------------
# 1. Closed-form Toy Model 1 curvature source
# ------------------------------------------------------------

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
        Delta K_excess(alpha,beta,r)

    Microsoft/plain:
        delta_K_excess = network curvature response

    Normal:
        This is our exact Toy Model 1 source response.
    """
    radius = int(radius)
    if radius not in COEFFS:
        raise ValueError("radius must be one of 4, 5, 6, 7, 8")

    c_beta, c_inv_beta, c_ab, c_ba = COEFFS[radius]

    value = (
        c_beta * (np.sqrt(beta) - 1.0)
        + c_inv_beta * (1.0 / np.sqrt(beta) - 1.0)
        + c_ab * (np.sqrt(alpha / beta) - 1.0)
        + c_ba * (np.sqrt(beta / alpha) - 1.0)
    )
    return value


def network_source_strength(alpha, beta, radius, gamma=1.0):
    """
    Formal:
        mu_eff = gamma*|Delta K_excess|

    Microsoft/plain:
        mu_eff = gamma*abs(delta_K_excess)

    Normal:
        Network curvature becomes source strength.
    """
    return gamma * abs(delta_K_excess(alpha, beta, radius))


# ------------------------------------------------------------
# 2. Build 3D source density
# ------------------------------------------------------------

def build_grid(N=51, L=90.0):
    """
    Microsoft/plain:
        make a cubic grid from -L to +L
    """
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X*X + Y*Y + Z*Z)
    return x, dx, X, Y, Z, R


def gaussian_network_density(R, core_radius, mu_eff, dx):
    """
    Formal:
        rho(r) proportional to exp(-r^2/(2*sigma^2))
        integral rho dV = mu_eff

    Microsoft/plain:
        density = smooth central blob
        total density = mu_eff

    Normal:
        We use a smooth source so the Poisson solver behaves nicely.
    """
    sigma = core_radius / 2.0
    rho = np.exp(-0.5 * (R / sigma) ** 2)

    total = np.sum(rho) * dx**3
    rho = rho * (mu_eff / (total + EPS))

    return rho


# ------------------------------------------------------------
# 3. Poisson solver
# ------------------------------------------------------------

def analytic_boundary_potential(R, mu_eff, softening):
    """
    Formal:
        Phi_boundary = -mu_eff/sqrt(r^2 + softening^2)

    Microsoft/plain:
        boundary phi = -mu_eff/r softened

    Normal:
        This anchors the box to the correct far-field behaviour.
    """
    return -mu_eff / np.sqrt(R*R + softening*softening)


def solve_poisson_3d(rho, R, dx, mu_eff, core_radius, iterations=2500, omega=1.0):
    """
    Formal:
        laplacian(Phi) = 4*pi*rho

        finite difference update:
        Phi = (sum_neighbours - dx^2*4*pi*rho)/6

    Microsoft/plain:
        solve laplacian(phi)=4*pi*density

    Normal:
        This is a direct grid field-equation solve.
    """
    S = 4.0 * np.pi * rho
    dx2 = dx * dx

    softening = core_radius / 3.0
    phi = analytic_boundary_potential(R, mu_eff, softening)

    boundary = phi.copy()

    print("\nSolving Poisson equation...")
    print("iterations =", iterations)

    for it in range(1, iterations + 1):
        old_center = phi[1:-1, 1:-1, 1:-1].copy()

        jacobi = (
            phi[2:, 1:-1, 1:-1]
            + phi[:-2, 1:-1, 1:-1]
            + phi[1:-1, 2:, 1:-1]
            + phi[1:-1, :-2, 1:-1]
            + phi[1:-1, 1:-1, 2:]
            + phi[1:-1, 1:-1, :-2]
            - dx2 * S[1:-1, 1:-1, 1:-1]
        ) / 6.0

        phi[1:-1, 1:-1, 1:-1] = (1.0 - omega) * old_center + omega * jacobi

        phi[0, :, :] = boundary[0, :, :]
        phi[-1, :, :] = boundary[-1, :, :]
        phi[:, 0, :] = boundary[:, 0, :]
        phi[:, -1, :] = boundary[:, -1, :]
        phi[:, :, 0] = boundary[:, :, 0]
        phi[:, :, -1] = boundary[:, :, -1]

        if it % 500 == 0 or it == 1 or it == iterations:
            max_change = np.max(np.abs(phi[1:-1, 1:-1, 1:-1] - old_center))
            print(f"iteration {it:5d} | max update = {max_change:.6e}")

    return phi


def compute_force(phi, dx):
    """
    Formal:
        F = -grad(Phi)

    Microsoft/plain:
        force = -gradient(phi)
    """
    dphi_dx, dphi_dy, dphi_dz = np.gradient(phi, dx, dx, dx, edge_order=2)

    Fx = -dphi_dx
    Fy = -dphi_dy
    Fz = -dphi_dz

    return Fx, Fy, Fz


def poisson_residual(phi, rho, dx):
    """
    Formal:
        residual = laplacian(Phi) - 4*pi*rho

    Microsoft/plain:
        residual = left side - right side
    """
    lap = np.zeros_like(phi)

    lap[1:-1, 1:-1, 1:-1] = (
        phi[2:, 1:-1, 1:-1]
        + phi[:-2, 1:-1, 1:-1]
        + phi[1:-1, 2:, 1:-1]
        + phi[1:-1, :-2, 1:-1]
        + phi[1:-1, 1:-1, 2:]
        + phi[1:-1, 1:-1, :-2]
        - 6.0 * phi[1:-1, 1:-1, 1:-1]
    ) / (dx * dx)

    res = lap - 4.0 * np.pi * rho
    interior = res[2:-2, 2:-2, 2:-2]

    return float(np.mean(np.abs(interior))), float(np.max(np.abs(interior)))


# ------------------------------------------------------------
# 4. Fitting tools
# ------------------------------------------------------------

def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot <= EPS:
        return np.nan

    return 1.0 - ss_res / ss_tot


def fit_potential_and_force(x, phi, Fx, Fy, Fz, core_radius, L):
    """
    Formal:
        Phi(r) = A*(1/r) + B
        |F(r)| = A_F/r^p

    Microsoft/plain:
        phi ~ 1/r
        force ~ 1/r^p

    Normal:
        For 3D Newtonian gravity, p should be close to 2.
    """
    mid = len(x) // 2

    r_axis = x[mid+1:]
    phi_axis = phi[mid+1:, mid, mid]

    F_axis = np.sqrt(
        Fx[mid+1:, mid, mid]**2
        + Fy[mid+1:, mid, mid]**2
        + Fz[mid+1:, mid, mid]**2
    )

    mask = (r_axis > 2.5 * core_radius) & (r_axis < 0.70 * L)

    r_fit = r_axis[mask]
    phi_fit = phi_axis[mask]
    F_fit = F_axis[mask]

    u = 1.0 / r_fit
    Xmat = np.vstack([u, np.ones_like(u)]).T

    coeffs, _, _, _ = np.linalg.lstsq(Xmat, phi_fit, rcond=None)

    A_phi = coeffs[0]
    B_phi = coeffs[1]

    phi_pred = A_phi * u + B_phi
    R2_phi = r2_score(phi_fit, phi_pred)

    good = F_fit > 0

    log_r = np.log(r_fit[good])
    log_F = np.log(F_fit[good])

    Xpow = np.vstack([np.ones_like(log_r), -log_r]).T

    coeffp, _, _, _ = np.linalg.lstsq(Xpow, log_F, rcond=None)

    log_A_F = coeffp[0]
    p_force = coeffp[1]

    pred_log_F = log_A_F - p_force * log_r
    R2_force = r2_score(log_F, pred_log_F)

    return {
        "A_phi": A_phi,
        "B_phi": B_phi,
        "R2_phi": R2_phi,
        "A_force": np.exp(log_A_F),
        "p_force": p_force,
        "R2_force": R2_force,
        "r_axis": r_axis,
        "phi_axis": phi_axis,
        "F_axis": F_axis,
    }


# ------------------------------------------------------------
# 5. Particle dynamics through solved field
# ------------------------------------------------------------

def analytic_acceleration(xp, yp, zp, mu_eff, softening):
    r2 = xp*xp + yp*yp + zp*zp + softening*softening
    r = np.sqrt(r2)

    factor = -mu_eff / (r2 * r)

    return factor * xp, factor * yp, factor * zp


def bilinear_interp(arr2d, x_grid, xp, yp):
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    xmin = x_grid[0]
    xmax = x_grid[-1]

    if xp <= xmin or xp >= xmax or yp <= xmin or yp >= xmax:
        return None

    u = (xp - xmin) / dx
    v = (yp - xmin) / dx

    i = int(np.floor(u))
    j = int(np.floor(v))

    if i < 0 or i >= N - 1 or j < 0 or j >= N - 1:
        return None

    du = u - i
    dv = v - j

    val = (
        (1.0 - du) * (1.0 - dv) * arr2d[i, j]
        + du * (1.0 - dv) * arr2d[i+1, j]
        + (1.0 - du) * dv * arr2d[i, j+1]
        + du * dv * arr2d[i+1, j+1]
    )

    return float(val)


def acceleration_from_grid(xp, yp, x_grid, Fx_slice, Fy_slice, mu_eff, softening):
    ax = bilinear_interp(Fx_slice, x_grid, xp, yp)
    ay = bilinear_interp(Fy_slice, x_grid, xp, yp)

    if ax is None or ay is None:
        ax, ay, _ = analytic_acceleration(xp, yp, 0.0, mu_eff, softening)
        return ax, ay

    return ax, ay


def rk4_step(state, dt, accel_func):
    def deriv(s):
        x, y, vx, vy = s
        ax, ay = accel_func(x, y)
        return np.array([vx, vy, ax, ay], dtype=float)

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_scattering(x_grid, Fx, Fy, mu_eff, core_radius, b, v0=2.0, L=90.0, dt=0.05):
    mid = len(x_grid) // 2

    Fx_slice = Fx[:, :, mid]
    Fy_slice = Fy[:, :, mid]

    softening = core_radius / 3.0

    x_start = -0.85 * L
    x_end = 0.85 * L

    def accel_func(xp, yp):
        return acceleration_from_grid(xp, yp, x_grid, Fx_slice, Fy_slice, mu_eff, softening)

    state = np.array([x_start, b, v0, 0.0], dtype=float)

    xs = []
    ys = []

    min_r = 1e99
    entered_core = False
    reached_right = False
    escaped = False

    for step in range(30000):
        xp, yp, vx, vy = state

        r = np.sqrt(xp*xp + yp*yp)
        min_r = min(min_r, r)

        xs.append(xp)
        ys.append(yp)

        if r <= core_radius:
            entered_core = True

        if xp >= x_end and vx > 0:
            reached_right = True
            break

        if abs(xp) > 1.5 * L or abs(yp) > 1.5 * L:
            escaped = True
            break

        state = rk4_step(state, dt, accel_func)

    xp, yp, vx, vy = state

    theta = np.arctan2(vy, vx)

    return {
        "b": b,
        "v0": v0,
        "theta_rad": theta,
        "abs_theta_rad": abs(theta),
        "theta_deg": np.degrees(theta),
        "abs_theta_deg": abs(np.degrees(theta)),
        "min_r": min_r,
        "entered_core": entered_core,
        "reached_right": reached_right,
        "escaped": escaped,
        "xs": xs,
        "ys": ys,
    }


def fit_scattering(rows):
    usable = [
        row for row in rows
        if row["reached_right"]
        and not row["entered_core"]
        and row["abs_theta_rad"] > 1e-10
    ]

    if len(usable) < 3:
        return np.nan, np.nan, np.nan, len(usable)

    b = np.array([row["b"] for row in usable], dtype=float)
    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    x = 1.0 / b

    q = np.sum(x * theta) / np.sum(x * x)
    pred = q * x

    R2_inv = r2_score(theta, pred)

    log_b = np.log(b)
    log_theta = np.log(theta)

    Xmat = np.vstack([np.ones_like(log_b), -log_b]).T

    coeffs, _, _, _ = np.linalg.lstsq(Xmat, log_theta, rcond=None)

    p = coeffs[1]

    return q, R2_inv, p, len(usable)


# ------------------------------------------------------------
# 6. Plots and CSV
# ------------------------------------------------------------

def make_plots(x, rho, phi, Fx, Fy, Fz, fit, rows, core_radius, L):
    mid = len(x) // 2

    plt.figure(figsize=(8, 7))
    plt.imshow(
        rho[:, :, mid].T,
        origin="lower",
        extent=[x[0], x[-1], x[0], x[-1]]
    )
    plt.colorbar(label="rho_network")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 4: network source density slice")
    plt.tight_layout()
    plt.savefig("toy_model_4_density_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.imshow(
        phi[:, :, mid].T,
        origin="lower",
        extent=[x[0], x[-1], x[0], x[-1]]
    )
    plt.colorbar(label="Phi")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 4: solved Poisson potential slice")
    plt.tight_layout()
    plt.savefig("toy_model_4_potential_slice.png", dpi=200)
    plt.show()

    r_axis = fit["r_axis"]
    phi_axis = fit["phi_axis"]
    F_axis = fit["F_axis"]

    plt.figure(figsize=(10, 6))
    plt.plot(r_axis, phi_axis, marker="o", label="solved Phi on x-axis")

    A = fit["A_phi"]
    B = fit["B_phi"]

    plt.plot(r_axis, A*(1.0/r_axis) + B, label="fit: A/r + B")
    plt.xlabel("r")
    plt.ylabel("Phi(r)")
    plt.title("Toy Model 4: potential fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4_potential_fit.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.loglog(r_axis, F_axis, marker="o", label="force magnitude")

    p = fit["p_force"]
    A_force = fit["A_force"]

    plt.loglog(r_axis, A_force/(r_axis**p), label=f"fit: A/r^{p:.3f}")
    plt.xlabel("r")
    plt.ylabel("force magnitude")
    plt.title("Toy Model 4: force power law")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4_force_power_law.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    b = [row["b"] for row in rows]
    theta = [row["abs_theta_deg"] for row in rows]

    plt.plot(b, theta, marker="o")
    plt.xlabel("impact parameter b")
    plt.ylabel("abs(deflection angle) in degrees")
    plt.title("Toy Model 4: dynamic bending from Poisson field")
    plt.tight_layout()
    plt.savefig("toy_model_4_deflection_vs_b.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    b_arr = np.array([row["b"] for row in rows], dtype=float)
    theta_arr = np.array([row["abs_theta_rad"] for row in rows], dtype=float)

    plt.plot(b_arr, b_arr*theta_arr, marker="o")
    plt.xlabel("impact parameter b")
    plt.ylabel("b*abs(theta)")
    plt.title("Toy Model 4: 1/b bending check")
    plt.tight_layout()
    plt.savefig("toy_model_4_b_times_theta.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 8))
    t = np.linspace(0, 2*np.pi, 400)

    plt.plot(core_radius*np.cos(t), core_radius*np.sin(t), label="core/source")

    selected = [14, 26, 42, 58]

    for b0 in selected:
        row = min(rows, key=lambda r: abs(r["b"] - b0))
        plt.plot(row["xs"], row["ys"], label=f"b={row['b']}")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-0.9*L, 0.9*L)
    plt.ylim(-0.5*L, 0.8*L)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 4: trajectories through solved Poisson field")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4_trajectories.png", dpi=200)
    plt.show()


def save_csv(rows, summary):
    with open("toy_model_4_poisson_scattering_data.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "b",
            "v0",
            "theta_rad",
            "abs_theta_rad",
            "theta_deg",
            "abs_theta_deg",
            "min_r",
            "entered_core",
            "reached_right",
            "escaped"
        ])

        for row in rows:
            writer.writerow([
                row["b"],
                row["v0"],
                row["theta_rad"],
                row["abs_theta_rad"],
                row["theta_deg"],
                row["abs_theta_deg"],
                row["min_r"],
                row["entered_core"],
                row["reached_right"],
                row["escaped"]
            ])

    with open("toy_model_4_poisson_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    print("Saved CSV files:")
    print("toy_model_4_poisson_scattering_data.csv")
    print("toy_model_4_poisson_summary.csv")


# ------------------------------------------------------------
# 7. Main run
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n==============================================================")
    print("TOY MODEL 4: POISSON FIELD EQUATION FROM NETWORK SOURCE")
    print("==============================================================")

    # Representative network parameters from Toy Model 1 / Toy Model 3c.
    alpha = 8.0
    beta = 0.25
    radius = 6.0
    gamma = 1.0

    # Grid controls.
    N = 51
    L = 90.0
    iterations = 2500

    # Particle controls.
    v0 = 2.0
    b_values = [14, 18, 22, 26, 30, 36, 42, 50, 58]

    dK = delta_K_excess(alpha, beta, radius)
    mu_eff = network_source_strength(alpha, beta, radius, gamma)

    print("\nFormal:")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    F = -grad(Phi)")
    print("    mu_eff = gamma*abs(delta_K_excess)")

    print("\nMicrosoft/plain:")
    print("    laplacian(phi) = 4*pi*network_density")
    print("    force = -gradient(phi)")
    print("    mu_eff = gamma*abs(delta_K_excess)")

    print("\nNormal:")
    print("    We solve the field equation directly from the network source.")

    print("\nNetwork source:")
    print("alpha =", alpha)
    print("beta =", beta)
    print("radius =", radius)
    print("delta_K_excess =", dK)
    print("mu_eff =", mu_eff)

    x, dx, X, Y, Z, R = build_grid(N=N, L=L)
    rho = gaussian_network_density(R, radius, mu_eff, dx)

    total_rho = np.sum(rho) * dx**3

    print("\nGrid:")
    print("N =", N)
    print("L =", L)
    print("dx =", dx)
    print("integral rho dV =", total_rho)

    phi = solve_poisson_3d(
        rho,
        R,
        dx,
        mu_eff,
        radius,
        iterations=iterations,
        omega=1.0
    )

    Fx, Fy, Fz = compute_force(phi, dx)

    mean_res, max_res = poisson_residual(phi, rho, dx)

    fit = fit_potential_and_force(x, phi, Fx, Fy, Fz, radius, L)

    print("\n==============================================================")
    print("FIELD EQUATION TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    Phi(r) should fit A/r + B")
    print("    |F(r)| should fit A/r^p with p close to 2")

    print("\nMicrosoft/plain:")
    print("    phi ~ 1/r")
    print("    force ~ 1/r^2")

    print("\nNormal:")
    print("    If the Poisson field is gravity-like, the far field is inverse-square.")

    print("\nPoisson residual:")
    print("mean abs residual =", mean_res)
    print("max abs residual  =", max_res)

    print("\nPotential fit:")
    print("R2_phi =", fit["R2_phi"])

    print("\nForce fit:")
    print("p_force =", fit["p_force"])
    print("R2_force =", fit["R2_force"])

    print("\nRunning particle scattering through solved field...")

    rows = []

    for b in b_values:
        row = simulate_scattering(
            x,
            Fx,
            Fy,
            mu_eff,
            radius,
            b=b,
            v0=v0,
            L=L,
            dt=0.05
        )

        rows.append(row)

        print(
            f"b={b:5.1f} | theta={row['abs_theta_deg']:.6f} deg | "
            f"min_r={row['min_r']:.3f} | reached={row['reached_right']}"
        )

    q, R2_scatter, p_scatter, usable = fit_scattering(rows)

    print("\n==============================================================")
    print("SCATTERING TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    |theta| proportional to 1/b")

    print("\nMicrosoft/plain:")
    print("    abs(deflection_angle) ~ 1/b")

    print("\nNormal:")
    print("    A particle moving through the solved field should bend like weak gravity.")

    print("\nScattering fit:")
    print("R2_1_over_b =", R2_scatter)
    print("p_scatter =", p_scatter)
    print("usable trajectories =", usable)

    summary = {
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "delta_K_excess": dK,
        "mu_eff": mu_eff,
        "N": N,
        "L": L,
        "dx": dx,
        "iterations": iterations,
        "rho_integral": total_rho,
        "poisson_mean_abs_residual": mean_res,
        "poisson_max_abs_residual": max_res,
        "R2_phi_1_over_r": fit["R2_phi"],
        "p_force": fit["p_force"],
        "R2_force_power": fit["R2_force"],
        "R2_scattering_1_over_b": R2_scatter,
        "p_scattering": p_scatter,
        "usable_scattering": usable,
    }

    save_csv(rows, summary)

    make_plots(
        x,
        rho,
        phi,
        Fx,
        Fy,
        Fz,
        fit,
        rows,
        radius,
        L
    )

    print("\n==============================================================")
    print("FINAL TOY MODEL 4 SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    mu_eff = gamma*abs(delta_K_excess)")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    F = -grad(Phi)")
    print("    |theta| ~ 1/b")

    print("\nMicrosoft/plain:")
    print("    mu_eff = gamma*abs(delta_K_excess)")
    print("    laplacian(phi) = 4*pi*network_density")
    print("    force = -gradient(phi)")
    print("    abs(deflection_angle) ~ 1/b")

    print("\nNormal:")
    print("    The network curvature source is turned into a density.")
    print("    The density solves a Poisson field equation.")
    print("    The solved potential produces a force.")
    print("    The force bends particles dynamically.")

    print("\nSource:")
    print("    delta_K_excess =", dK)
    print("    mu_eff =", mu_eff)

    print("\nField equation results:")
    print("    R2_phi_1_over_r =", fit["R2_phi"])
    print("    p_force =", fit["p_force"])
    print("    R2_force_power =", fit["R2_force"])

    print("\nScattering results:")
    print("    R2_scattering_1_over_b =", R2_scatter)
    print("    p_scattering =", p_scatter)
    print("    usable trajectories =", usable)

    field_pass = (
        fit["R2_phi"] > 0.98
        and abs(fit["p_force"] - 2.0) < 0.15
    )

    scatter_pass = (
        R2_scatter > 0.90
        and abs(p_scatter - 1.0) < 0.25
    )

    if field_pass:
        print("\nFIELD EQUATION RESULT: PASS")
        print("    The solved Poisson field gives phi ~ 1/r and force ~ 1/r^2.")
    else:
        print("\nFIELD EQUATION RESULT: PARTIAL / CHECK")

    if scatter_pass:
        print("\nDYNAMIC RESULT: PASS")
        print("    The solved field gives approximately 1/b weak bending.")
    else:
        print("\nDYNAMIC RESULT: PARTIAL / CHECK")

    print("\nInterpretation:")
    print("    Toy Model 4 replaces hand-built radial force with a field equation.")
    print("    This is the Newtonian-like bridge:")
    print("    network source -> Poisson potential -> force -> motion")
    print("==============================================================")
