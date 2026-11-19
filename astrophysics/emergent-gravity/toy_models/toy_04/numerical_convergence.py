import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 4c: NUMERICAL ROBUSTNESS / CONVERGENCE TEST
# Network source -> 3D Poisson solve -> force -> bending
# ==============================================================
#
# Formal:
#     mu_eff = gamma*|Delta K_excess|
#     laplacian(Phi) = 4*pi*rho_network
#     F = -grad(Phi)
#     |theta| ~ mu_eff/(b*v0^2)
#
# Microsoft/plain:
#     mu_eff = gamma*abs(delta_K_excess)
#     laplacian(phi) = 4*pi*network_density
#     force = -gradient(phi)
#     abs(deflection_angle) ~ mu_eff/(b*v0^2)
#
# Normal:
#     Toy Model 4c checks whether Toy Model 4 survives changes in
#     grid size, box size, source width, and solver accuracy.

EPS = 1e-12


# --------------------------------------------------------------
# USER CONTROLS
# --------------------------------------------------------------

ALPHA = 8.0
BETA = 0.25
RADIUS = 6.0
GAMMA = 1.0

V0 = 2.0
B_VALUES = [14, 18, 22, 26, 30, 36, 42, 50, 58]
DT = 0.05

# If this feels slow, set RUN_ALL_CONFIGS = False.
RUN_ALL_CONFIGS = True

BASELINE_CONFIGS = [
    {"name": "baseline",       "N": 51, "L": 90.0,  "iterations": 2500, "sigma_factor": 0.50},
    {"name": "coarse_grid",    "N": 41, "L": 90.0,  "iterations": 1800, "sigma_factor": 0.50},
    {"name": "finer_grid",     "N": 57, "L": 90.0,  "iterations": 2800, "sigma_factor": 0.50},
    {"name": "small_box",      "N": 51, "L": 75.0,  "iterations": 2500, "sigma_factor": 0.50},
    {"name": "large_box",      "N": 51, "L": 105.0, "iterations": 2500, "sigma_factor": 0.50},
    {"name": "narrow_source",  "N": 51, "L": 90.0,  "iterations": 2500, "sigma_factor": 0.35},
    {"name": "wide_source",    "N": 51, "L": 90.0,  "iterations": 2500, "sigma_factor": 0.75},
    {"name": "low_iterations", "N": 51, "L": 90.0,  "iterations": 1000, "sigma_factor": 0.50},
]

SHORT_CONFIGS = [
    {"name": "baseline",       "N": 51, "L": 90.0, "iterations": 2500, "sigma_factor": 0.50},
    {"name": "coarse_grid",    "N": 41, "L": 90.0, "iterations": 1800, "sigma_factor": 0.50},
    {"name": "narrow_source",  "N": 51, "L": 90.0, "iterations": 2500, "sigma_factor": 0.35},
    {"name": "wide_source",    "N": 51, "L": 90.0, "iterations": 2500, "sigma_factor": 0.75},
]

CONFIGS = BASELINE_CONFIGS if RUN_ALL_CONFIGS else SHORT_CONFIGS

SUMMARY_CSV = "toy_model_4c_convergence_summary.csv"
SCATTER_CSV = "toy_model_4c_scattering_data.csv"


# --------------------------------------------------------------
# CLOSED-FORM TOY MODEL 1 NETWORK SOURCE
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
        This is the exact Toy Model 1 curvature source.
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


def source_strength(alpha, beta, radius, gamma=1.0):
    """
    Formal:
        mu_eff = gamma*|Delta K_excess|

    Microsoft/plain:
        mu_eff = gamma*abs(delta_K_excess)

    Normal:
        The graph-curvature response becomes effective gravitational source.
    """
    return gamma * abs(delta_K_excess(alpha, beta, radius))


# --------------------------------------------------------------
# GRID, SOURCE, AND 3D POISSON SOLVER
# --------------------------------------------------------------


def build_grid(N, L):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X*X + Y*Y + Z*Z)
    return x, dx, X, Y, Z, R


def gaussian_density(R, radius, mu_eff, dx, sigma_factor):
    """
    Formal:
        rho(r) proportional to exp(-r^2/(2*sigma^2))
        integral rho dV = mu_eff

    Microsoft/plain:
        total_density = mu_eff

    Normal:
        We vary sigma_factor to test source-width robustness.
    """
    sigma = max(0.75 * dx, sigma_factor * radius)
    rho = np.exp(-0.5 * (R / sigma) ** 2)
    total = np.sum(rho) * dx**3
    rho = rho * (mu_eff / (total + EPS))
    return rho, sigma


def boundary_potential(R, mu_eff, softening):
    """
    Formal:
        Phi_boundary = -mu_eff/sqrt(r^2 + softening^2)

    Microsoft/plain:
        boundary_phi = -mu_eff/softened_r
    """
    return -mu_eff / np.sqrt(R*R + softening*softening)


def solve_poisson_3d(rho, R, dx, mu_eff, radius, iterations, sigma):
    """
    Formal:
        laplacian(Phi) = 4*pi*rho

        Phi_new = (sum_neighbour_phi - dx^2*4*pi*rho)/6

    Microsoft/plain:
        solve laplacian(phi)=4*pi*rho by Jacobi iteration

    Normal:
        This is the numerical part whose robustness we are testing.
    """
    S = 4.0 * math.pi * rho
    dx2 = dx * dx

    softening = max(radius / 3.0, sigma / 2.0)
    phi = boundary_potential(R, mu_eff, softening)
    boundary = phi.copy()

    last_update = np.nan

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

        phi[1:-1, 1:-1, 1:-1] = jacobi

        # Fixed Dirichlet boundary.
        phi[0, :, :] = boundary[0, :, :]
        phi[-1, :, :] = boundary[-1, :, :]
        phi[:, 0, :] = boundary[:, 0, :]
        phi[:, -1, :] = boundary[:, -1, :]
        phi[:, :, 0] = boundary[:, :, 0]
        phi[:, :, -1] = boundary[:, :, -1]

        if it == 1 or it == iterations or it % 500 == 0:
            last_update = float(np.max(np.abs(phi[1:-1, 1:-1, 1:-1] - old_center)))
            print("      iteration %5d | max update = %.6e" % (it, last_update))

    return phi, last_update


def compute_force(phi, dx):
    """
    Formal:
        F = -grad(Phi)

    Microsoft/plain:
        force = -gradient(phi)
    """
    dphi_dx, dphi_dy, dphi_dz = np.gradient(phi, dx, dx, dx, edge_order=2)
    return -dphi_dx, -dphi_dy, -dphi_dz


def poisson_residual(phi, rho, dx):
    """
    Formal:
        residual = laplacian(Phi) - 4*pi*rho

    Microsoft/plain:
        residual = left_side - right_side
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

    res = lap - 4.0 * math.pi * rho
    interior = res[2:-2, 2:-2, 2:-2]

    return float(np.mean(np.abs(interior))), float(np.max(np.abs(interior)))


# --------------------------------------------------------------
# FITTING TOOLS
# --------------------------------------------------------------


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if len(y) < 2:
        return np.nan

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot <= EPS:
        return np.nan

    return 1.0 - ss_res / ss_tot


def fit_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = np.sum(x*x)
    if denom <= EPS:
        return np.nan, np.full_like(y, np.nan), np.nan

    q = np.sum(x*y) / denom
    pred = q*x
    return q, pred, r2_score(y, pred)


def fit_power_law(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0)

    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, np.nan

    lx = np.log(x[mask])
    ly = np.log(y[mask])

    X = np.vstack([np.ones_like(lx), -lx]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)

    log_A = coeffs[0]
    p = coeffs[1]

    pred_log = log_A - p*lx
    R2_log = r2_score(ly, pred_log)

    return math.exp(log_A), p, pred_log, R2_log


def fit_field_shape(x, phi, Fx, Fy, Fz, radius, L):
    mid = len(x) // 2

    r_axis = x[mid+1:]
    phi_axis = phi[mid+1:, mid, mid]
    F_axis = np.sqrt(
        Fx[mid+1:, mid, mid]**2
        + Fy[mid+1:, mid, mid]**2
        + Fz[mid+1:, mid, mid]**2
    )

    r_min = max(2.5 * radius, 12.0)
    r_max = 0.70 * L
    mask = (r_axis > r_min) & (r_axis < r_max) & (F_axis > 0)

    r_fit = r_axis[mask]
    phi_fit = phi_axis[mask]
    F_fit = F_axis[mask]

    if len(r_fit) < 5:
        return {
            "A_phi": np.nan,
            "B_phi": np.nan,
            "R2_phi": np.nan,
            "A_force": np.nan,
            "p_force": np.nan,
            "R2_force": np.nan,
            "r_axis": r_axis,
            "phi_axis": phi_axis,
            "F_axis": F_axis,
        }

    Xmat = np.vstack([1.0 / r_fit, np.ones_like(r_fit)]).T
    coeffs, _, _, _ = np.linalg.lstsq(Xmat, phi_fit, rcond=None)
    A_phi, B_phi = coeffs
    phi_pred = A_phi / r_fit + B_phi
    R2_phi = r2_score(phi_fit, phi_pred)

    A_force, p_force, pred_log, R2_force = fit_power_law(r_fit, F_fit)

    return {
        "A_phi": float(A_phi),
        "B_phi": float(B_phi),
        "R2_phi": float(R2_phi),
        "A_force": float(A_force),
        "p_force": float(p_force),
        "R2_force": float(R2_force),
        "r_axis": r_axis,
        "phi_axis": phi_axis,
        "F_axis": F_axis,
    }


# --------------------------------------------------------------
# PARTICLE DYNAMICS THROUGH THE SOLVED FIELD
# --------------------------------------------------------------


def analytic_acceleration(xp, yp, zp, mu_eff, softening):
    r2 = xp*xp + yp*yp + zp*zp + softening*softening
    r = math.sqrt(r2)
    factor = -mu_eff / (r2 * r)
    return factor*xp, factor*yp, factor*zp


def bilinear_interp(arr2d, grid, xp, yp):
    N = len(grid)
    dx = grid[1] - grid[0]
    xmin = grid[0]
    xmax = grid[-1]

    if xp <= xmin or xp >= xmax or yp <= xmin or yp >= xmax:
        return None

    u = (xp - xmin) / dx
    v = (yp - xmin) / dx

    i = int(math.floor(u))
    j = int(math.floor(v))

    if i < 0 or i >= N - 1 or j < 0 or j >= N - 1:
        return None

    du = u - i
    dv = v - j

    return float(
        (1.0-du)*(1.0-dv)*arr2d[i, j]
        + du*(1.0-dv)*arr2d[i+1, j]
        + (1.0-du)*dv*arr2d[i, j+1]
        + du*dv*arr2d[i+1, j+1]
    )


def rk4_step(state, dt, accel_func):
    def deriv(s):
        x, y, vx, vy = s
        ax, ay = accel_func(x, y)
        return np.array([vx, vy, ax, ay], dtype=float)

    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)

    return state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_scattering(x_grid, Fx, Fy, mu_eff, radius, sigma, b, v0, L, dt):
    mid = len(x_grid) // 2
    Fx_slice = Fx[:, :, mid]
    Fy_slice = Fy[:, :, mid]

    softening = max(radius / 3.0, sigma / 2.0)

    x_start = -0.85 * L
    x_end = 0.85 * L

    def accel_func(xp, yp):
        ax = bilinear_interp(Fx_slice, x_grid, xp, yp)
        ay = bilinear_interp(Fy_slice, x_grid, xp, yp)

        if ax is None or ay is None:
            ax, ay, _ = analytic_acceleration(xp, yp, 0.0, mu_eff, softening)

        return ax, ay

    state = np.array([x_start, float(b), float(v0), 0.0], dtype=float)

    xs = []
    ys = []
    min_r = 1e99
    entered_core = False
    reached = False
    escaped = False

    for _ in range(30000):
        xp, yp, vx, vy = state
        rr = math.sqrt(xp*xp + yp*yp)
        min_r = min(min_r, rr)

        xs.append(xp)
        ys.append(yp)

        if rr <= radius:
            entered_core = True

        if xp >= x_end and vx > 0:
            reached = True
            break

        if abs(xp) > 1.5*L or abs(yp) > 1.5*L:
            escaped = True
            break

        state = rk4_step(state, dt, accel_func)

    xp, yp, vx, vy = state
    theta = abs(math.atan2(vy, vx))

    usable = reached and (not entered_core) and theta > 1e-10

    return {
        "b": float(b),
        "v0": float(v0),
        "theta_rad": float(theta),
        "theta_deg": float(math.degrees(theta)),
        "min_r": float(min_r),
        "entered_core": int(entered_core),
        "reached": int(reached),
        "escaped": int(escaped),
        "usable": int(usable),
        "xs": xs,
        "ys": ys,
    }


def summarize_scattering(rows, mu_eff):
    usable = [row for row in rows if row["usable"] == 1]

    if len(usable) < 3:
        return {
            "usable": len(usable),
            "Q_1_over_b": np.nan,
            "R2_1_over_b": np.nan,
            "p_scatter": np.nan,
            "R2_power": np.nan,
            "C_mean": np.nan,
            "C_std": np.nan,
            "C_cv": np.nan,
        }

    b = np.array([row["b"] for row in usable], dtype=float)
    theta = np.array([row["theta_rad"] for row in usable], dtype=float)
    v0 = np.array([row["v0"] for row in usable], dtype=float)

    x = 1.0 / b
    Q, pred, R2_inv = fit_origin(x, theta)

    A, p, pred_log, R2_pow = fit_power_law(b, theta)

    C = theta * b * v0*v0 / (mu_eff + EPS)

    C_mean = float(np.mean(C))
    C_std = float(np.std(C))
    C_cv = C_std / C_mean if abs(C_mean) > EPS else np.nan

    return {
        "usable": len(usable),
        "Q_1_over_b": float(Q),
        "R2_1_over_b": float(R2_inv),
        "p_scatter": float(p),
        "R2_power": float(R2_pow),
        "C_mean": C_mean,
        "C_std": C_std,
        "C_cv": float(C_cv),
    }


# --------------------------------------------------------------
# CSV AND PLOTS
# --------------------------------------------------------------


def save_dict_rows(filename, rows):
    if not rows:
        print("No rows to save:", filename)
        return

    keys = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filename)


def make_plots(summary_rows, scatter_rows, baseline_name):
    names = [row["config"] for row in summary_rows]
    idx = np.arange(len(names))

    p_force = np.array([row["p_force"] for row in summary_rows], dtype=float)
    p_scatter = np.array([row["p_scatter"] for row in summary_rows], dtype=float)
    C_mean = np.array([row["C_mean"] for row in summary_rows], dtype=float)
    R2_scatter = np.array([row["R2_1_over_b"] for row in summary_rows], dtype=float)
    residual = np.array([row["mean_abs_residual"] for row in summary_rows], dtype=float)

    baseline_C = C_mean[0]

    plt.figure(figsize=(11, 6))
    plt.bar(idx, p_force)
    plt.axhline(2.0, linestyle="--", label="target p=2")
    plt.xticks(idx, names, rotation=35, ha="right")
    plt.ylabel("force power p")
    plt.title("Toy Model 4c: force-law robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4c_force_power_by_config.png", dpi=200)
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.bar(idx, p_scatter)
    plt.axhline(1.0, linestyle="--", label="target p=1")
    plt.xticks(idx, names, rotation=35, ha="right")
    plt.ylabel("scattering power p")
    plt.title("Toy Model 4c: scattering-law robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4c_scattering_power_by_config.png", dpi=200)
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.bar(idx, C_mean)
    plt.axhline(baseline_C, linestyle="--", label="baseline C")
    plt.xticks(idx, names, rotation=35, ha="right")
    plt.ylabel("C = theta*b*v0^2/mu_eff")
    plt.title("Toy Model 4c: collapse constant robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4c_C_mean_by_config.png", dpi=200)
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.bar(idx, R2_scatter)
    plt.axhline(0.90, linestyle="--", label="pass threshold")
    plt.xticks(idx, names, rotation=35, ha="right")
    plt.ylabel("R2 for theta ~ 1/b")
    plt.title("Toy Model 4c: inverse-b fit robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4c_R2_scattering_by_config.png", dpi=200)
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.semilogy(idx, residual, marker="o")
    plt.xticks(idx, names, rotation=35, ha="right")
    plt.ylabel("mean abs Poisson residual")
    plt.title("Toy Model 4c: Poisson residual by config")
    plt.tight_layout()
    plt.savefig("toy_model_4c_poisson_residual_by_config.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 7))
    for name in names:
        subset = [row for row in scatter_rows if row["config"] == name and row["usable"] == 1]
        subset = sorted(subset, key=lambda row: row["b"])
        if not subset:
            continue
        b = np.array([row["b"] for row in subset], dtype=float)
        theta = np.array([row["theta_deg"] for row in subset], dtype=float)
        plt.plot(b, theta, marker="o", label=name)

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 4c: deflection curves across numerical choices")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_4c_deflection_curves.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_4c_force_power_by_config.png")
    print("toy_model_4c_scattering_power_by_config.png")
    print("toy_model_4c_C_mean_by_config.png")
    print("toy_model_4c_R2_scattering_by_config.png")
    print("toy_model_4c_poisson_residual_by_config.png")
    print("toy_model_4c_deflection_curves.png")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------


def main():
    print("")
    print("==============================================================")
    print("TOY MODEL 4c: NUMERICAL ROBUSTNESS / CONVERGENCE TEST")
    print("==============================================================")
    print("")
    print("Formal:")
    print("    mu_eff = gamma*|Delta K_excess|")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    F = -grad(Phi)")
    print("    |theta| ~ mu_eff/(b*v0^2)")
    print("")
    print("Microsoft/plain:")
    print("    mu_eff = gamma*abs(delta_K_excess)")
    print("    laplacian(phi) = 4*pi*network_density")
    print("    force = -gradient(phi)")
    print("    abs(deflection_angle) ~ mu_eff/(b*v0^2)")
    print("")
    print("Normal:")
    print("    We now test whether Toy Model 4 was robust, not lucky.")
    print("==============================================================")

    dK = delta_K_excess(ALPHA, BETA, RADIUS)
    mu_eff = source_strength(ALPHA, BETA, RADIUS, GAMMA)

    print("")
    print("Network source:")
    print("    alpha =", ALPHA)
    print("    beta =", BETA)
    print("    radius =", RADIUS)
    print("    delta_K_excess =", dK)
    print("    mu_eff =", mu_eff)
    print("")

    summary_rows = []
    scatter_rows = []

    for k, cfg in enumerate(CONFIGS, start=1):
        name = cfg["name"]
        N = int(cfg["N"])
        L = float(cfg["L"])
        iterations = int(cfg["iterations"])
        sigma_factor = float(cfg["sigma_factor"])

        print("--------------------------------------------------------------")
        print("Config %d/%d: %s" % (k, len(CONFIGS), name))
        print("    N =", N)
        print("    L =", L)
        print("    iterations =", iterations)
        print("    sigma_factor =", sigma_factor)

        x, dx, X, Y, Z, R = build_grid(N, L)
        rho, sigma = gaussian_density(R, RADIUS, mu_eff, dx, sigma_factor)
        rho_integral = float(np.sum(rho) * dx**3)

        print("    dx =", dx)
        print("    sigma =", sigma)
        print("    integral rho dV =", rho_integral)
        print("    Solving Poisson equation...")

        phi, last_update = solve_poisson_3d(
            rho=rho,
            R=R,
            dx=dx,
            mu_eff=mu_eff,
            radius=RADIUS,
            iterations=iterations,
            sigma=sigma,
        )

        Fx, Fy, Fz = compute_force(phi, dx)
        mean_res, max_res = poisson_residual(phi, rho, dx)
        field_fit = fit_field_shape(x, phi, Fx, Fy, Fz, RADIUS, L)

        print("    Field fit:")
        print("        R2_phi_1_over_r =", field_fit["R2_phi"])
        print("        p_force =", field_fit["p_force"])
        print("        R2_force_power =", field_fit["R2_force"])
        print("    Residual:")
        print("        mean_abs_residual =", mean_res)
        print("        max_abs_residual  =", max_res)

        case_scatter_rows = []

        print("    Running trajectories...")
        for b in B_VALUES:
            traj = simulate_scattering(
                x_grid=x,
                Fx=Fx,
                Fy=Fy,
                mu_eff=mu_eff,
                radius=RADIUS,
                sigma=sigma,
                b=b,
                v0=V0,
                L=L,
                dt=DT,
            )

            row = {
                "config": name,
                "N": N,
                "L": L,
                "dx": dx,
                "iterations": iterations,
                "sigma_factor": sigma_factor,
                "sigma": sigma,
                "alpha": ALPHA,
                "beta": BETA,
                "radius": RADIUS,
                "delta_K_excess": dK,
                "mu_eff": mu_eff,
                "b": traj["b"],
                "v0": traj["v0"],
                "theta_rad": traj["theta_rad"],
                "theta_deg": traj["theta_deg"],
                "min_r": traj["min_r"],
                "entered_core": traj["entered_core"],
                "reached": traj["reached"],
                "escaped": traj["escaped"],
                "usable": traj["usable"],
            }

            scatter_rows.append(row)
            case_scatter_rows.append(row)

            print(
                "        b=%5.1f | theta=%10.6f deg | min_r=%8.3f | usable=%d"
                % (b, traj["theta_deg"], traj["min_r"], traj["usable"])
            )

        scatter_fit = summarize_scattering(case_scatter_rows, mu_eff)

        field_pass = (
            field_fit["R2_phi"] > 0.98
            and abs(field_fit["p_force"] - 2.0) < 0.20
            and field_fit["R2_force"] > 0.98
        )

        scatter_pass = (
            scatter_fit["R2_1_over_b"] > 0.90
            and 0.75 <= scatter_fit["p_scatter"] <= 1.35
            and scatter_fit["C_cv"] < 0.15
        )

        config_pass = field_pass and scatter_pass

        summary = {
            "config": name,
            "N": N,
            "L": L,
            "dx": dx,
            "iterations": iterations,
            "sigma_factor": sigma_factor,
            "sigma": sigma,
            "alpha": ALPHA,
            "beta": BETA,
            "radius": RADIUS,
            "delta_K_excess": dK,
            "mu_eff": mu_eff,
            "rho_integral": rho_integral,
            "last_update": last_update,
            "mean_abs_residual": mean_res,
            "max_abs_residual": max_res,
            "R2_phi_1_over_r": field_fit["R2_phi"],
            "p_force": field_fit["p_force"],
            "R2_force_power": field_fit["R2_force"],
            "usable_scattering": scatter_fit["usable"],
            "Q_1_over_b": scatter_fit["Q_1_over_b"],
            "R2_1_over_b": scatter_fit["R2_1_over_b"],
            "p_scatter": scatter_fit["p_scatter"],
            "R2_scatter_power": scatter_fit["R2_power"],
            "C_mean": scatter_fit["C_mean"],
            "C_std": scatter_fit["C_std"],
            "C_cv": scatter_fit["C_cv"],
            "field_pass": int(field_pass),
            "scatter_pass": int(scatter_pass),
            "config_pass": int(config_pass),
        }

        summary_rows.append(summary)

        print("    Scattering fit:")
        print("        R2_1_over_b =", scatter_fit["R2_1_over_b"])
        print("        p_scatter =", scatter_fit["p_scatter"])
        print("        C_mean =", scatter_fit["C_mean"])
        print("        C_cv =", scatter_fit["C_cv"])
        print("    Result:", "PASS" if config_pass else "CHECK")

    save_dict_rows(SUMMARY_CSV, summary_rows)
    save_dict_rows(SCATTER_CSV, scatter_rows)

    pass_count = sum(row["config_pass"] for row in summary_rows)
    total_count = len(summary_rows)

    p_force_values = np.array([row["p_force"] for row in summary_rows], dtype=float)
    p_scatter_values = np.array([row["p_scatter"] for row in summary_rows], dtype=float)
    C_values = np.array([row["C_mean"] for row in summary_rows], dtype=float)

    baseline_C = summary_rows[0]["C_mean"]
    C_relative_spread = float(np.std(C_values) / np.mean(C_values))
    C_max_frac_from_baseline = float(np.max(np.abs(C_values - baseline_C) / (abs(baseline_C) + EPS)))

    print("")
    print("==============================================================")
    print("FINAL TOY MODEL 4c SUMMARY")
    print("==============================================================")
    print("")
    print("Formal:")
    print("    Test stability under N, L, dx, sigma_rho, iterations")
    print("    Phi(r) ~ 1/r")
    print("    |F(r)| ~ 1/r^2")
    print("    |theta(b)| ~ 1/b")
    print("    C = theta*b*v0^2/mu_eff approximately constant")
    print("")
    print("Microsoft/plain:")
    print("    change grid/domain/source_width/iterations")
    print("    check phi ~ 1/r")
    print("    check force ~ 1/r^2")
    print("    check theta ~ 1/b")
    print("    check C stays nearly constant")
    print("")
    print("Normal:")
    print("    This checks whether Toy Model 4 was numerically stable.")
    print("")
    print("Configs passed:")
    print("   ", str(pass_count) + "/" + str(total_count))
    print("")
    print("Field robustness:")
    print("    mean p_force =", float(np.mean(p_force_values)))
    print("    std  p_force =", float(np.std(p_force_values)))
    print("    expected p_force close to 2")
    print("")
    print("Scattering robustness:")
    print("    mean p_scatter =", float(np.mean(p_scatter_values)))
    print("    std  p_scatter =", float(np.std(p_scatter_values)))
    print("    expected p_scatter close to 1")
    print("")
    print("Collapse constant robustness:")
    print("    mean C =", float(np.mean(C_values)))
    print("    std  C =", float(np.std(C_values)))
    print("    relative C spread =", C_relative_spread)
    print("    max fractional change from baseline C =", C_max_frac_from_baseline)
    print("")

    if pass_count == total_count and C_relative_spread < 0.10:
        print("RESULT: PASS")
        print("Toy Model 4 is robust under the tested numerical choices.")
        print("This supports the bridge:")
        print("network source -> Poisson field -> inverse-square force -> 1/b bending")
    elif pass_count >= max(1, int(0.75 * total_count)):
        print("RESULT: MOSTLY PASS / CHECK DETAILS")
        print("The bridge is mostly stable, but at least one numerical choice should be inspected.")
    else:
        print("RESULT: MIXED")
        print("The result may depend too strongly on numerical choices.")

    print("==============================================================")

    make_plots(summary_rows, scatter_rows, baseline_name="baseline")


if __name__ == "__main__":
    main()
