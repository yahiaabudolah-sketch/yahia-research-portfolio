import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 5a: TWO-SOURCE GRAVITY / SUPERPOSITION TEST
# ==============================================================
#
# Formal:
#     mu_eff = gamma*|Delta K_excess|
#     rho_total = rho_1 + rho_2
#     laplacian(Phi_total) = 4*pi*rho_total
#     Phi_total ?= Phi_1 + Phi_2
#     F_total ?= F_1 + F_2
#
# Microsoft/plain:
#     mu_eff = gamma*abs(delta_K_excess)
#     rho_total = rho_1 + rho_2
#     laplacian(phi_total) = 4*pi*rho_total
#     phi_total should equal phi_1 + phi_2
#     force_total should equal force_1 + force_2
#
# Normal:
#     Toy Model 5a checks whether two independent network defects
#     behave like additive gravitational sources in the weak-field limit.

EPS = 1e-12


# --------------------------------------------------------------
# USER CONTROLS
# --------------------------------------------------------------

# Source 1 network parameters
ALPHA_1 = 8.0
BETA_1 = 0.25
RADIUS_1 = 6.0

# Source 2 network parameters
ALPHA_2 = 4.0
BETA_2 = 0.35
RADIUS_2 = 6.0

GAMMA = 1.0

# Two-source geometry
SOURCE_SEPARATION = 20.0
SOURCE_1_POS = (-SOURCE_SEPARATION / 2.0, 0.0, 0.0)
SOURCE_2_POS = ( SOURCE_SEPARATION / 2.0, 0.0, 0.0)

# 3D Poisson grid
N = 51
L = 100.0
ITERATIONS = 2200
SIGMA_FACTOR = 0.50

# Particle scattering
V0 = 2.4
B_VALUES = [16, 20, 24, 30, 36, 44, 54, 66]
DT = 0.05

SUMMARY_CSV = "toy_model_5a_summary.csv"
SCATTER_CSV = "toy_model_5a_scattering_data.csv"


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
# GRID, SOURCES, AND POISSON SOLVER
# --------------------------------------------------------------


def build_grid(N, L):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return x, dx, X, Y, Z


def shifted_radius(X, Y, Z, pos):
    x0, y0, z0 = pos
    return np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)


def gaussian_density_at_source(X, Y, Z, dx, pos, radius, mu_eff, sigma_factor):
    """
    Formal:
        rho_i(x) proportional to exp(-|x-x_i|^2/(2*sigma_i^2))
        integral rho_i dV = mu_i

    Microsoft/plain:
        density_i = shifted Gaussian
        total density_i = mu_i

    Normal:
        Each network defect becomes a localized smooth source density.
    """
    R = shifted_radius(X, Y, Z, pos)
    sigma = max(0.75*dx, sigma_factor*radius)
    rho = np.exp(-0.5*(R/sigma)**2)
    total = np.sum(rho) * dx**3
    rho = rho * (mu_eff / (total + EPS))
    return rho, sigma


def boundary_potential_for_sources(X, Y, Z, sources):
    """
    Formal:
        Phi_boundary = sum_i -mu_i/sqrt(|x-x_i|^2 + eps_i^2)

    Microsoft/plain:
        boundary_phi = sum of softened -mu/r fields

    Normal:
        The total boundary condition is the sum of the individual
        source boundary conditions. This makes the superposition test fair.
    """
    phi = np.zeros_like(X, dtype=float)
    for src in sources:
        R = shifted_radius(X, Y, Z, src["pos"])
        soft = src["softening"]
        phi += -src["mu"] / np.sqrt(R*R + soft*soft)
    return phi


def solve_poisson_3d(rho, boundary, dx, iterations, label):
    """
    Formal:
        laplacian(Phi) = 4*pi*rho
        Phi_new = (sum_neighbour_phi - dx^2*4*pi*rho)/6

    Microsoft/plain:
        solve laplacian(phi)=4*pi*rho using Jacobi iteration

    Normal:
        This solves the field equation for one source or both sources.
    """
    S = 4.0 * math.pi * rho
    dx2 = dx*dx
    phi = boundary.copy()

    print("\nSolving Poisson equation for", label)
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
            - dx2*S[1:-1, 1:-1, 1:-1]
        ) / 6.0

        phi[1:-1, 1:-1, 1:-1] = jacobi

        # Dirichlet boundary
        phi[0, :, :] = boundary[0, :, :]
        phi[-1, :, :] = boundary[-1, :, :]
        phi[:, 0, :] = boundary[:, 0, :]
        phi[:, -1, :] = boundary[:, -1, :]
        phi[:, :, 0] = boundary[:, :, 0]
        phi[:, :, -1] = boundary[:, :, -1]

        if it == 1 or it == iterations or it % 500 == 0:
            last_update = float(np.max(np.abs(phi[1:-1, 1:-1, 1:-1] - old_center)))
            print("  iteration %5d | max update = %.6e" % (it, last_update))

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
    lap = np.zeros_like(phi)
    lap[1:-1, 1:-1, 1:-1] = (
        phi[2:, 1:-1, 1:-1]
        + phi[:-2, 1:-1, 1:-1]
        + phi[1:-1, 2:, 1:-1]
        + phi[1:-1, :-2, 1:-1]
        + phi[1:-1, 1:-1, 2:]
        + phi[1:-1, 1:-1, :-2]
        - 6.0*phi[1:-1, 1:-1, 1:-1]
    ) / (dx*dx)

    res = lap - 4.0*math.pi*rho
    interior = res[2:-2, 2:-2, 2:-2]
    return float(np.mean(np.abs(interior))), float(np.max(np.abs(interior)))


# --------------------------------------------------------------
# FITTING AND ERROR METRICS
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
    return 1.0 - ss_res/ss_tot


def relative_l2(a, b):
    """
    Returns ||a-b||/||a||.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a-b)**2)) / (np.sqrt(np.sum(a*a)) + EPS))


def fit_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.sum(x*x)
    if denom <= EPS:
        return np.nan, np.full_like(y, np.nan), np.nan
    q = np.sum(x*y)/denom
    pred = q*x
    return float(q), pred, float(r2_score(y, pred))


def fit_power_law(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan

    lx = np.log(x[mask])
    ly = np.log(y[mask])
    Xmat = np.vstack([np.ones_like(lx), -lx]).T
    coeffs, _, _, _ = np.linalg.lstsq(Xmat, ly, rcond=None)
    logA, p = coeffs
    pred = logA - p*lx
    return float(math.exp(logA)), float(p), float(r2_score(ly, pred))


def far_field_fit(x, phi_total, Fx_total, Fy_total, Fz_total, sep, L):
    """
    Fit the far field along the positive y-axis at x=0,z=0.

    Formal:
        Phi(r) ~ A/r + B
        |F(r)| ~ A/r^p

    Microsoft/plain:
        far phi ~ 1/r
        far force ~ 1/r^2

    Normal:
        A binary source should look like one combined source far away.
    """
    mid = len(x)//2
    y_axis = x[mid+1:]
    phi_axis = phi_total[mid, mid+1:, mid]
    F_axis = np.sqrt(
        Fx_total[mid, mid+1:, mid]**2
        + Fy_total[mid, mid+1:, mid]**2
        + Fz_total[mid, mid+1:, mid]**2
    )

    r_min = max(1.3*sep, 25.0)
    r_max = 0.70*L
    mask = (y_axis > r_min) & (y_axis < r_max) & (F_axis > 0)

    r_fit = y_axis[mask]
    phi_fit = phi_axis[mask]
    F_fit = F_axis[mask]

    if len(r_fit) < 4:
        return {
            "R2_phi": np.nan,
            "p_force": np.nan,
            "R2_force": np.nan,
            "r_fit": r_fit,
            "phi_fit": phi_fit,
            "phi_pred": np.full_like(phi_fit, np.nan),
            "F_fit": F_fit,
            "F_pred": np.full_like(F_fit, np.nan),
        }

    Xmat = np.vstack([1.0/r_fit, np.ones_like(r_fit)]).T
    coeffs, _, _, _ = np.linalg.lstsq(Xmat, phi_fit, rcond=None)
    A_phi, B_phi = coeffs
    phi_pred = A_phi/r_fit + B_phi
    R2_phi = r2_score(phi_fit, phi_pred)

    A_force, p_force, R2_force = fit_power_law(r_fit, F_fit)
    F_pred = A_force/(r_fit**p_force)

    return {
        "R2_phi": float(R2_phi),
        "A_phi": float(A_phi),
        "B_phi": float(B_phi),
        "p_force": float(p_force),
        "R2_force": float(R2_force),
        "A_force": float(A_force),
        "r_fit": r_fit,
        "phi_fit": phi_fit,
        "phi_pred": phi_pred,
        "F_fit": F_fit,
        "F_pred": F_pred,
    }


# --------------------------------------------------------------
# PARTICLE DYNAMICS THROUGH GRID FIELD
# --------------------------------------------------------------


def bilinear_interp(arr2d, grid, xp, yp):
    N = len(grid)
    dx = grid[1] - grid[0]
    xmin = grid[0]
    xmax = grid[-1]

    if xp <= xmin or xp >= xmax or yp <= xmin or yp >= xmax:
        return None

    u = (xp - xmin)/dx
    v = (yp - xmin)/dx
    i = int(math.floor(u))
    j = int(math.floor(v))

    if i < 0 or i >= N-1 or j < 0 or j >= N-1:
        return None

    du = u - i
    dv = v - j

    return float(
        (1-du)*(1-dv)*arr2d[i, j]
        + du*(1-dv)*arr2d[i+1, j]
        + (1-du)*dv*arr2d[i, j+1]
        + du*dv*arr2d[i+1, j+1]
    )


def analytic_accel_from_sources(xp, yp, sources):
    ax = 0.0
    ay = 0.0
    for src in sources:
        x0, y0, z0 = src["pos"]
        dxp = xp - x0
        dyp = yp - y0
        soft = src["softening"]
        r2 = dxp*dxp + dyp*dyp + soft*soft
        r = math.sqrt(r2)
        fac = -src["mu"]/(r2*r)
        ax += fac*dxp
        ay += fac*dyp
    return ax, ay


def rk4_step(state, dt, accel_func):
    def deriv(s):
        x, y, vx, vy = s
        ax, ay = accel_func(x, y)
        return np.array([vx, vy, ax, ay], dtype=float)

    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate_scattering(x_grid, Fx, Fy, sources, b, v0, L, dt):
    mid = len(x_grid)//2
    Fx_slice = Fx[:, :, mid]
    Fy_slice = Fy[:, :, mid]

    x_start = -0.85*L
    x_end = 0.85*L

    def accel_func(xp, yp):
        ax = bilinear_interp(Fx_slice, x_grid, xp, yp)
        ay = bilinear_interp(Fy_slice, x_grid, xp, yp)
        if ax is None or ay is None:
            return analytic_accel_from_sources(xp, yp, sources)
        return ax, ay

    state = np.array([x_start, float(b), float(v0), 0.0], dtype=float)

    min_distance_to_any_source = 1e99
    reached = False
    entered_core = False
    escaped = False

    for _ in range(30000):
        xp, yp, vx, vy = state

        for src in sources:
            x0, y0, z0 = src["pos"]
            rr = math.sqrt((xp-x0)**2 + (yp-y0)**2)
            min_distance_to_any_source = min(min_distance_to_any_source, rr)
            if rr < src["radius"]:
                entered_core = True

        if xp >= x_end and vx > 0:
            reached = True
            break

        if abs(xp) > 1.5*L or abs(yp) > 1.5*L:
            escaped = True
            break

        state = rk4_step(state, dt, accel_func)

    xp, yp, vx, vy = state
    signed_theta = math.atan2(vy, vx)

    usable = reached and (not entered_core) and abs(signed_theta) > 1e-10

    return {
        "b": float(b),
        "v0": float(v0),
        "theta_signed_rad": float(signed_theta),
        "theta_abs_rad": float(abs(signed_theta)),
        "theta_signed_deg": float(math.degrees(signed_theta)),
        "theta_abs_deg": float(abs(math.degrees(signed_theta))),
        "min_distance_to_any_source": float(min_distance_to_any_source),
        "entered_core": int(entered_core),
        "reached": int(reached),
        "escaped": int(escaped),
        "usable": int(usable),
    }


# --------------------------------------------------------------
# OUTPUT HELPERS
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


def make_plots(x, rho_total, phi_total, fit, scatter_rows):
    mid = len(x)//2

    plt.figure(figsize=(8, 7))
    plt.imshow(
        rho_total[:, :, mid].T,
        origin="lower",
        extent=[x[0], x[-1], x[0], x[-1]],
    )
    plt.colorbar(label="rho_total")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 5a: two-source network density")
    plt.tight_layout()
    plt.savefig("toy_model_5a_density_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.imshow(
        phi_total[:, :, mid].T,
        origin="lower",
        extent=[x[0], x[-1], x[0], x[-1]],
    )
    plt.colorbar(label="Phi_total")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 5a: two-source Poisson potential")
    plt.tight_layout()
    plt.savefig("toy_model_5a_potential_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(fit["r_fit"], fit["phi_fit"], marker="o", label="solved far-field phi")
    plt.plot(fit["r_fit"], fit["phi_pred"], label="fit: A/r + B")
    plt.xlabel("distance from binary centre")
    plt.ylabel("Phi")
    plt.title("Toy Model 5a: far-field potential fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5a_far_potential_fit.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.loglog(fit["r_fit"], fit["F_fit"], marker="o", label="force magnitude")
    plt.loglog(fit["r_fit"], fit["F_pred"], label="fit: A/r^%.3f" % fit["p_force"])
    plt.xlabel("distance from binary centre")
    plt.ylabel("force magnitude")
    plt.title("Toy Model 5a: far-field force power law")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5a_far_force_power.png", dpi=200)
    plt.show()

    usable = [row for row in scatter_rows if row["usable_total"] == 1]
    b = np.array([row["b"] for row in usable], dtype=float)
    theta_total = np.array([row["theta_total_abs_deg"] for row in usable], dtype=float)
    theta_sum = np.array([abs(row["theta_1_signed_rad"] + row["theta_2_signed_rad"])*180.0/math.pi for row in usable], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(b, theta_total, marker="o", label="two-source trajectory")
    plt.plot(b, theta_sum, marker="o", label="single-source deflections added")
    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 5a: bending superposition")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5a_bending_superposition.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.scatter(theta_sum, theta_total, s=40)
    if len(theta_sum) > 0:
        lo = min(np.min(theta_sum), np.min(theta_total))
        hi = max(np.max(theta_sum), np.max(theta_total))
        plt.plot([lo, hi], [lo, hi], label="perfect superposition")
    plt.xlabel("|theta_1 + theta_2| in degrees")
    plt.ylabel("|theta_total| in degrees")
    plt.title("Toy Model 5a: deflection addition check")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_5a_deflection_addition_check.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_5a_density_slice.png")
    print("toy_model_5a_potential_slice.png")
    print("toy_model_5a_far_potential_fit.png")
    print("toy_model_5a_far_force_power.png")
    print("toy_model_5a_bending_superposition.png")
    print("toy_model_5a_deflection_addition_check.png")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------


def main():
    print("\n==============================================================")
    print("TOY MODEL 5a: TWO-SOURCE GRAVITY / SUPERPOSITION TEST")
    print("==============================================================")
    print("\nFormal:")
    print("    rho_total = rho_1 + rho_2")
    print("    laplacian(Phi_total) = 4*pi*rho_total")
    print("    Phi_total ?= Phi_1 + Phi_2")
    print("    F_total ?= F_1 + F_2")
    print("\nMicrosoft/plain:")
    print("    rho_total = rho_1 + rho_2")
    print("    laplacian(phi_total) = 4*pi*rho_total")
    print("    phi_total should equal phi_1 + phi_2")
    print("    force_total should equal force_1 + force_2")
    print("\nNormal:")
    print("    We test whether multiple network defects add like weak gravity.")

    dK1 = delta_K_excess(ALPHA_1, BETA_1, RADIUS_1)
    dK2 = delta_K_excess(ALPHA_2, BETA_2, RADIUS_2)
    mu1 = source_strength(ALPHA_1, BETA_1, RADIUS_1, GAMMA)
    mu2 = source_strength(ALPHA_2, BETA_2, RADIUS_2, GAMMA)
    mu_total = mu1 + mu2

    print("\nSource 1:")
    print("    alpha =", ALPHA_1, "beta =", BETA_1, "radius =", RADIUS_1)
    print("    delta_K_excess =", dK1)
    print("    mu_1 =", mu1)

    print("\nSource 2:")
    print("    alpha =", ALPHA_2, "beta =", BETA_2, "radius =", RADIUS_2)
    print("    delta_K_excess =", dK2)
    print("    mu_2 =", mu2)

    print("\nTotal effective source:")
    print("    mu_total = mu_1 + mu_2 =", mu_total)

    x, dx, X, Y, Z = build_grid(N, L)

    rho1, sigma1 = gaussian_density_at_source(X, Y, Z, dx, SOURCE_1_POS, RADIUS_1, mu1, SIGMA_FACTOR)
    rho2, sigma2 = gaussian_density_at_source(X, Y, Z, dx, SOURCE_2_POS, RADIUS_2, mu2, SIGMA_FACTOR)
    rho_total = rho1 + rho2

    src1 = {
        "name": "source_1",
        "pos": SOURCE_1_POS,
        "mu": mu1,
        "radius": RADIUS_1,
        "softening": max(RADIUS_1/3.0, sigma1/2.0),
    }
    src2 = {
        "name": "source_2",
        "pos": SOURCE_2_POS,
        "mu": mu2,
        "radius": RADIUS_2,
        "softening": max(RADIUS_2/3.0, sigma2/2.0),
    }

    boundary1 = boundary_potential_for_sources(X, Y, Z, [src1])
    boundary2 = boundary_potential_for_sources(X, Y, Z, [src2])
    boundary_total = boundary_potential_for_sources(X, Y, Z, [src1, src2])

    print("\nGrid:")
    print("    N =", N)
    print("    L =", L)
    print("    dx =", dx)
    print("    source separation =", SOURCE_SEPARATION)
    print("    integral rho_1 dV =", np.sum(rho1)*dx**3)
    print("    integral rho_2 dV =", np.sum(rho2)*dx**3)
    print("    integral rho_total dV =", np.sum(rho_total)*dx**3)

    phi1, update1 = solve_poisson_3d(rho1, boundary1, dx, ITERATIONS, "source 1")
    phi2, update2 = solve_poisson_3d(rho2, boundary2, dx, ITERATIONS, "source 2")
    phi_total, update_total = solve_poisson_3d(rho_total, boundary_total, dx, ITERATIONS, "source 1 + source 2")

    Fx1, Fy1, Fz1 = compute_force(phi1, dx)
    Fx2, Fy2, Fz2 = compute_force(phi2, dx)
    Fx_total, Fy_total, Fz_total = compute_force(phi_total, dx)

    mean_res_1, max_res_1 = poisson_residual(phi1, rho1, dx)
    mean_res_2, max_res_2 = poisson_residual(phi2, rho2, dx)
    mean_res_total, max_res_total = poisson_residual(phi_total, rho_total, dx)

    interior = np.zeros_like(phi_total, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True

    phi_sum = phi1 + phi2
    Fx_sum = Fx1 + Fx2
    Fy_sum = Fy1 + Fy2
    Fz_sum = Fz1 + Fz2

    phi_rel_error = relative_l2(phi_total[interior], phi_sum[interior])

    F_total_norm = np.sqrt(Fx_total[interior]**2 + Fy_total[interior]**2 + Fz_total[interior]**2)
    F_vector_diff_norm = np.sqrt(
        (Fx_total[interior] - Fx_sum[interior])**2
        + (Fy_total[interior] - Fy_sum[interior])**2
        + (Fz_total[interior] - Fz_sum[interior])**2
    )
    force_rel_error = float(np.sqrt(np.sum(F_vector_diff_norm**2)) / (np.sqrt(np.sum(F_total_norm**2)) + EPS))

    far_fit = far_field_fit(x, phi_total, Fx_total, Fy_total, Fz_total, SOURCE_SEPARATION, L)

    print("\n==============================================================")
    print("FIELD SUPERPOSITION TEST")
    print("==============================================================")
    print("\nFormal:")
    print("    epsilon_phi = ||Phi_total - (Phi_1 + Phi_2)|| / ||Phi_total||")
    print("    epsilon_F = ||F_total - (F_1 + F_2)|| / ||F_total||")
    print("\nMicrosoft/plain:")
    print("    phi_error = norm(phi_total - phi_sum)/norm(phi_total)")
    print("    force_error = norm(force_total - force_sum)/norm(force_total)")
    print("\nNormal:")
    print("    If weak-field superposition works, both errors should be very small.")
    print("\nResults:")
    print("    phi_relative_error =", phi_rel_error)
    print("    force_relative_error =", force_rel_error)
    print("    total residual mean =", mean_res_total)
    print("    total residual max  =", max_res_total)

    print("\nFar-field binary test:")
    print("    R2_phi_1_over_r =", far_fit["R2_phi"])
    print("    p_force =", far_fit["p_force"])
    print("    R2_force_power =", far_fit["R2_force"])

    print("\nRunning scattering tests...")

    scatter_rows = []

    for b in B_VALUES:
        traj1 = simulate_scattering(x, Fx1, Fy1, [src1], b, V0, L, DT)
        traj2 = simulate_scattering(x, Fx2, Fy2, [src2], b, V0, L, DT)
        traj_total = simulate_scattering(x, Fx_total, Fy_total, [src1, src2], b, V0, L, DT)

        theta_sum_signed = traj1["theta_signed_rad"] + traj2["theta_signed_rad"]
        theta_total_signed = traj_total["theta_signed_rad"]
        theta_addition_error = abs(theta_total_signed - theta_sum_signed)
        theta_addition_frac_error = theta_addition_error / (abs(theta_total_signed) + EPS)

        row = {
            "b": float(b),
            "v0": V0,
            "theta_1_signed_rad": traj1["theta_signed_rad"],
            "theta_2_signed_rad": traj2["theta_signed_rad"],
            "theta_total_signed_rad": traj_total["theta_signed_rad"],
            "theta_1_abs_deg": traj1["theta_abs_deg"],
            "theta_2_abs_deg": traj2["theta_abs_deg"],
            "theta_total_abs_deg": traj_total["theta_abs_deg"],
            "theta_sum_abs_deg": abs(theta_sum_signed)*180.0/math.pi,
            "theta_addition_error_rad": theta_addition_error,
            "theta_addition_frac_error": theta_addition_frac_error,
            "usable_1": traj1["usable"],
            "usable_2": traj2["usable"],
            "usable_total": traj_total["usable"],
            "min_distance_total": traj_total["min_distance_to_any_source"],
        }
        scatter_rows.append(row)

        print(
            "    b=%5.1f | theta_total=%9.6f deg | theta_sum=%9.6f deg | frac_error=%.6f"
            % (b, row["theta_total_abs_deg"], row["theta_sum_abs_deg"], theta_addition_frac_error)
        )

    usable = [r for r in scatter_rows if r["usable_total"] == 1 and r["usable_1"] == 1 and r["usable_2"] == 1]
    b_arr = np.array([r["b"] for r in usable], dtype=float)
    theta_total_abs = np.array([r["theta_total_abs_deg"]*math.pi/180.0 for r in usable], dtype=float)
    theta_sum_abs = np.array([r["theta_sum_abs_deg"]*math.pi/180.0 for r in usable], dtype=float)
    frac_errors = np.array([r["theta_addition_frac_error"] for r in usable], dtype=float)

    Q_inv, pred_inv, R2_inv_b = fit_origin(1.0/b_arr, theta_total_abs)
    A_pow, p_scatter, R2_power = fit_power_law(b_arr, theta_total_abs)
    Q_add, pred_add, R2_addition = fit_origin(theta_sum_abs, theta_total_abs)

    mean_frac_error = float(np.mean(frac_errors))
    max_frac_error = float(np.max(frac_errors))

    print("\n==============================================================")
    print("SCATTERING SUPERPOSITION TEST")
    print("==============================================================")
    print("\nFormal:")
    print("    theta_total approximately theta_1 + theta_2")
    print("    |theta_total| approximately 1/b")
    print("\nMicrosoft/plain:")
    print("    theta_total ~ theta_1 + theta_2")
    print("    abs(theta_total) ~ 1/b")
    print("\nNormal:")
    print("    Field superposition is exact for Poisson. Trajectory superposition")
    print("    should be approximate because particle motion is nonlinear.")
    print("\nResults:")
    print("    usable trajectories =", len(usable))
    print("    R2_total_vs_sum =", R2_addition)
    print("    slope_total_vs_sum =", Q_add)
    print("    mean fractional addition error =", mean_frac_error)
    print("    max fractional addition error =", max_frac_error)
    print("    R2_theta_total_1_over_b =", R2_inv_b)
    print("    p_scatter_total =", p_scatter)

    summary_rows = [
        {
            "alpha_1": ALPHA_1,
            "beta_1": BETA_1,
            "radius_1": RADIUS_1,
            "delta_K_1": dK1,
            "mu_1": mu1,
            "alpha_2": ALPHA_2,
            "beta_2": BETA_2,
            "radius_2": RADIUS_2,
            "delta_K_2": dK2,
            "mu_2": mu2,
            "mu_total": mu_total,
            "N": N,
            "L": L,
            "dx": dx,
            "iterations": ITERATIONS,
            "source_separation": SOURCE_SEPARATION,
            "rho1_integral": float(np.sum(rho1)*dx**3),
            "rho2_integral": float(np.sum(rho2)*dx**3),
            "rho_total_integral": float(np.sum(rho_total)*dx**3),
            "phi_relative_error": phi_rel_error,
            "force_relative_error": force_rel_error,
            "mean_residual_total": mean_res_total,
            "max_residual_total": max_res_total,
            "far_R2_phi_1_over_r": far_fit["R2_phi"],
            "far_p_force": far_fit["p_force"],
            "far_R2_force_power": far_fit["R2_force"],
            "scattering_R2_total_vs_sum": R2_addition,
            "scattering_slope_total_vs_sum": Q_add,
            "scattering_mean_frac_error": mean_frac_error,
            "scattering_max_frac_error": max_frac_error,
            "scattering_R2_1_over_b": R2_inv_b,
            "p_scatter_total": p_scatter,
            "usable_scattering": len(usable),
        }
    ]

    save_csv(SUMMARY_CSV, summary_rows)
    save_csv(SCATTER_CSV, scatter_rows)

    field_superposition_pass = phi_rel_error < 1e-8 and force_rel_error < 1e-8
    far_field_pass = far_fit["R2_phi"] > 0.98 and abs(far_fit["p_force"] - 2.0) < 0.20 and far_fit["R2_force"] > 0.98
    scattering_add_pass = R2_addition > 0.98 and mean_frac_error < 0.10
    scattering_inv_b_pass = R2_inv_b > 0.90 and 0.75 <= p_scatter <= 1.35

    print("\n==============================================================")
    print("FINAL TOY MODEL 5a SUMMARY")
    print("==============================================================")
    print("\nFormal:")
    print("    rho_total = rho_1 + rho_2")
    print("    Phi_total = Phi_1 + Phi_2")
    print("    F_total = F_1 + F_2")
    print("    theta_total approximately theta_1 + theta_2")
    print("\nMicrosoft/plain:")
    print("    rho_total = rho_1 + rho_2")
    print("    phi_total = phi_1 + phi_2")
    print("    force_total = force_1 + force_2")
    print("    theta_total ~ theta_1 + theta_2")
    print("\nNormal:")
    print("    This checks whether multiple network defects add like weak gravity.")
    print("\nResults:")
    print("    phi_relative_error =", phi_rel_error)
    print("    force_relative_error =", force_rel_error)
    print("    far-field p_force =", far_fit["p_force"])
    print("    far-field R2_force =", far_fit["R2_force"])
    print("    scattering R2 total-vs-sum =", R2_addition)
    print("    scattering mean fractional addition error =", mean_frac_error)
    print("    scattering p_total =", p_scatter)
    print("    scattering R2 1/b =", R2_inv_b)

    if field_superposition_pass:
        print("\nFIELD SUPERPOSITION: PASS")
    else:
        print("\nFIELD SUPERPOSITION: CHECK")

    if far_field_pass:
        print("FAR-FIELD GRAVITY-LIKE BEHAVIOUR: PASS")
    else:
        print("FAR-FIELD GRAVITY-LIKE BEHAVIOUR: CHECK")

    if scattering_add_pass:
        print("SCATTERING ADDITION: PASS")
    else:
        print("SCATTERING ADDITION: APPROXIMATE / CHECK")

    if scattering_inv_b_pass:
        print("TOTAL 1/b BENDING: PASS")
    else:
        print("TOTAL 1/b BENDING: CHECK")

    if field_superposition_pass and far_field_pass and scattering_inv_b_pass:
        print("\nRESULT: PASS")
        print("Toy Model 5a supports a weak-field multi-source extension:")
        print("network sources -> additive Poisson field -> multi-source bending")
    else:
        print("\nRESULT: MIXED")
        print("The two-source field is useful, but check the diagnostics.")

    print("==============================================================")

    make_plots(x, rho_total, phi_total, far_fit, scatter_rows)


if __name__ == "__main__":
    main()

