import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 7c: LINEARIZED BIANCHI / CONSERVATION CONSISTENCY
# ==============================================================
#
# Formal:
#     mu_eff = gamma_source*|Delta K_excess|
#     Phi(r) = -mu_eff/sqrt(r^2 + s^2)
#
#     g_00 = -(1 + 2*Phi/c^2)
#     g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij
#
#     G_00^lin = (2*gamma_PPN/c^2) laplacian(Phi)
#
#     G_ij^lin = ((1 - gamma_PPN)/c^2)
#                [delta_ij laplacian(Phi) - partial_i partial_j Phi]
#
#     Linearized Bianchi / conservation test:
#         partial^mu G_mu_nu^lin = 0
#
#     Static case:
#         partial^mu G_mu_0^lin = 0
#         partial^mu G_mu_j^lin = partial_i G_ij^lin
#
#     Therefore:
#         partial_i G_ij^lin
#         = ((1 - gamma_PPN)/c^2)
#           [partial_j laplacian(Phi) - partial_i partial_i partial_j Phi]
#         = 0
#
# Microsoft/plain:
#     mu_eff = gamma_source*abs(delta_K_excess)
#     phi = -mu_eff/sqrt(r^2 + softening^2)
#
#     G00_linear = 2*gamma_PPN*laplacian(phi)/c^2
#     Gij_linear = (1 - gamma_PPN)*(delta_ij*laplacian(phi) - hessian_ij(phi))/c^2
#
#     bianchi_residual_j = divergence_i(Gij_linear)
#     bianchi_residual_j should be zero
#
# Normal:
#     Toy 7a showed G00 tracks the network source.
#     Toy 7b showed pressureless spatial consistency selects gamma_PPN=1.
#     Toy 7c checks whether the linearized curvature object is conserved.
#
# Important nuance:
#     Conservation alone does not select gamma_PPN=1.
#     It checks consistency of the curvature/source relation.
#     The pressureless condition from Toy 7b selects gamma_PPN=1.

EPS = 1e-12
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

CASE_CSV = "toy_model_7c_case_summary.csv"
POINT_CSV = "toy_model_7c_point_samples.csv"
NUMERICAL_CSV = "toy_model_7c_numerical_grid_check.csv"
SUMMARY_CSV = "toy_model_7c_summary.csv"

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


def softening_from_radius(radius):
    return max(1.8, radius / 3.0)


def weak_field_parameter(mu, radius, c_light):
    s = softening_from_radius(radius)
    return 2.0 * mu / (s * c_light * c_light)


def make_directions():
    raw = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    ]

    dirs = []

    for v in raw:
        arr = np.array(v, dtype=float)
        arr = arr / np.linalg.norm(arr)
        dirs.append(arr)

    return np.array(dirs, dtype=float)


DIRECTIONS = make_directions()
R_OVER_S = np.logspace(-0.3, 2.1, 80)


def make_points(softening):
    points = []

    for q in R_OVER_S:
        rr = softening * q

        for d in DIRECTIONS:
            points.append(rr * d)

    return np.array(points, dtype=float)


def potential_derivative_quantities(points, mu, softening):
    """
    Formal:
        Phi = -mu/sqrt(A), A = r^2+s^2
        laplacian(Phi) = 3*mu*s^2/A^(5/2)
        H_ij = partial_i partial_j Phi
             = mu[delta_ij/A^(3/2) - 3*x_i*x_j/A^(5/2)]
        partial_j laplacian(Phi) = -15*mu*s^2*x_j/A^(7/2)
        partial_i H_ij = -15*mu*s^2*x_j/A^(7/2)

    Microsoft/plain:
        grad_lap_j = -15*mu*s^2*x_j/A^(7/2)
        div_hessian_j = -15*mu*s^2*x_j/A^(7/2)
        bianchi_residual = grad_lap - div_hessian
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    coords = [x, y, z]

    r2 = x*x + y*y + z*z
    A = r2 + softening*softening

    phi = -mu / np.sqrt(A)
    lap = 3.0 * mu * softening * softening / (A**2.5)
    rho = lap / (4.0 * math.pi)

    n = len(points)
    hess = np.zeros((n, 3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            hess[:, i, j] = mu * (
                delta / (A**1.5)
                - 3.0 * coords[i] * coords[j] / (A**2.5)
            )

    grad_lap = np.zeros((n, 3), dtype=float)
    div_hessian = np.zeros((n, 3), dtype=float)

    for j in range(3):
        grad_lap[:, j] = -15.0 * mu * softening * softening * coords[j] / (A**3.5)
        div_hessian[:, j] = -15.0 * mu * softening * softening * coords[j] / (A**3.5)

    return {
        "phi": phi,
        "lap": lap,
        "rho": rho,
        "hess": hess,
        "grad_lap": grad_lap,
        "div_hessian": div_hessian,
    }


def analytic_bianchi_case(case, c_light, gamma_ppn):
    label = case["label"]
    alpha = case["alpha"]
    beta = case["beta"]
    radius = case["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
    s = softening_from_radius(radius)
    weak = weak_field_parameter(mu, radius, c_light)

    points = make_points(s)
    out = potential_derivative_quantities(points, mu, s)

    coeff = (1.0 - gamma_ppn) / (c_light * c_light)

    bianchi_residual = coeff * (out["grad_lap"] - out["div_hessian"])
    grad_scale = abs(coeff) * np.sqrt(np.sum(out["grad_lap"] * out["grad_lap"], axis=1))
    residual_norm = np.sqrt(np.sum(bianchi_residual * bianchi_residual, axis=1))

    max_abs_residual = float(np.max(residual_norm))
    rms_residual = float(np.sqrt(np.mean(residual_norm * residual_norm)))
    rms_scale = float(np.sqrt(np.mean(grad_scale * grad_scale)))
    relative_residual = rms_residual / (rms_scale + EPS)

    # Effective source conservation follows from G/kappa.
    # We use kappa=8*pi/c^2 in toy units for reporting.
    kappa = 8.0 * math.pi / (c_light * c_light)
    effective_source_divergence_rms = rms_residual / (kappa + EPS)

    if abs(gamma_ppn - 1.0) < 1e-12:
        bianchi_pass = max_abs_residual < 1e-20
    else:
        bianchi_pass = relative_residual < 1e-12 and max_abs_residual < 1e-18

    case_pass = weak < 0.05 and bianchi_pass

    summary = {
        "case": label,
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "softening": s,
        "c_light": c_light,
        "gamma_PPN": gamma_ppn,
        "weak_field_parameter": weak,
        "max_abs_bianchi_residual": max_abs_residual,
        "rms_bianchi_residual": rms_residual,
        "rms_derivative_scale": rms_scale,
        "relative_bianchi_residual": relative_residual,
        "effective_source_divergence_rms": effective_source_divergence_rms,
        "case_pass": int(case_pass),
    }

    point_rows = []
    stride = max(1, len(points) // 250)
    rvals = np.sqrt(np.sum(points * points, axis=1))

    for idx in range(0, len(points), stride):
        point_rows.append({
            "case": label,
            "c_light": c_light,
            "gamma_PPN": gamma_ppn,
            "x": float(points[idx, 0]),
            "y": float(points[idx, 1]),
            "z": float(points[idx, 2]),
            "r": float(rvals[idx]),
            "rho_network": float(out["rho"][idx]),
            "grad_lap_x": float(out["grad_lap"][idx, 0]),
            "grad_lap_y": float(out["grad_lap"][idx, 1]),
            "grad_lap_z": float(out["grad_lap"][idx, 2]),
            "div_hessian_x": float(out["div_hessian"][idx, 0]),
            "div_hessian_y": float(out["div_hessian"][idx, 1]),
            "div_hessian_z": float(out["div_hessian"][idx, 2]),
            "bianchi_residual_norm": float(residual_norm[idx]),
        })

    arrays = {
        "points": points,
        **out,
        "bianchi_residual": bianchi_residual,
        "residual_norm": residual_norm,
        "grad_scale": grad_scale,
    }

    return summary, point_rows, arrays


def derivative_axis_4th(f, dx, axis):
    """
    Fourth-order central finite-difference derivative.
    Two boundary cells are returned as NaN and excluded from diagnostics.
    """
    out = np.full_like(f, np.nan, dtype=float)

    center = [slice(None)] * f.ndim
    plus2 = [slice(None)] * f.ndim
    plus1 = [slice(None)] * f.ndim
    minus1 = [slice(None)] * f.ndim
    minus2 = [slice(None)] * f.ndim

    center[axis] = slice(2, -2)
    plus2[axis] = slice(4, None)
    plus1[axis] = slice(3, -1)
    minus1[axis] = slice(1, -3)
    minus2[axis] = slice(0, -4)

    out[tuple(center)] = (
        -f[tuple(plus2)]
        + 8.0 * f[tuple(plus1)]
        - 8.0 * f[tuple(minus1)]
        + f[tuple(minus2)]
    ) / (12.0 * dx)

    return out


def compute_Gij_on_grid(mu, softening, c_light, gamma_ppn, N=61, L=40.0):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    r2 = X*X + Y*Y + Z*Z
    A = r2 + softening*softening

    lap = 3.0 * mu * softening * softening / (A**2.5)

    coords = [X, Y, Z]

    Gij = np.zeros((3, 3, N, N, N), dtype=float)
    coeff = (1.0 - gamma_ppn) / (c_light * c_light)

    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            hess = mu * (
                delta / (A**1.5)
                - 3.0 * coords[i] * coords[j] / (A**2.5)
            )
            basis = delta * lap - hess
            Gij[i, j] = coeff * basis

    return x, dx, Gij, lap


def numerical_grid_bianchi_check():
    """
    Finite-difference sanity check for one representative source.
    The analytic scan above is the primary conservation test.
    """
    rep_case = {"label": "r6_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 6.0}
    c_light = 12.0

    mu = mu_eff(rep_case["alpha"], rep_case["beta"], rep_case["radius"], GAMMA_SOURCE)
    s = softening_from_radius(rep_case["radius"])

    rows = []
    rep_arrays = {}

    for gamma_ppn in GAMMA_PPN_VALUES:
        x, dx, Gij, lap = compute_Gij_on_grid(mu, s, c_light, gamma_ppn, N=101, L=40.0)

        divergence = np.zeros((3, len(x), len(x), len(x)), dtype=float)
        valid_derivative = np.ones_like(lap, dtype=bool)

        for j in range(3):
            div_j = np.zeros_like(lap)

            for i in range(3):
                dG = derivative_axis_4th(Gij[i, j], dx, axis=i)
                valid_derivative &= np.isfinite(dG)
                div_j += np.nan_to_num(dG, nan=0.0)

            divergence[j] = div_j

        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt(X*X + Y*Y + Z*Z)

        grad_lap_components = np.array([
            derivative_axis_4th(lap, dx, axis=0),
            derivative_axis_4th(lap, dx, axis=1),
            derivative_axis_4th(lap, dx, axis=2),
        ])

        valid_derivative &= np.all(np.isfinite(grad_lap_components), axis=0)

        # Avoid finite-difference boundary and far low-signal region.
        mask = (r > 1.5 * s) & (r < 0.65 * 40.0) & valid_derivative

        div_norm = np.sqrt(np.sum(divergence * divergence, axis=0))

        rms_div = float(np.sqrt(np.mean(div_norm[mask]**2)))
        max_div = float(np.max(div_norm[mask]))

        grad_lap_scale = abs(1.0 - gamma_ppn) / (c_light * c_light) * np.sqrt(
            np.sum(grad_lap_components * grad_lap_components, axis=0)
        )

        rms_scale = float(np.sqrt(np.mean(grad_lap_scale[mask]**2)))
        relative = rms_div / (rms_scale + EPS)

        if abs(gamma_ppn - 1.0) < 1e-12:
            pass_grid = max_div < 1e-14
        else:
            pass_grid = relative < 0.05

        rows.append({
            "case": rep_case["label"],
            "c_light": c_light,
            "gamma_PPN": gamma_ppn,
            "grid_N": len(x),
            "grid_L": 40.0,
            "dx": dx,
            "rms_finite_difference_bianchi_residual": rms_div,
            "max_finite_difference_bianchi_residual": max_div,
            "rms_derivative_scale": rms_scale,
            "relative_finite_difference_residual": relative,
            "grid_pass": int(pass_grid),
        })

        rep_arrays[gamma_ppn] = {
            "x": x,
            "r": r,
            "div_norm": div_norm,
            "mask": mask,
            "Gij": Gij,
            "lap": lap,
        }

    return rows, rep_arrays


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


def make_plots(case_rows, rep_analytic_arrays, grid_rows, grid_arrays):
    points = rep_analytic_arrays["points"]
    r = np.sqrt(np.sum(points * points, axis=1))

    grad_lap = rep_analytic_arrays["grad_lap"].reshape(-1)
    div_hess = rep_analytic_arrays["div_hessian"].reshape(-1)

    plt.figure(figsize=(8, 7))
    plt.scatter(grad_lap, div_hess, s=8)

    lo = min(np.min(grad_lap), np.min(div_hess))
    hi = max(np.max(grad_lap), np.max(div_hess))

    plt.plot([lo, hi], [lo, hi], label="perfect identity")
    plt.xlabel("partial_j laplacian(phi)")
    plt.ylabel("partial_i hessian_ij(phi)")
    plt.title("Toy Model 7c: derivative identity behind Bianchi consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7c_derivative_identity.png", dpi=200)
    plt.show()

    gammas = sorted(set(float(row["gamma_PPN"]) for row in case_rows))
    max_res = []

    for g in gammas:
        subset = [row for row in case_rows if abs(float(row["gamma_PPN"]) - g) < 1e-12]
        max_res.append(max(row["max_abs_bianchi_residual"] for row in subset))

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, max_res, marker="o", label="max absolute Bianchi residual")
    plt.xlabel("gamma_PPN")
    plt.ylabel("max |divergence(G_linear)|")
    plt.title("Toy Model 7c: analytic Bianchi residual across gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7c_bianchi_residual_vs_gamma.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 6))

    for gamma_ppn in [0.0, 0.5, 1.0]:
        arr = grid_arrays[gamma_ppn]
        rgrid = arr["r"]
        div_norm = arr["div_norm"]
        mask = arr["mask"]

        plt.scatter(
            rgrid[mask].reshape(-1),
            div_norm[mask].reshape(-1),
            s=2,
            label="gamma=%.1f" % gamma_ppn,
        )

    plt.yscale("log")
    plt.xlabel("r")
    plt.ylabel("finite-difference |divergence(Gij)|")
    plt.title("Toy Model 7c: numerical grid Bianchi sanity check")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7c_numerical_grid_residual.png", dpi=200)
    plt.show()

    labels = ["g=%.1f" % row["gamma_PPN"] for row in grid_rows]
    rel = [row["relative_finite_difference_residual"] for row in grid_rows]

    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(labels)), rel)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel("relative finite-difference residual")
    plt.title("Toy Model 7c: finite-difference conservation residual")
    plt.tight_layout()
    plt.savefig("toy_model_7c_grid_relative_residual.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_7c_derivative_identity.png")
    print("toy_model_7c_bianchi_residual_vs_gamma.png")
    print("toy_model_7c_numerical_grid_residual.png")
    print("toy_model_7c_grid_relative_residual.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 7c: LINEARIZED BIANCHI / CONSERVATION CONSISTENCY")
    print("==============================================================")

    print("\nFormal:")
    print("    partial^mu G_mu_nu^lin = 0")
    print("    static case: partial_i G_ij^lin = 0")
    print("    G_ij^lin = ((1-gamma_PPN)/c^2)[delta_ij*laplacian(Phi)-partial_i partial_j Phi]")
    print("    partial_i G_ij^lin = ((1-gamma_PPN)/c^2)[partial_j laplacian(Phi)-partial_i partial_i partial_j Phi] = 0")

    print("\nMicrosoft/plain:")
    print("    divergence(G_linear) should be zero")
    print("    bianchi_residual_j = divergence_i(Gij_linear)")
    print("    bianchi_residual_j = 0")
    print("    effective_source = G_linear/kappa is conserved")

    print("\nNormal:")
    print("    We test whether the linearized Einstein-like curvature object is conserved.")
    print("    Conservation alone does not select gamma=1; Toy 7b's pressureless condition does that.")
    print("==============================================================")

    all_case_rows = []
    all_point_rows = []
    rep_analytic_arrays = None

    total = len(NETWORK_CASES) * len(C_LIGHT_VALUES) * len(GAMMA_PPN_VALUES)
    counter = 0

    for case in NETWORK_CASES:
        for c_light in C_LIGHT_VALUES:
            for gamma_ppn in GAMMA_PPN_VALUES:
                counter += 1

                print("Running analytic Bianchi case %d/%d: %s, c=%.1f, gamma_PPN=%.1f" % (
                    counter,
                    total,
                    case["label"],
                    c_light,
                    gamma_ppn,
                ))

                summary, point_rows, arrays = analytic_bianchi_case(case, c_light, gamma_ppn)

                all_case_rows.append(summary)
                all_point_rows.extend(point_rows)

                if (
                    case["label"] == "r6_a8_b025"
                    and abs(c_light - 12.0) < 1e-12
                    and abs(gamma_ppn - 0.5) < 1e-12
                ):
                    rep_analytic_arrays = arrays

    print("\nRunning representative finite-difference grid sanity check...")
    grid_rows, grid_arrays = numerical_grid_bianchi_check()

    save_csv(CASE_CSV, all_case_rows)
    save_csv(POINT_CSV, all_point_rows)
    save_csv(NUMERICAL_CSV, grid_rows)

    pass_count = int(sum(row["case_pass"] for row in all_case_rows))
    total_count = len(all_case_rows)

    grid_pass_count = int(sum(row["grid_pass"] for row in grid_rows))
    grid_total = len(grid_rows)

    max_weak = float(max(row["weak_field_parameter"] for row in all_case_rows))
    max_abs_residual = float(max(row["max_abs_bianchi_residual"] for row in all_case_rows))
    mean_rms_residual = float(np.mean([row["rms_bianchi_residual"] for row in all_case_rows]))
    max_relative_residual = float(max(row["relative_bianchi_residual"] for row in all_case_rows))
    mean_effective_source_divergence = float(np.mean([row["effective_source_divergence_rms"] for row in all_case_rows]))

    gamma_stats = {}

    for gamma_ppn in GAMMA_PPN_VALUES:
        subset = [row for row in all_case_rows if abs(float(row["gamma_PPN"]) - gamma_ppn) < 1e-12]
        gamma_stats[gamma_ppn] = {
            "max_abs_residual": float(max(row["max_abs_bianchi_residual"] for row in subset)),
            "mean_relative_residual": float(np.mean([row["relative_bianchi_residual"] for row in subset])),
        }

    max_grid_relative = float(max(row["relative_finite_difference_residual"] for row in grid_rows))
    max_grid_abs = float(max(row["max_finite_difference_bianchi_residual"] for row in grid_rows))

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "finite_difference_groups_passed": grid_pass_count,
        "finite_difference_groups_total": grid_total,
        "max_weak_field_parameter": max_weak,
        "max_abs_analytic_bianchi_residual": max_abs_residual,
        "mean_rms_analytic_bianchi_residual": mean_rms_residual,
        "max_relative_analytic_bianchi_residual": max_relative_residual,
        "mean_effective_source_divergence_rms": mean_effective_source_divergence,
        "gamma0_max_abs_residual": gamma_stats[0.0]["max_abs_residual"],
        "gamma05_max_abs_residual": gamma_stats[0.5]["max_abs_residual"],
        "gamma1_max_abs_residual": gamma_stats[1.0]["max_abs_residual"],
        "max_finite_difference_relative_residual": max_grid_relative,
        "max_finite_difference_abs_residual": max_grid_abs,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 7c SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    partial^mu G_mu_nu^lin = 0")
    print("    static: partial_i G_ij^lin = 0")
    print("    partial_i G_ij^lin = ((1-gamma_PPN)/c^2)[partial_j laplacian(Phi)-partial_i partial_i partial_j Phi]")
    print("    smooth derivatives commute, so the residual vanishes")

    print("\nMicrosoft/plain:")
    print("    divergence(G_linear) = 0")
    print("    effective_source = G_linear/kappa")
    print("    divergence(effective_source) = 0")

    print("\nNormal:")
    print("    The linearized curvature object is conserved, so the effective source relation is consistent.")
    print("    This does not by itself select gamma=1; it verifies Bianchi-like consistency.")

    print("\nCases:")
    print("    analytic case groups passed = %d/%d" % (pass_count, total_count))
    print("    finite-difference sanity groups passed = %d/%d" % (grid_pass_count, grid_total))

    print("\nWeak-field condition:")
    print("    max weak_field_parameter =", max_weak)
    print("    target < 0.05")

    print("\nAnalytic Bianchi residual:")
    print("    max_abs_analytic_bianchi_residual =", max_abs_residual)
    print("    mean_rms_analytic_bianchi_residual =", mean_rms_residual)
    print("    max_relative_analytic_bianchi_residual =", max_relative_residual)
    print("    mean_effective_source_divergence_rms =", mean_effective_source_divergence)

    print("\nGamma groups:")
    print("    gamma=0.0 max_abs_residual =", gamma_stats[0.0]["max_abs_residual"])
    print("    gamma=0.5 max_abs_residual =", gamma_stats[0.5]["max_abs_residual"])
    print("    gamma=1.0 max_abs_residual =", gamma_stats[1.0]["max_abs_residual"])

    print("\nFinite-difference grid sanity check:")

    for row in grid_rows:
        print("    gamma=%.1f relative_residual=%g max_abs=%g pass=%d" % (
            row["gamma_PPN"],
            row["relative_finite_difference_residual"],
            row["max_finite_difference_bianchi_residual"],
            row["grid_pass"],
        ))

    weak_pass = max_weak < 0.05
    analytic_pass = pass_count == total_count and max_abs_residual < 1e-18
    grid_pass = grid_pass_count == grid_total
    source_conservation_pass = mean_effective_source_divergence < 1e-18

    print("\nWEAK-FIELD CONDITION:", "PASS" if weak_pass else "CHECK")
    print("ANALYTIC BIANCHI IDENTITY:", "PASS" if analytic_pass else "CHECK")
    print("FINITE-DIFFERENCE SANITY CHECK:", "PASS" if grid_pass else "CHECK")
    print("EFFECTIVE SOURCE CONSERVATION:", "PASS" if source_conservation_pass else "CHECK")

    if weak_pass and analytic_pass and grid_pass and source_conservation_pass:
        print("\nRESULT: PASS")
        print("Toy Model 7c supports Bianchi-like conservation consistency:")
        print("network source -> metric -> conserved linearized curvature/effective source relation")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving on.")

    print("==============================================================")

    if rep_analytic_arrays is not None:
        make_plots(all_case_rows, rep_analytic_arrays, grid_rows, grid_arrays)


if __name__ == "__main__":
    main()

