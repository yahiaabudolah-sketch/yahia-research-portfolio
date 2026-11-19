import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 7b: LINEARIZED EINSTEIN-LIKE SPATIAL COMPONENTS
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
#     For pressureless weak-field matter, T_ij approx 0.
#     Therefore a GR-like weak-field metric should have:
#         gamma_PPN = 1
#         G_ij^lin = 0
#
# Microsoft/plain:
#     mu_eff = gamma_source*abs(delta_K_excess)
#     phi = -mu_eff/sqrt(r^2 + softening^2)
#
#     g00 = -(1 + 2*phi/c^2)
#     gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij
#
#     G00_linear = 2*gamma_PPN*laplacian(phi)/c^2
#
#     Gij_linear = (1 - gamma_PPN)*(delta_ij*laplacian(phi) - hessian_ij(phi))/c^2
#
#     gamma_PPN = 1 makes Gij_linear vanish for pressureless weak-field matter
#
# Normal:
#     Toy Model 7a tested the 00 source-curvature relation.
#     Toy Model 7b tests the spatial components of the same linearized metric.
#     The key question is whether spatial curvature is required to remove
#     unphysical spatial stress/curvature for a pressureless source.
#
# This is NOT full GR.
# It is a linearized, weak-field, static, spherical-source component test.

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

PROFILE_CSV = "toy_model_7b_component_profiles.csv"
SUMMARY_CSV = "toy_model_7b_summary.csv"
CASE_CSV = "toy_model_7b_case_summary.csv"

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
    return 2.0*mu/(s*c_light*c_light)


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
R_OVER_S = np.logspace(-0.35, 2.05, 75)


def make_points(softening):
    points = []

    for q in R_OVER_S:
        rr = softening*q
        for d in DIRECTIONS:
            points.append(rr*d)

    return np.array(points, dtype=float)


def potential_laplacian_and_hessian(points, mu, softening):
    """
    Formal:
        Phi = -mu/sqrt(r^2+s^2)
        partial_i partial_j Phi = mu[delta_ij/A^(3/2) - 3*x_i*x_j/A^(5/2)]
        laplacian(Phi) = 3*mu*s^2/A^(5/2)

    Microsoft/plain:
        hessian_ij = mu*(delta_ij/A^(3/2) - 3*x_i*x_j/A^(5/2))
        laplacian_phi = 3*mu*s^2/A^(5/2)
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r2 = x*x + y*y + z*z
    A = r2 + softening*softening

    phi = -mu / np.sqrt(A)
    lap = 3.0*mu*softening*softening/(A**2.5)
    rho = lap/(4.0*math.pi)

    n = len(points)
    hess = np.zeros((n, 3, 3), dtype=float)

    coords = [x, y, z]

    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            hess[:, i, j] = mu*(delta/(A**1.5) - 3.0*coords[i]*coords[j]/(A**2.5))

    return phi, lap, rho, hess


def linearized_components(points, mu, softening, c_light, gamma_ppn):
    phi, lap, rho, hess = potential_laplacian_and_hessian(points, mu, softening)

    n = len(points)
    delta = np.eye(3)

    basis = np.zeros((n, 3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            basis[:, i, j] = (delta[i, j]*lap - hess[:, i, j])/(c_light*c_light)

    Gij = (1.0 - gamma_ppn)*basis
    G00 = 2.0*gamma_ppn*lap/(c_light*c_light)
    G00_source = 8.0*math.pi*gamma_ppn*rho/(c_light*c_light)
    G00_GR_target = 8.0*math.pi*rho/(c_light*c_light)

    trace_Gij = Gij[:, 0, 0] + Gij[:, 1, 1] + Gij[:, 2, 2]
    trace_basis = 2.0*lap/(c_light*c_light)

    spatial_norm = np.sqrt(np.sum(Gij*Gij, axis=(1, 2)))
    basis_norm = np.sqrt(np.sum(basis*basis, axis=(1, 2)))

    return {
        "phi": phi,
        "lap": lap,
        "rho": rho,
        "hess": hess,
        "basis": basis,
        "Gij": Gij,
        "G00": G00,
        "G00_source": G00_source,
        "G00_GR_target": G00_GR_target,
        "trace_Gij": trace_Gij,
        "trace_basis": trace_basis,
        "spatial_norm": spatial_norm,
        "basis_norm": basis_norm,
    }


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]

    if len(y) < 2:
        return np.nan

    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot <= EPS:
        if np.sqrt(np.mean((y - pred)**2)) < 1e-14:
            return 1.0
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


def run_case(case, c_light, gamma_ppn):
    label = case["label"]
    alpha = case["alpha"]
    beta = case["beta"]
    radius = case["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
    s = softening_from_radius(radius)
    weak = weak_field_parameter(mu, radius, c_light)

    points = make_points(s)
    out = linearized_components(points, mu, s, c_light, gamma_ppn)

    G00_q, _, G00_R2 = fit_origin(out["G00_source"], out["G00"])
    G00_GR_q, _, G00_GR_R2 = fit_origin(out["G00_GR_target"], out["G00"])

    component_x = out["basis"].reshape(-1)
    component_y = out["Gij"].reshape(-1)

    component_q, _, component_R2 = fit_origin(component_x, component_y)

    trace_q, _, trace_R2 = fit_origin(out["trace_basis"], out["trace_Gij"])
    norm_q, _, norm_R2 = fit_origin(out["basis_norm"], out["spatial_norm"])

    expected_spatial_coeff = 1.0 - gamma_ppn
    expected_norm_coeff = abs(1.0 - gamma_ppn)

    max_abs_Gij = float(np.max(np.abs(out["Gij"])))
    mean_abs_Gij = float(np.mean(np.abs(out["Gij"])))
    mean_abs_G00_GR_target = float(np.mean(np.abs(out["G00_GR_target"])))
    spatial_to_GR_source_ratio = mean_abs_Gij/(mean_abs_G00_GR_target + EPS)

    pressureless_residual_pass = True

    if abs(gamma_ppn - 1.0) < 1e-12:
        pressureless_residual_pass = max_abs_Gij < 1e-14

    G00_pass = True

    if gamma_ppn > 0:
        G00_pass = (
            abs(G00_q - 1.0) < 1e-12
            and G00_R2 > 0.999999
        )
    else:
        G00_pass = abs(G00_GR_q) < 1e-14

    spatial_coeff_pass = (
        abs(component_q - expected_spatial_coeff) < 1e-12
        and abs(trace_q - expected_spatial_coeff) < 1e-12
        and abs(norm_q - expected_norm_coeff) < 1e-12
    )

    gamma_GR_pass = abs(G00_GR_q - gamma_ppn) < 1e-12

    case_pass = (
        weak < 0.05
        and G00_pass
        and spatial_coeff_pass
        and gamma_GR_pass
        and pressureless_residual_pass
    )

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
        "G00_metric_vs_source_Q": G00_q,
        "G00_metric_vs_source_R2": G00_R2,
        "G00_vs_GR_target_Q": G00_GR_q,
        "G00_vs_GR_target_R2": G00_GR_R2,
        "Gij_component_Q": component_q,
        "Gij_component_R2": component_R2,
        "Gij_trace_Q": trace_q,
        "Gij_trace_R2": trace_R2,
        "Gij_norm_Q": norm_q,
        "Gij_norm_R2": norm_R2,
        "expected_spatial_coeff": expected_spatial_coeff,
        "expected_norm_coeff": expected_norm_coeff,
        "max_abs_Gij": max_abs_Gij,
        "mean_abs_Gij": mean_abs_Gij,
        "mean_abs_G00_GR_target": mean_abs_G00_GR_target,
        "spatial_to_GR_source_ratio": spatial_to_GR_source_ratio,
        "pressureless_residual_pass": int(pressureless_residual_pass),
        "case_pass": int(case_pass),
    }

    profile_rows = []

    rvals = np.sqrt(np.sum(points*points, axis=1))
    stride = max(1, len(points)//300)

    for idx in range(0, len(points), stride):
        profile_rows.append({
            "case": label,
            "c_light": c_light,
            "gamma_PPN": gamma_ppn,
            "x": float(points[idx, 0]),
            "y": float(points[idx, 1]),
            "z": float(points[idx, 2]),
            "r": float(rvals[idx]),
            "rho_network": float(out["rho"][idx]),
            "laplacian_phi": float(out["lap"][idx]),
            "G00_metric": float(out["G00"][idx]),
            "G00_source": float(out["G00_source"][idx]),
            "Gxx": float(out["Gij"][idx, 0, 0]),
            "Gyy": float(out["Gij"][idx, 1, 1]),
            "Gzz": float(out["Gij"][idx, 2, 2]),
            "Gxy": float(out["Gij"][idx, 0, 1]),
            "Gxz": float(out["Gij"][idx, 0, 2]),
            "Gyz": float(out["Gij"][idx, 1, 2]),
            "Gij_norm": float(out["spatial_norm"][idx]),
            "Gij_basis_norm": float(out["basis_norm"][idx]),
            "trace_Gij": float(out["trace_Gij"][idx]),
            "trace_basis": float(out["trace_basis"][idx]),
        })

    arrays = {
        "points": points,
        **out,
    }

    return summary, profile_rows, arrays


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


def make_plots(case_rows, rep_gamma05, rep_gamma1):
    gammas = sorted(set(float(r["gamma_PPN"]) for r in case_rows))

    measured_component = []
    measured_norm = []
    measured_trace = []
    expected = []

    for g in gammas:
        subset = [r for r in case_rows if abs(float(r["gamma_PPN"]) - g) < 1e-12]
        measured_component.append(np.mean([r["Gij_component_Q"] for r in subset]))
        measured_norm.append(np.mean([r["Gij_norm_Q"] for r in subset]))
        measured_trace.append(np.mean([r["Gij_trace_Q"] for r in subset]))
        expected.append(1.0 - g)

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, measured_component, marker="o", label="component coefficient")
    plt.plot(gammas, measured_norm, marker="o", label="norm coefficient")
    plt.plot(gammas, measured_trace, marker="o", label="trace coefficient")
    plt.plot(gammas, expected, linestyle="--", label="expected = 1 - gamma_PPN")
    plt.xlabel("gamma_PPN")
    plt.ylabel("spatial Gij coefficient")
    plt.title("Toy Model 7b: spatial components vanish only at gamma_PPN=1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7b_spatial_coefficients_vs_gamma.png", dpi=200)
    plt.show()

    bx = rep_gamma05["basis"].reshape(-1)
    gy = rep_gamma05["Gij"].reshape(-1)

    plt.figure(figsize=(8, 7))
    plt.scatter(bx, gy, s=8)
    lo = min(np.min(bx), np.min(gy))
    hi = max(np.max(bx), np.max(gy))
    xx = np.linspace(lo, hi, 200)
    plt.plot(xx, 0.5*xx, label="expected gamma=0.5: Gij = 0.5*basis")
    plt.xlabel("basis = (delta_ij*laplacian(phi) - hessian_ij(phi))/c^2")
    plt.ylabel("Gij_linear")
    plt.title("Toy Model 7b: spatial component fit at gamma_PPN=0.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7b_Gij_component_fit_gamma05.png", dpi=200)
    plt.show()

    points05 = rep_gamma05["points"]
    r05 = np.sqrt(np.sum(points05*points05, axis=1))
    order05 = np.argsort(r05)

    points1 = rep_gamma1["points"]
    r1 = np.sqrt(np.sum(points1*points1, axis=1))
    order1 = np.argsort(r1)

    plt.figure(figsize=(9, 6))
    plt.loglog(r05[order05], rep_gamma05["spatial_norm"][order05], label="gamma=0.5 spatial Gij norm")
    zero_floor = np.full_like(r1[order1], 1e-18)
    plt.loglog(r1[order1], rep_gamma1["spatial_norm"][order1] + zero_floor, label="gamma=1 spatial Gij norm + tiny floor")
    plt.xlabel("r")
    plt.ylabel("spatial Gij norm")
    plt.title("Toy Model 7b: GR-like gamma removes pressureless spatial residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7b_pressureless_spatial_residual.png", dpi=200)
    plt.show()

    trace_x = rep_gamma05["trace_basis"]
    trace_y = rep_gamma05["trace_Gij"]

    plt.figure(figsize=(8, 7))
    plt.scatter(trace_x, trace_y, s=10)
    lo = min(np.min(trace_x), np.min(trace_y))
    hi = max(np.max(trace_x), np.max(trace_y))
    xx = np.linspace(lo, hi, 200)
    plt.plot(xx, 0.5*xx, label="expected gamma=0.5")
    plt.xlabel("trace basis = 2*laplacian(phi)/c^2")
    plt.ylabel("trace Gij")
    plt.title("Toy Model 7b: spatial trace relation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7b_spatial_trace_relation.png", dpi=200)
    plt.show()

    gammas2 = sorted(set(float(r["gamma_PPN"]) for r in case_rows))
    G00_coeff = []

    for g in gammas2:
        subset = [r for r in case_rows if abs(float(r["gamma_PPN"]) - g) < 1e-12]
        G00_coeff.append(np.mean([r["G00_vs_GR_target_Q"] for r in subset]))

    plt.figure(figsize=(8, 6))
    plt.plot(gammas2, G00_coeff, marker="o", label="G00 / GR-target coefficient")
    plt.plot(gammas2, gammas2, linestyle="--", label="expected = gamma_PPN")
    plt.xlabel("gamma_PPN")
    plt.ylabel("G00 coefficient")
    plt.title("Toy Model 7b: time and spatial components are complementary")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7b_G00_gamma_coefficient.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_7b_spatial_coefficients_vs_gamma.png")
    print("toy_model_7b_Gij_component_fit_gamma05.png")
    print("toy_model_7b_pressureless_spatial_residual.png")
    print("toy_model_7b_spatial_trace_relation.png")
    print("toy_model_7b_G00_gamma_coefficient.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 7b: LINEARIZED EINSTEIN-LIKE SPATIAL COMPONENTS")
    print("==============================================================")

    print("\nFormal:")
    print("    G_00^lin = (2*gamma_PPN/c^2)*laplacian(Phi)")
    print("    G_ij^lin = ((1-gamma_PPN)/c^2)[delta_ij*laplacian(Phi) - partial_i partial_j Phi]")
    print("    pressureless weak-field matter wants G_ij^lin approx 0")
    print("    therefore the GR-like case is gamma_PPN = 1")

    print("\nMicrosoft/plain:")
    print("    G00_linear = 2*gamma_PPN*laplacian(phi)/c^2")
    print("    Gij_linear = (1-gamma_PPN)*(delta_ij*laplacian(phi)-hessian_ij(phi))/c^2")
    print("    pressureless_source means target_Gij = 0")
    print("    gamma_PPN=1 makes Gij_linear vanish")

    print("\nNormal:")
    print("    We test the spatial parts of the same weak-field metric.")
    print("    This checks whether spatial curvature is required for a pressureless GR-like structure.")
    print("==============================================================")

    all_case_rows = []
    all_profile_rows = []
    rep_gamma05 = None
    rep_gamma1 = None

    total = len(NETWORK_CASES)*len(C_LIGHT_VALUES)*len(GAMMA_PPN_VALUES)
    counter = 0

    for case in NETWORK_CASES:
        for c_light in C_LIGHT_VALUES:
            for gamma_ppn in GAMMA_PPN_VALUES:
                counter += 1
                print("Running case %d/%d: %s, c=%.1f, gamma_PPN=%.1f" % (
                    counter, total, case["label"], c_light, gamma_ppn
                ))

                summary, profile_rows, arrays = run_case(case, c_light, gamma_ppn)

                all_case_rows.append(summary)
                all_profile_rows.extend(profile_rows)

                if case["label"] == "r6_a8_b025" and abs(c_light - 12.0) < 1e-12:
                    if abs(gamma_ppn - 0.5) < 1e-12:
                        rep_gamma05 = arrays
                    if abs(gamma_ppn - 1.0) < 1e-12:
                        rep_gamma1 = arrays

    save_csv(CASE_CSV, all_case_rows)
    save_csv(PROFILE_CSV, all_profile_rows)

    pass_count = int(sum(row["case_pass"] for row in all_case_rows))
    total_count = len(all_case_rows)
    max_weak = float(max(row["weak_field_parameter"] for row in all_case_rows))

    mean_G00_Q_positive_gamma = float(np.mean([
        row["G00_metric_vs_source_Q"]
        for row in all_case_rows
        if float(row["gamma_PPN"]) > 0
    ]))

    mean_G00_R2_positive_gamma = float(np.mean([
        row["G00_metric_vs_source_R2"]
        for row in all_case_rows
        if float(row["gamma_PPN"]) > 0
    ]))

    gamma_groups = {}
    for gamma in GAMMA_PPN_VALUES:
        subset = [row for row in all_case_rows if abs(float(row["gamma_PPN"]) - gamma) < 1e-12]
        gamma_groups[gamma] = {
            "mean_Gij_component_Q": float(np.mean([row["Gij_component_Q"] for row in subset])),
            "mean_Gij_trace_Q": float(np.mean([row["Gij_trace_Q"] for row in subset])),
            "mean_Gij_norm_Q": float(np.mean([row["Gij_norm_Q"] for row in subset])),
            "mean_G00_GR_target_Q": float(np.mean([row["G00_vs_GR_target_Q"] for row in subset])),
            "mean_spatial_to_GR_source_ratio": float(np.mean([row["spatial_to_GR_source_ratio"] for row in subset])),
            "max_abs_Gij": float(np.max([row["max_abs_Gij"] for row in subset])),
        }

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "max_weak_field_parameter": max_weak,
        "mean_G00_metric_vs_source_Q_positive_gamma": mean_G00_Q_positive_gamma,
        "mean_G00_metric_vs_source_R2_positive_gamma": mean_G00_R2_positive_gamma,
        "gamma0_mean_Gij_component_Q": gamma_groups[0.0]["mean_Gij_component_Q"],
        "gamma05_mean_Gij_component_Q": gamma_groups[0.5]["mean_Gij_component_Q"],
        "gamma1_mean_Gij_component_Q": gamma_groups[1.0]["mean_Gij_component_Q"],
        "gamma0_mean_Gij_trace_Q": gamma_groups[0.0]["mean_Gij_trace_Q"],
        "gamma05_mean_Gij_trace_Q": gamma_groups[0.5]["mean_Gij_trace_Q"],
        "gamma1_mean_Gij_trace_Q": gamma_groups[1.0]["mean_Gij_trace_Q"],
        "gamma0_mean_Gij_norm_Q": gamma_groups[0.0]["mean_Gij_norm_Q"],
        "gamma05_mean_Gij_norm_Q": gamma_groups[0.5]["mean_Gij_norm_Q"],
        "gamma1_mean_Gij_norm_Q": gamma_groups[1.0]["mean_Gij_norm_Q"],
        "gamma0_mean_G00_GR_target_Q": gamma_groups[0.0]["mean_G00_GR_target_Q"],
        "gamma05_mean_G00_GR_target_Q": gamma_groups[0.5]["mean_G00_GR_target_Q"],
        "gamma1_mean_G00_GR_target_Q": gamma_groups[1.0]["mean_G00_GR_target_Q"],
        "gamma1_max_abs_Gij": gamma_groups[1.0]["max_abs_Gij"],
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 7b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    G_ij^lin = ((1-gamma_PPN)/c^2)[delta_ij*laplacian(Phi) - partial_i partial_j Phi]")
    print("    For pressureless matter, target G_ij^lin = 0")
    print("    Therefore gamma_PPN=1 is selected by spatial-component consistency")

    print("\nMicrosoft/plain:")
    print("    Gij_linear = (1-gamma_PPN)*spatial_basis")
    print("    pressureless_target = 0")
    print("    gamma_PPN=1 makes Gij_linear = 0")

    print("\nNormal:")
    print("    The spatial Einstein-like components vanish only in the GR-like gamma=1 case.")
    print("    This complements Toy Model 7a, which tested the 00 component.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))

    print("\nWeak-field condition:")
    print("    max weak_field_parameter =", max_weak)
    print("    target < 0.05")

    print("\n00-component consistency:")
    print("    mean G00_metric_vs_source_Q for gamma>0 =", mean_G00_Q_positive_gamma)
    print("    mean G00_metric_vs_source_R2 for gamma>0 =", mean_G00_R2_positive_gamma)

    print("\nSpatial-component coefficient check:")
    print("    expected Gij coefficient = 1 - gamma_PPN")
    print("    gamma=0.0: component Q =", gamma_groups[0.0]["mean_Gij_component_Q"])
    print("    gamma=0.5: component Q =", gamma_groups[0.5]["mean_Gij_component_Q"])
    print("    gamma=1.0: component Q =", gamma_groups[1.0]["mean_Gij_component_Q"])

    print("\nSpatial trace check:")
    print("    gamma=0.0: trace Q =", gamma_groups[0.0]["mean_Gij_trace_Q"])
    print("    gamma=0.5: trace Q =", gamma_groups[0.5]["mean_Gij_trace_Q"])
    print("    gamma=1.0: trace Q =", gamma_groups[1.0]["mean_Gij_trace_Q"])

    print("\nSpatial norm check:")
    print("    gamma=0.0: norm Q =", gamma_groups[0.0]["mean_Gij_norm_Q"])
    print("    gamma=0.5: norm Q =", gamma_groups[0.5]["mean_Gij_norm_Q"])
    print("    gamma=1.0: norm Q =", gamma_groups[1.0]["mean_Gij_norm_Q"])

    print("\nGR-like pressureless residual check:")
    print("    gamma=1.0 max_abs_Gij =", gamma_groups[1.0]["max_abs_Gij"])
    print("    target approx 0")

    weak_pass = max_weak < 0.05
    cases_pass = pass_count == total_count
    G00_pass = abs(mean_G00_Q_positive_gamma - 1.0) < 1e-12 and mean_G00_R2_positive_gamma > 0.999999
    spatial_pass = (
        abs(gamma_groups[0.0]["mean_Gij_component_Q"] - 1.0) < 1e-12
        and abs(gamma_groups[0.5]["mean_Gij_component_Q"] - 0.5) < 1e-12
        and abs(gamma_groups[1.0]["mean_Gij_component_Q"] - 0.0) < 1e-12
    )
    trace_pass = (
        abs(gamma_groups[0.0]["mean_Gij_trace_Q"] - 1.0) < 1e-12
        and abs(gamma_groups[0.5]["mean_Gij_trace_Q"] - 0.5) < 1e-12
        and abs(gamma_groups[1.0]["mean_Gij_trace_Q"] - 0.0) < 1e-12
    )
    norm_pass = (
        abs(gamma_groups[0.0]["mean_Gij_norm_Q"] - 1.0) < 1e-12
        and abs(gamma_groups[0.5]["mean_Gij_norm_Q"] - 0.5) < 1e-12
        and abs(gamma_groups[1.0]["mean_Gij_norm_Q"] - 0.0) < 1e-12
    )
    pressureless_pass = gamma_groups[1.0]["max_abs_Gij"] < 1e-14

    print("\nWEAK-FIELD CONDITION:", "PASS" if weak_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")
    print("00 SOURCE-CURVATURE CONSISTENCY:", "PASS" if G00_pass else "CHECK")
    print("SPATIAL COMPONENT COEFFICIENT:", "PASS" if spatial_pass else "CHECK")
    print("SPATIAL TRACE RELATION:", "PASS" if trace_pass else "CHECK")
    print("SPATIAL NORM SCALING:", "PASS" if norm_pass else "CHECK")
    print("PRESSURELESS GAMMA=1 RESIDUAL:", "PASS" if pressureless_pass else "CHECK")

    if weak_pass and cases_pass and G00_pass and spatial_pass and trace_pass and norm_pass and pressureless_pass:
        print("\nRESULT: PASS")
        print("Toy Model 7b supports the spatial-component consistency bridge:")
        print("network source -> metric -> Gij_linear vanishes for pressureless matter only when gamma_PPN=1")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving on.")

    print("==============================================================")

    if rep_gamma05 is not None and rep_gamma1 is not None:
        make_plots(all_case_rows, rep_gamma05, rep_gamma1)


if __name__ == "__main__":
    main()
